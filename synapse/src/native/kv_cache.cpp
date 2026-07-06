#include "kv_cache.h"
#include <algorithm>

// CPU build (ik_llama.cpp) uses the legacy llama_kv_cache_* API.
// GPU build (upstream llama.cpp >= 2025-06) migrated to llama_memory_t handle:
//   llama_get_memory(ctx) → llama_memory_t
//   llama_memory_seq_rm/cp/keep/add/div/clear, llama_memory_seq_pos_min/max
// SUBVOCAL_UPSTREAM_LLAMA is set by CMakeLists.txt only for the GPU build.

#ifdef SUBVOCAL_UPSTREAM_LLAMA
// Helpers to reduce boilerplate in the upstream API path
#define KV_MEM(ctx) llama_get_memory(ctx)
#define KV_SEQ_RM(ctx, seq, p0, p1)          llama_memory_seq_rm (KV_MEM(ctx), seq, p0, p1)
#define KV_SEQ_CP(ctx, s, d, p0, p1)         llama_memory_seq_cp (KV_MEM(ctx), s, d, p0, p1)
#define KV_SEQ_KEEP(ctx, seq)                llama_memory_seq_keep(KV_MEM(ctx), seq)
#define KV_SEQ_ADD(ctx, seq, p0, p1, delta)  llama_memory_seq_add (KV_MEM(ctx), seq, p0, p1, delta)
#define KV_CLEAR(ctx)                        llama_memory_clear   (KV_MEM(ctx), true)
#define KV_POS_MIN(ctx, seq)                 llama_memory_seq_pos_min(KV_MEM(ctx), seq)
#define KV_POS_MAX(ctx, seq)                 llama_memory_seq_pos_max(KV_MEM(ctx), seq)
#else
#define KV_SEQ_RM(ctx, seq, p0, p1)          llama_kv_cache_seq_rm  (ctx, seq, p0, p1)
#define KV_SEQ_CP(ctx, s, d, p0, p1)         llama_kv_cache_seq_cp  (ctx, s, d, p0, p1)
#define KV_SEQ_KEEP(ctx, seq)                llama_kv_cache_seq_keep(ctx, seq)
#define KV_SEQ_ADD(ctx, seq, p0, p1, delta)  llama_kv_cache_seq_add (ctx, seq, p0, p1, delta)
#define KV_CLEAR(ctx)                        llama_kv_cache_clear   (ctx)
#define KV_POS_MIN(ctx, seq)                 llama_kv_cache_seq_pos_min(ctx, seq)
#define KV_POS_MAX(ctx, seq)                 llama_kv_cache_seq_pos_max(ctx, seq)
#endif

KVCacheManager::KVCacheManager(llama_context* ctx, uint32_t n_ctx, uint32_t num_attention_sinks)
    : ctx_(ctx), n_ctx_(n_ctx), attention_sinks_(num_attention_sinks) {}

void KVCacheManager::SetAttentionSinks(uint32_t count) { attention_sinks_ = count; }

uint32_t KVCacheManager::GetAttentionSinks() const { return attention_sinks_; }

void KVCacheManager::SeqRemove(int32_t seq_id, int32_t p0, int32_t p1) {
    KV_SEQ_RM(ctx_, seq_id, p0, p1);
}

void KVCacheManager::SeqCopy(int32_t src_seq, int32_t dst_seq, int32_t p0, int32_t p1) {
    KV_SEQ_CP(ctx_, src_seq, dst_seq, p0, p1);
}

void KVCacheManager::SeqKeep(int32_t seq_id) {
    KV_SEQ_KEEP(ctx_, seq_id);
}

void KVCacheManager::SeqShift(int32_t seq_id, int32_t p0, int32_t p1, int32_t delta) {
    KV_SEQ_ADD(ctx_, seq_id, p0, p1, delta);
}

void KVCacheManager::Clear() {
    KV_CLEAR(ctx_);
}

int32_t KVCacheManager::Fork(int32_t src_seq) {
    const int32_t new_seq_id = next_seq_id_++;
    KV_SEQ_CP(ctx_, src_seq, new_seq_id, -1, -1);
    return new_seq_id;
}

void KVCacheManager::Prune(int32_t seq_id) {
    KV_SEQ_RM(ctx_, seq_id, -1, -1);
}

void KVCacheManager::Evict(int32_t keep_last_n, int32_t seq_id) {
    const llama_pos max_pos = KV_POS_MAX(ctx_, seq_id);
    if (max_pos < 0) return;
    const int32_t evict_pos = std::max(0, static_cast<int32_t>(max_pos) - keep_last_n);
    const int32_t safe_evict = std::max(evict_pos, static_cast<int32_t>(attention_sinks_));
    if (safe_evict <= 0) return;
    SeqRemove(seq_id, 0, safe_evict);
    SeqShift(seq_id, safe_evict, -1, -safe_evict);
}

void KVCacheManager::EvictWithSinks(uint32_t keep_last_n, int32_t seq_id) {
    Evict(static_cast<int32_t>(keep_last_n), seq_id);
}

KVCacheManager::View KVCacheManager::GetView() {
    View result;
    result.max_size = n_ctx_;
    result.used = 0;

    const llama_pos pos_min = KV_POS_MIN(ctx_, 0);
    const llama_pos pos_max = KV_POS_MAX(ctx_, 0);
    if (pos_min < 0 || pos_max < 0 || pos_max < pos_min) return result;

    const int32_t count = static_cast<int32_t>(pos_max - pos_min + 1);
    result.cells.reserve(count);
    for (llama_pos p = pos_min; p <= pos_max; ++p) {
        CellInfo ci;
        ci.pos = static_cast<int32_t>(p);
        ci.seq_id = 0;
        ci.has_value = true;
        result.cells.push_back(ci);
    }
    result.used = static_cast<uint32_t>(count);
    return result;
}

uint32_t KVCacheManager::GetUsedCount() {
    return GetView().used;
}
