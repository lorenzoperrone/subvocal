#ifndef KV_CACHE_H
#define KV_CACHE_H

#include <vector>
#include <cstdint>
#include "llama.h"

class KVCacheManager {
public:
    KVCacheManager(llama_context* ctx, uint32_t n_ctx, uint32_t num_attention_sinks = 10);
    ~KVCacheManager() = default;
    void SetAttentionSinks(uint32_t count);
    uint32_t GetAttentionSinks() const;
    void EvictWithSinks(uint32_t keep_last_n, int32_t seq_id = 0);

    void SeqRemove(int32_t seq_id, int32_t p0, int32_t p1);
    void SeqCopy(int32_t src_seq, int32_t dst_seq, int32_t p0, int32_t p1);
    void SeqKeep(int32_t seq_id);
    void SeqShift(int32_t seq_id, int32_t p0, int32_t p1, int32_t delta);
    void Clear();

    int32_t Fork(int32_t src_seq);
    void Prune(int32_t seq_id);
    void Evict(int32_t keep_last_n, int32_t seq_id = 0);

    struct CellInfo {
        int32_t pos;
        int32_t seq_id;
        bool has_value;
    };

    struct View {
        uint32_t max_size;
        uint32_t used;
        std::vector<CellInfo> cells;
    };

    View GetView();
    uint32_t GetUsedCount();

private:
    llama_context* ctx_;
    uint32_t n_ctx_;
    uint32_t attention_sinks_ = 10;
    int32_t next_seq_id_ = 1;
};

#endif
