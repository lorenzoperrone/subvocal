// Subvocal FFI binder — GPU variant (modern llama.cpp upstream API)
// Static-linked to llama.cpp upstream + CUDA. Targets subvocal-large (Qwen 14B-A3B).
//
// API differences from binding_cpu.cpp (ik_llama legacy):
//   llama_free_model           -> llama_model_free
//   llama_n_vocab(model)       -> llama_vocab_n_tokens(llama_model_get_vocab(model))
//   llama_tokenize(model, ...)  -> llama_tokenize(vocab, ...)
//   llama_token_to_piece(model)-> llama_token_to_piece(vocab, ...)
//   llama_batch_get_one(t,n,p,s)-> llama_batch_get_one(t, n)  (only 2 args)
//   llama_kv_cache_clear(ctx)  -> llama_memory_clear(llama_get_memory(ctx), true)
//
// Class shape is identical to the CPU binder, so TS wrapper stays uniform.

#include <napi.h>
#include "llama.h"
#include "llama-ext.h"  // staging API: ctx_other / nextn embeddings, used for MTP (see LoadDraftModel)
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>

#include "suffix_tree_wrapper.h"

#include "logits_processor.h"
#include "kv_cache.h"
#include "git_binding.h"

class Model : public Napi::ObjectWrap<Model> {
 public:
  static Napi::Function Init(Napi::Env env);
  Model(const Napi::CallbackInfo& info);
  ~Model();

 private:
  Napi::Value Tokenize(const Napi::CallbackInfo& info);
  Napi::Value Forward(const Napi::CallbackInfo& info);
  Napi::Value ForwardAsync(const Napi::CallbackInfo& info);
  // V7: incremental decode — append tokens onto existing KV (no clear/re-prefill).
  Napi::Value DecodeAppend(const Napi::CallbackInfo& info);
  Napi::Value DecodeAppendAsync(const Napi::CallbackInfo& info);
  // M11.3 variant (b): decode onto an explicit KV sequence (not always seq 0) at an
  // explicit position, for out-of-band work (e.g. intent classification) that must be
  // attention-isolated from the live conversation on seq 0. Requires the model to have been
  // constructed with ModelOptions.auxSeq (n_seq_max=2, kv_unified=true) — see the constructor.
  // Deliberately does NOT touch n_past_ (that tracks seq 0 only); the caller tracks the aux
  // sequence's own position and cleans it up with the existing kvCacheSeqRemove(seqId, ...).
  Napi::Value DecodeAppendSeq(const Napi::CallbackInfo& info);
  Napi::Value ForwardEmbedding(const Napi::CallbackInfo& info);
  Napi::Value GetLogits(const Napi::CallbackInfo& info);
  // SUBVOCAL-PATCH (Mac port, 2026-06-28): ported from binding.cpp (cpu backend) — was
  // missing here, forcing JS callers to sort the full vocab themselves every token.
  Napi::Value GetLogitsTopK(const Napi::CallbackInfo& info);
  Napi::Value GetHiddenState(const Napi::CallbackInfo& info);
  Napi::Value GetHiddenStateLayer(const Napi::CallbackInfo& info);
  Napi::Value Detokenize(const Napi::CallbackInfo& info);
  Napi::Value VocabSize(const Napi::CallbackInfo& info);
  Napi::Value ContextSize(const Napi::CallbackInfo& info);
  Napi::Value EmbeddingSize(const Napi::CallbackInfo& info);
  Napi::Value LayerCount(const Napi::CallbackInfo& info);
  void Free(const Napi::CallbackInfo& info);

  // V3.2 + V4: see binding.cpp for full doc.
  static bool LayerHiddenCaptureCb(struct ggml_tensor* t, bool ask, void* user_data);
  static bool PartialAbortCb(void* user_data);
  Napi::Value ForwardPartial(const Napi::CallbackInfo& info);
  // V5: KV snapshot/restore — agent branching primitive
  Napi::Value GetKVState(const Napi::CallbackInfo& info);
  Napi::Value SetKVState(const Napi::CallbackInfo& info);

  // V6: zero-copy logits extraction
  Napi::Value GetLogitsFast(const Napi::CallbackInfo& info);
  Napi::Value GetLogitsUnsafe(const Napi::CallbackInfo& info);
  Napi::Value GetLogitsBatch(const Napi::CallbackInfo& info);
  // V6.1: logit bias / steering
  Napi::Value ApplyLogitBias(const Napi::CallbackInfo& info);
  Napi::Value SetPersistentBiases(const Napi::CallbackInfo& info);
  Napi::Value ClearLogitBiases(const Napi::CallbackInfo& info);
  Napi::Value SetSteeringVector(const Napi::CallbackInfo& info);
  Napi::Value ClearSteeringVector(const Napi::CallbackInfo& info);
  // V6.2: KV cache manipulation
  Napi::Value KVCacheSeqRemove(const Napi::CallbackInfo& info);
  Napi::Value KVCacheSeqCopy(const Napi::CallbackInfo& info);
  Napi::Value KVCacheSeqKeep(const Napi::CallbackInfo& info);
  Napi::Value KVCacheSeqShift(const Napi::CallbackInfo& info);
  Napi::Value KVCacheClear(const Napi::CallbackInfo& info);
  // V7.1: reset the internal n_past_ counter without touching the KV cache.
  // Call after kvCacheSeqRemove() to correct the position pointer so the next
  // decodeAppend() places tokens at the right sequence position.
  Napi::Value ResetNPast(const Napi::CallbackInfo& info);

  // V8: multi-context support — fork a new llama_context sharing model_ weights
  Napi::Value ForkContext(const Napi::CallbackInfo& info);

  // V9: token-to-pointer routing — action token callbacks
  Napi::Value RegisterActionToken(const Napi::CallbackInfo& info);
  Napi::Value RemoveActionToken(const Napi::CallbackInfo& info);
  Napi::Value HandleActionToken(const Napi::CallbackInfo& info);

  Napi::Value KVCacheFork(const Napi::CallbackInfo& info);
  Napi::Value KVCacheEvict(const Napi::CallbackInfo& info);
  Napi::Value KVCacheView(const Napi::CallbackInfo& info);
  // V6.3: suffix tree for self-speculative drafting
  Napi::Value SuffixTreeExtend(const Napi::CallbackInfo& info);
  Napi::Value SuffixTreeClear(const Napi::CallbackInfo& info);
  Napi::Value SuffixTreeSpeculate(const Napi::CallbackInfo& info);
  Napi::Value SuffixTreeTokenCount(const Napi::CallbackInfo& info);
  Napi::Value SuffixTreeMaxDepth(const Napi::CallbackInfo& info);

  // MTP speculative decoding (Mac port, 2026-06-28) — proof-of-concept scope, see
  // doc/research/mtp-speculative-decoding-scoping.md. Loads the MTP drafter GGUF as a
  // second context sharing this model's KV cache (llama_context_params.ctx_other),
  // then drives one draft step at a time via the target's h_nextn embeddings.
  // NOT YET a full batched draft+verify loop — this only proves the mechanism works
  // (load, embeddings_nextn extraction, draft-model decode) end to end.
  Napi::Value LoadDraftModel(const Napi::CallbackInfo& info);
  Napi::Value MtpDraftNext(const Napi::CallbackInfo& info);
  // Batched draft+verify (the actual speedup phase, see scoping doc's "What's NOT done
  // yet" — now done). MtpDraftChain produces up to N candidate tokens by repeatedly
  // re-decoding the SAME shared KV position, each pass conditioned on the previous pass's
  // own nextn output (not the target's — the target hasn't moved). MtpVerifyBatch submits
  // all N to the target in one decode, accepts up to the first mismatch, and always emits
  // one additional "correction" token (the target's own true prediction at the disagreement
  // point, or its next-token prediction if all N were accepted).
  Napi::Value MtpDraftChain(const Napi::CallbackInfo& info);
  Napi::Value MtpVerifyBatch(const Napi::CallbackInfo& info);

  llama_model* model_ = nullptr;
  llama_context* ctx_ = nullptr;
  // M15.7: optional LoRA adapter (GGUF) loaded on top of the base at construction, applied to
  // ctx_ with lora_scale_. Freed in the destructor. nullptr when no loraPath option was given.
  llama_adapter_lora* lora_adapter_ = nullptr;
  const llama_vocab* vocab_ = nullptr;
  int32_t n_vocab_ = 0;
  uint32_t n_ctx_ = 0;
  int32_t n_layer_ = 0;
  int32_t n_embd_ = 0;
  bool embeddings_enabled_ = false;
  bool capture_layers_ = false;
  // V7: number of tokens currently held in the KV cache (seq 0). forward()/
  // forwardEmbedding() reset it to the prompt length; decodeAppend() advances it;
  // KV clears reset it to 0. Used to position appended tokens correctly.
  int32_t n_past_ = 0;
  uint32_t n_threads_ = 4;
  uint32_t n_threads_batch_ = 4;
  std::vector<std::vector<float>> layer_hidden_;
   // V6: pre-allocated shadow buffer for getLogitsFast()
   std::vector<float> logits_shadow_;
   // V6.1: logit bias and steering processor
   LogitProcessor* logit_processor_ = nullptr;
   // V6.2: KV cache manager
   KVCacheManager* kv_cache_ = nullptr;
   // V6.3: suffix tree for self-speculative drafting
   common_suffix_tree* suffix_tree_ = nullptr;

   // V9: token-to-pointer routing — action token callbacks
   std::unordered_map<int, Napi::ThreadSafeFunction> action_callbacks_;

   std::atomic<int32_t> partial_target_layer_{-1};
  std::atomic<bool> partial_target_reached_{false};

  // MTP speculative decoding (proof-of-concept) — see LoadDraftModel/MtpDraftNext.
  llama_model* draft_model_ = nullptr;
  llama_context* draft_ctx_ = nullptr;
  int32_t draft_n_vocab_ = 0;
  int32_t draft_n_past_ = 0;

  friend class ForwardWorker;
  friend class DecodeAppendWorker;
  friend class DecodeAppendSeqWorker;
};

Napi::Function Model::Init(Napi::Env env) {
  return DefineClass(env, "Model", {
    InstanceMethod("tokenize", &Model::Tokenize),
    InstanceMethod("forward", &Model::Forward),
    InstanceMethod("forwardAsync", &Model::ForwardAsync),
    InstanceMethod("decodeAppend", &Model::DecodeAppend),
    InstanceMethod("decodeAppendAsync", &Model::DecodeAppendAsync),
    InstanceMethod("decodeAppendSeq", &Model::DecodeAppendSeq),
    InstanceMethod("forwardPartial", &Model::ForwardPartial),
    InstanceMethod("getKVState", &Model::GetKVState),
    InstanceMethod("setKVState", &Model::SetKVState),
    InstanceMethod("forwardEmbedding", &Model::ForwardEmbedding),
    InstanceMethod("getLogits", &Model::GetLogits),
    InstanceMethod("getLogitsTopK", &Model::GetLogitsTopK),
    InstanceMethod("getHiddenState", &Model::GetHiddenState),
    InstanceMethod("getHiddenStateLayer", &Model::GetHiddenStateLayer),
    InstanceMethod("detokenize", &Model::Detokenize),
    InstanceMethod("vocabSize", &Model::VocabSize),
    InstanceMethod("contextSize", &Model::ContextSize),
    InstanceMethod("embeddingSize", &Model::EmbeddingSize),
    InstanceMethod("layerCount", &Model::LayerCount),
    InstanceMethod("free", &Model::Free),
    // V6: zero-copy logits extraction
    InstanceMethod("getLogitsFast", &Model::GetLogitsFast),
    InstanceMethod("getLogitsUnsafe", &Model::GetLogitsUnsafe),
    InstanceMethod("getLogitsBatch", &Model::GetLogitsBatch),
    // V6.1: logit bias / steering
    InstanceMethod("applyLogitBias", &Model::ApplyLogitBias),
    InstanceMethod("setPersistentBiases", &Model::SetPersistentBiases),
    InstanceMethod("clearLogitBiases", &Model::ClearLogitBiases),
    InstanceMethod("setSteeringVector", &Model::SetSteeringVector),
    InstanceMethod("clearSteeringVector", &Model::ClearSteeringVector),
    // V6.2: KV cache manipulation
    InstanceMethod("kvCacheSeqRemove", &Model::KVCacheSeqRemove),
    InstanceMethod("kvCacheSeqCopy", &Model::KVCacheSeqCopy),
    InstanceMethod("kvCacheSeqKeep", &Model::KVCacheSeqKeep),
    InstanceMethod("kvCacheSeqShift", &Model::KVCacheSeqShift),
    InstanceMethod("kvCacheClear", &Model::KVCacheClear),
     InstanceMethod("resetNPast", &Model::ResetNPast),
     InstanceMethod("forkContext", &Model::ForkContext),
     // V9: token-to-pointer routing
     InstanceMethod("registerActionToken", &Model::RegisterActionToken),
     InstanceMethod("removeActionToken", &Model::RemoveActionToken),
     InstanceMethod("handleActionToken", &Model::HandleActionToken),
     InstanceMethod("kvCacheFork", &Model::KVCacheFork),
    InstanceMethod("kvCacheEvict", &Model::KVCacheEvict),
     InstanceMethod("kvCacheView", &Model::KVCacheView),
     // V6.3: suffix tree for self-speculative drafting
     InstanceMethod("suffixTreeExtend", &Model::SuffixTreeExtend),
     InstanceMethod("suffixTreeClear", &Model::SuffixTreeClear),
     InstanceMethod("suffixTreeSpeculate", &Model::SuffixTreeSpeculate),
     InstanceMethod("suffixTreeTokenCount", &Model::SuffixTreeTokenCount),
     InstanceMethod("suffixTreeMaxDepth", &Model::SuffixTreeMaxDepth),
     // MTP speculative decoding
     InstanceMethod("loadDraftModel", &Model::LoadDraftModel),
     InstanceMethod("mtpDraftNext", &Model::MtpDraftNext),
     InstanceMethod("mtpDraftChain", &Model::MtpDraftChain),
     InstanceMethod("mtpVerifyBatch", &Model::MtpVerifyBatch),
   });
}

bool Model::PartialAbortCb(void* user_data) {
  Model* self = static_cast<Model*>(user_data);
  if (self->partial_target_layer_.load(std::memory_order_relaxed) < 0) return false;
  return self->partial_target_reached_.load(std::memory_order_acquire);
}

bool Model::LayerHiddenCaptureCb(struct ggml_tensor* t, bool ask, void* user_data) {
  Model* self = static_cast<Model*>(user_data);
  if (!t || !t->name[0]) return false;
  if (std::strncmp(t->name, "l_out-", 6) != 0) return false;
  char* end = nullptr;
  long layer = std::strtol(t->name + 6, &end, 10);
  if (end == t->name + 6 || layer < 0 || layer >= self->n_layer_) return false;
  if (ask) return true;
  if (t->ne[0] != self->n_embd_) return true;
  const int64_t n_tokens = t->ne[1];
  if (n_tokens <= 0) return true;
  const size_t row_bytes = self->n_embd_ * sizeof(float);
  const size_t offset = static_cast<size_t>(n_tokens - 1) * row_bytes;
  ggml_backend_tensor_get(t, self->layer_hidden_[layer].data(), offset, row_bytes);
  // V4: signal partial-forward abort
  const int32_t target = self->partial_target_layer_.load(std::memory_order_relaxed);
  if (target >= 0 && layer >= target) {
    self->partial_target_reached_.store(true, std::memory_order_release);
  }
  return true;
}

Model::Model(const Napi::CallbackInfo& info) : Napi::ObjectWrap<Model>(info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsString()) {
    Napi::TypeError::New(env, "Model(path: string, opts?: {contextSize, threads, threadsBatch, gpuLayers})")
        .ThrowAsJavaScriptException();
    return;
  }

  std::string path = info[0].As<Napi::String>();
  if (path == "__subvocal_fork__") return;  // fork sentinel — fields populated by ForkContext

  uint32_t n_ctx = 2048;
  uint32_t n_threads = 4;
  uint32_t n_threads_batch = 0;
  int32_t n_gpu_layers = 0;
  bool embeddings = false;
  bool capture_layers = false;
  bool offload_kqv = true;  // default: KV in VRAM; set noKvOffload=true for RAM/ISWA split
  // KV-cache element type. Default f16. q8_0 halves KV memory but is throughput-NEGATIVE on
  // Metal (~45% slower at 4-6k ctx — per-token dequant costs more than the bandwidth saved;
  // benchmarked 2026-06-28). Keep f16 by default; the only reason to pass q8_0 is to fit 16k+
  // ctx in this Mini's Metal budget without OOM, accepting the speed hit. Requires flash
  // attention (auto-on by default) for the quantized V path.
  ggml_type kv_type = GGML_TYPE_F16;

  // M15.7: optional LoRA adapter (GGUF). Loaded after ctx_ init below.
  std::string lora_path;
  float lora_scale = 1.0f;
  // M11.3 variant (b): reserve a second, attention-isolated KV sequence (seq_id=1) alongside
  // the main conversation (seq_id=0), for out-of-band work like intent classification that
  // must not see or affect the live conversation. OFF by default (n_seq_max stays 1, matching
  // every model load before this option existed).
  bool aux_seq = false;

  if (info.Length() > 1 && info[1].IsObject()) {
    auto opts = info[1].As<Napi::Object>();
    if (opts.Has("loraPath")) lora_path = opts.Get("loraPath").As<Napi::String>();
    if (opts.Has("loraScale")) lora_scale = opts.Get("loraScale").As<Napi::Number>().FloatValue();
    if (opts.Has("contextSize")) n_ctx = opts.Get("contextSize").As<Napi::Number>().Uint32Value();
    if (opts.Has("threads")) n_threads = opts.Get("threads").As<Napi::Number>().Uint32Value();
    if (opts.Has("threadsBatch")) n_threads_batch = opts.Get("threadsBatch").As<Napi::Number>().Uint32Value();
    if (opts.Has("gpuLayers")) n_gpu_layers = opts.Get("gpuLayers").As<Napi::Number>().Int32Value();
    if (opts.Has("embeddings")) embeddings = opts.Get("embeddings").ToBoolean().Value();
    if (opts.Has("captureLayerHidden")) capture_layers = opts.Get("captureLayerHidden").ToBoolean().Value();
    if (opts.Has("noKvOffload")) offload_kqv = !opts.Get("noKvOffload").ToBoolean().Value();
    if (opts.Has("auxSeq")) aux_seq = opts.Get("auxSeq").ToBoolean().Value();
    if (opts.Has("kvType")) {
      std::string s = opts.Get("kvType").As<Napi::String>();
      if (s == "q8_0")      kv_type = GGML_TYPE_Q8_0;
      else if (s == "q5_1") kv_type = GGML_TYPE_Q5_1;
      else if (s == "q4_0") kv_type = GGML_TYPE_Q4_0;
      else                  kv_type = GGML_TYPE_F16;  // "f16" or anything unknown
    }
  }
  if (n_threads_batch == 0) n_threads_batch = n_threads;
  n_threads_ = n_threads;
  n_threads_batch_ = n_threads_batch;
  embeddings_enabled_ = embeddings;
  capture_layers_ = capture_layers;

  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = n_gpu_layers;

  model_ = llama_model_load_from_file(path.c_str(), mparams);
  if (!model_) {
    Napi::Error::New(env, "Failed to load model from " + path).ThrowAsJavaScriptException();
    return;
  }
  vocab_ = llama_model_get_vocab(model_);
  n_vocab_ = llama_vocab_n_tokens(vocab_);
  n_layer_ = llama_model_n_layer(model_);
  n_embd_ = llama_model_n_embd(model_);
  // V6: pre-allocate shadow buffer for getLogitsFast()
  logits_shadow_.resize(n_vocab_);

  // V6.1: initialize logit processor
  logit_processor_ = new LogitProcessor(n_vocab_);

  if (capture_layers_) {
    layer_hidden_.assign(n_layer_, std::vector<float>(n_embd_, 0.0f));
  }

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = n_ctx;
  // Leave n_batch / n_ubatch at library defaults for cache locality.
  cparams.n_threads = n_threads;
  cparams.n_threads_batch = n_threads_batch;
  cparams.embeddings = embeddings;
  cparams.offload_kqv = offload_kqv;
  cparams.type_k = kv_type;
  cparams.type_v = kv_type;
  if (capture_layers_) {
    cparams.cb_eval = &Model::LayerHiddenCaptureCb;
    cparams.cb_eval_user_data = this;
  }
  // M11.3 variant (b): n_seq_max=2 reserves a second sequence slot for decodeAppendSeq().
  // CRITICAL: llama.cpp partitions n_ctx ACROSS sequences by default (n_ctx_seq = n_ctx /
  // n_seq_max) — reserving a second slot WITHOUT kv_unified would silently HALVE every
  // conversation's usable window. kv_unified=true makes n_ctx_seq = n_ctx instead (each
  // sequence gets the full budget; verified against llama-context.cpp's cparams derivation
  // before writing this). Off by default — every model load before this option existed keeps
  // n_seq_max=1, kv_unified=false, byte-identical to today.
  if (aux_seq) {
    cparams.n_seq_max = 2;
    cparams.kv_unified = true;
  }
  // V4 abort callback registered after ctx init below.

  ctx_ = llama_init_from_model(model_, cparams);
  if (!ctx_) {
    llama_model_free(model_);
    model_ = nullptr;
    Napi::Error::New(env, "Failed to init context").ThrowAsJavaScriptException();
    return;
  }
  n_ctx_ = n_ctx;

   // M15.7: load + apply the optional LoRA adapter (dialect fine-tune) on top of the QAT base.
   // The adapter is a GGUF produced from a QLoRA train (see doc/substories/M15.7-stage2-lora-
   // recipe.md); loading it here keeps the base's QAT intact and just adds the delta.
   if (!lora_path.empty()) {
     lora_adapter_ = llama_adapter_lora_init(model_, lora_path.c_str());
     if (!lora_adapter_) {
       llama_free(ctx_); ctx_ = nullptr;
       llama_model_free(model_); model_ = nullptr;
       Napi::Error::New(env, "Failed to load LoRA adapter: " + lora_path).ThrowAsJavaScriptException();
       return;
     }
     llama_adapter_lora* adapters[1] = { lora_adapter_ };
     float scales[1] = { lora_scale };
     llama_set_adapters_lora(ctx_, adapters, 1, scales);
   }

   // V6.2: initialize KV cache manager
   kv_cache_ = new KVCacheManager(ctx_, n_ctx_);

   // V6.3: initialize suffix tree for self-speculative drafting
   suffix_tree_ = new common_suffix_tree(64);  // default max depth 64

   if (capture_layers_) {
    // V4: see binding.cpp — abort_cb is a no-op outside ForwardPartial.
    llama_set_abort_callback(ctx_, &Model::PartialAbortCb, this);
  }
}

Model::~Model() {
  delete logit_processor_;
  logit_processor_ = nullptr;
  delete kv_cache_;
  kv_cache_ = nullptr;
  delete suffix_tree_;
  suffix_tree_ = nullptr;
  for (auto& [id, tsfn] : action_callbacks_) {
    tsfn.Release();
  }
  action_callbacks_.clear();
  // Free the draft context/model first — it holds a ctx_other pointer into ctx_.
  if (draft_ctx_) { llama_free(draft_ctx_); draft_ctx_ = nullptr; }
  if (draft_model_) { llama_model_free(draft_model_); draft_model_ = nullptr; }
  // M15.7: free the LoRA adapter before the context/model it was bound to.
  if (lora_adapter_) { llama_adapter_lora_free(lora_adapter_); lora_adapter_ = nullptr; }
  if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
  if (model_) { llama_model_free(model_); model_ = nullptr; }
  vocab_ = nullptr;
}

void Model::Free(const Napi::CallbackInfo& info) {
  delete logit_processor_;
  logit_processor_ = nullptr;
  delete kv_cache_;
  kv_cache_ = nullptr;
  delete suffix_tree_;
  suffix_tree_ = nullptr;
  for (auto& [id, tsfn] : action_callbacks_) {
    tsfn.Release();
  }
  action_callbacks_.clear();
  if (draft_ctx_) { llama_free(draft_ctx_); draft_ctx_ = nullptr; }
  if (draft_model_) { llama_model_free(draft_model_); draft_model_ = nullptr; }
  if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
  if (model_) { llama_model_free(model_); model_ = nullptr; }
  vocab_ = nullptr;
}

Napi::Value Model::VocabSize(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), n_vocab_);
}

Napi::Value Model::ContextSize(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), n_ctx_);
}

Napi::Value Model::LayerCount(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), n_layer_);
}

Napi::Value Model::EmbeddingSize(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), n_embd_);
}

// V5: KV cache snapshot — see binding.cpp doc. Agent branching primitive.
Napi::Value Model::GetKVState(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  const size_t needed = llama_state_get_size(ctx_);
  auto buf = Napi::ArrayBuffer::New(env, needed);
  const size_t written = llama_state_get_data(ctx_,
      reinterpret_cast<uint8_t*>(buf.Data()), needed);
  if (written == 0 || written > needed) {
    Napi::Error::New(env, "llama_state_get_data failed").ThrowAsJavaScriptException();
    return env.Null();
  }
  return Napi::Uint8Array::New(env, written, buf, 0);
}
Napi::Value Model::SetKVState(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 1 || !info[0].IsTypedArray()) {
    Napi::TypeError::New(env, "setKVState(state: Uint8Array)").ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Uint8Array arr = info[0].As<Napi::Uint8Array>();
  const size_t read = llama_state_set_data(ctx_,
      reinterpret_cast<const uint8_t*>(arr.Data()), arr.ElementLength());
  // SUBVOCAL-PATCH (Mac port, 2026-06-28): llama_state_set_data() restores the KV cache
  // itself but NOT this binding's own n_past_ tracker — without this, the next
  // decodeAppend() would compute wrong positions (continuing to think n_past_ is whatever
  // it was before the restore, e.g. 0 for a freshly-constructed Model). Recover it from the
  // restored KV cache's own max position.
  if (read > 0) {
    const llama_pos max_pos = llama_memory_seq_pos_max(llama_get_memory(ctx_), 0);
    n_past_ = (max_pos >= 0) ? (int32_t)max_pos + 1 : 0;
  }
  return Napi::Number::New(env, (double)read);
}

// V4: forward but abort right after the chosen layer is computed (see binding.cpp doc).
Napi::Value Model::ForwardPartial(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!capture_layers_) {
    Napi::Error::New(env, "forwardPartial() requires { captureLayerHidden: true } at construction")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 2 || !info[0].IsTypedArray() || !info[1].IsNumber()) {
    Napi::TypeError::New(env, "forwardPartial(tokens: Int32Array, layerLimit: number)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Int32Array arr = info[0].As<Napi::Int32Array>();
  int32_t n_tokens = (int32_t)arr.ElementLength();
  int32_t layer_limit = info[1].As<Napi::Number>().Int32Value();
  if (layer_limit < 0 || layer_limit >= n_layer_) {
    Napi::RangeError::New(env, "layerLimit out of range [0, n_layer)").ThrowAsJavaScriptException();
    return env.Null();
  }
  if (n_tokens <= 0 || (uint32_t)n_tokens > n_ctx_) {
    Napi::Error::New(env, "tokens out of range").ThrowAsJavaScriptException();
    return env.Null();
  }

  llama_memory_clear(llama_get_memory(ctx_), true);

  // SUBVOCAL-PATCH-101 client-side: use the real build-time limit on GPU.
  // The abort_callback (smart approach) cannot stop CUDA mega-kernels in time;
  // the engine patch builds a physically smaller graph. layer_limit+1 because the
  // engine field counts LAYERS (1-based: limit=N means build layers 0..N-1),
  // while the JS API exposes a 0-based "last layer index".
  llama_set_n_layer_limit(ctx_, layer_limit + 1);

  // Keep the atomic flags armed too — harmless on GPU (abort never fires in time),
  // useful if anyone runs the same binder with a CPU backend in the future.
  partial_target_reached_.store(false, std::memory_order_relaxed);
  partial_target_layer_.store(layer_limit, std::memory_order_release);

  llama_token* tokens_ptr = reinterpret_cast<llama_token*>(arr.Data());
  constexpr int32_t CHUNK = 2048;
  int32_t final_status = 0;
  for (int32_t i = 0; i < n_tokens; i += CHUNK) {
    int32_t chunk_size = std::min(CHUNK, n_tokens - i);
    llama_batch batch = llama_batch_init(chunk_size, 0, 1);
    for (int32_t j = 0; j < chunk_size; j++) {
      batch.token[j] = tokens_ptr[i + j];
      batch.pos[j] = i + j;
      batch.n_seq_id[j] = 1;
      batch.seq_id[j][0] = 0;
      batch.logits[j] = (i + j == n_tokens - 1) ? 1 : 0;
    }
    batch.n_tokens = chunk_size;
    int32_t status = llama_decode(ctx_, batch);
    llama_batch_free(batch);
    if (status != 0 && status != 2) { final_status = status; break; }
  }

  // Reset both: next forward() should be full.
  llama_set_n_layer_limit(ctx_, 0);
  partial_target_layer_.store(-1, std::memory_order_release);
  partial_target_reached_.store(false, std::memory_order_relaxed);

  return Napi::Number::New(env, final_status);
}

Napi::Value Model::GetHiddenStateLayer(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!capture_layers_) {
    Napi::Error::New(env, "getHiddenStateLayer() requires { captureLayerHidden: true } at construction")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "getHiddenStateLayer(layer: number)").ThrowAsJavaScriptException();
    return env.Null();
  }
  int32_t layer = info[0].As<Napi::Number>().Int32Value();
  if (layer < 0 || layer >= n_layer_) {
    Napi::RangeError::New(env, "layer out of range [0, n_layer)").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto buf = Napi::ArrayBuffer::New(env, n_embd_ * sizeof(float));
  std::memcpy(buf.Data(), layer_hidden_[layer].data(), n_embd_ * sizeof(float));
  return Napi::Float32Array::New(env, n_embd_, buf, 0);
}

Napi::Value Model::GetHiddenState(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!embeddings_enabled_) {
    Napi::Error::New(env, "getHiddenState() requires constructor option { embeddings: true }")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  // Fallback chain — see binding.cpp for rationale.
  float* embd = llama_get_embeddings_seq(ctx_, 0);
  if (!embd) embd = llama_get_embeddings_ith(ctx_, -1);
  if (!embd) embd = llama_get_embeddings(ctx_);
  if (!embd) {
    Napi::Error::New(env, "Embeddings unavailable (call forward()/forwardEmbedding() first)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto buf = Napi::ArrayBuffer::New(env, n_embd_ * sizeof(float));
  std::memcpy(buf.Data(), embd, n_embd_ * sizeof(float));
  return Napi::Float32Array::New(env, n_embd_, buf, 0);
}

// V3: forward with a prefix embedding (one or more "fake tokens" supplied as
// raw float vectors instead of token IDs). Optional `tokens` Int32Array
// appended after the embedding prefix. Each embedding row must be exactly
// n_embd floats (Qwen 14B-A3B = 2048).
//
// This is the GPU-side "write" primitive of Subvocal's hidden-state passing:
// the small brain produces a hidden state, an adapter projects it to n_embd
// of the large brain, and we feed it here as if it were a token embedding.
Napi::Value Model::ForwardEmbedding(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 1 || !info[0].IsTypedArray()) {
    Napi::TypeError::New(env, "forwardEmbedding(embd: Float32Array, tokens?: Int32Array)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Float32Array embd_arr = info[0].As<Napi::Float32Array>();
  int32_t embd_floats = (int32_t)embd_arr.ElementLength();
  if (embd_floats % n_embd_ != 0) {
    Napi::Error::New(env, "embd length must be a multiple of n_embd (" + std::to_string(n_embd_) + ")")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  int32_t n_prefix = embd_floats / n_embd_;

  int32_t n_tokens_extra = 0;
  llama_token* tokens_ptr = nullptr;
  if (info.Length() > 1 && info[1].IsTypedArray()) {
    Napi::Int32Array t = info[1].As<Napi::Int32Array>();
    n_tokens_extra = (int32_t)t.ElementLength();
    tokens_ptr = reinterpret_cast<llama_token*>(t.Data());
  }
  int32_t total = n_prefix + n_tokens_extra;
  if (total <= 0 || (uint32_t)total > n_ctx_) {
    Napi::Error::New(env, "total tokens (prefix+tokens) out of range")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  llama_memory_clear(llama_get_memory(ctx_), true);

  // Decode the embedding prefix first.
  {
    llama_batch batch = llama_batch_init(n_prefix, n_embd_, 1);
    std::memcpy(batch.embd, embd_arr.Data(), n_prefix * n_embd_ * sizeof(float));
    for (int32_t i = 0; i < n_prefix; i++) {
      batch.pos[i] = i;
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      // Only set logits=true for the very last "slot" overall (prefix+tokens)
      batch.logits[i] = (n_tokens_extra == 0 && i == n_prefix - 1) ? 1 : 0;
    }
    batch.n_tokens = n_prefix;
    int32_t status = llama_decode(ctx_, batch);
    llama_batch_free(batch);
    if (status != 0) return Napi::Number::New(env, status);
  }

  // Decode the optional tokens after the prefix.
  if (n_tokens_extra > 0) {
    constexpr int32_t CHUNK = 2048;
    int32_t base_pos = n_prefix;
    for (int32_t i = 0; i < n_tokens_extra; i += CHUNK) {
      int32_t chunk_size = std::min(CHUNK, n_tokens_extra - i);
      llama_batch batch = llama_batch_init(chunk_size, 0, 1);
      for (int32_t j = 0; j < chunk_size; j++) {
        batch.token[j] = tokens_ptr[i + j];
        batch.pos[j] = base_pos + i + j;
        batch.n_seq_id[j] = 1;
        batch.seq_id[j][0] = 0;
        // Only set logits=true for the absolute last token
        batch.logits[j] = (i + j == n_tokens_extra - 1) ? 1 : 0;
      }
      batch.n_tokens = chunk_size;
      int32_t status = llama_decode(ctx_, batch);
      llama_batch_free(batch);
      if (status != 0) return Napi::Number::New(env, status);
    }
  }
  return Napi::Number::New(env, 0);
}

Napi::Value Model::Tokenize(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!vocab_) {
    Napi::Error::New(env, "Model has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 1 || !info[0].IsString()) {
    Napi::TypeError::New(env, "tokenize(text: string, addSpecial?: boolean)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  std::string text = info[0].As<Napi::String>();
  bool add_special = info.Length() > 1 ? info[1].ToBoolean().Value() : true;
  bool parse_special = info.Length() > 2 ? info[2].ToBoolean().Value() : true;

  std::vector<llama_token> tokens(text.size() + 16);
  int32_t n = llama_tokenize(vocab_, text.c_str(), (int32_t)text.size(),
                             tokens.data(), (int32_t)tokens.size(),
                             add_special, parse_special);
  if (n < 0) {
    tokens.resize(-n);
    n = llama_tokenize(vocab_, text.c_str(), (int32_t)text.size(),
                       tokens.data(), (int32_t)tokens.size(),
                       add_special, parse_special);
    if (n < 0) {
      Napi::Error::New(env, "Tokenization failed").ThrowAsJavaScriptException();
      return env.Null();
    }
  }

  auto buf = Napi::ArrayBuffer::New(env, n * sizeof(int32_t));
  std::memcpy(buf.Data(), tokens.data(), n * sizeof(int32_t));
  return Napi::Int32Array::New(env, n, buf, 0);
}

Napi::Value Model::Forward(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 1 || !info[0].IsTypedArray()) {
    Napi::TypeError::New(env, "forward(tokens: Int32Array)").ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Int32Array arr = info[0].As<Napi::Int32Array>();
  int32_t n_tokens = (int32_t)arr.ElementLength();
  if (n_tokens <= 0) {
    Napi::Error::New(env, "tokens must be non-empty").ThrowAsJavaScriptException();
    return env.Null();
  }
  if ((uint32_t)n_tokens > n_ctx_) {
    Napi::Error::New(env, "tokens exceeds context size").ThrowAsJavaScriptException();
    return env.Null();
  }

  // V1: stateless. Clear KV memory and prefill in chunks of n_batch (default 2048).
  // Use llama_batch_init + manual logits[last]=true for symmetry with the CPU
  // binder and to guarantee embeddings extraction when embeddings=true.
  llama_memory_clear(llama_get_memory(ctx_), true);

  llama_token* tokens_ptr = reinterpret_cast<llama_token*>(arr.Data());
  constexpr int32_t CHUNK = 2048;
  for (int32_t i = 0; i < n_tokens; i += CHUNK) {
    int32_t chunk_size = std::min(CHUNK, n_tokens - i);
    llama_batch batch = llama_batch_init(chunk_size, 0, 1);
    for (int32_t j = 0; j < chunk_size; j++) {
      batch.token[j] = tokens_ptr[i + j];
      batch.pos[j] = i + j;
      batch.n_seq_id[j] = 1;
      batch.seq_id[j][0] = 0;
      batch.logits[j] = (i + j == n_tokens - 1) ? 1 : 0;
    }
    batch.n_tokens = chunk_size;
    int32_t status = llama_decode(ctx_, batch);
    llama_batch_free(batch);
    if (status != 0) { n_past_ = 0; return Napi::Number::New(env, status); }
  }

  // V7: KV now holds positions 0..n_tokens-1; decodeAppend() continues from here.
  n_past_ = n_tokens;

  // V6.1: automatically apply biases + steering after decode if any are active
  if (logit_processor_ && (logit_processor_->HasBias() || logit_processor_->HasSteering())) {
    float* logits = llama_get_logits_ith(ctx_, -1);
    if (logits) {
      logit_processor_->Apply(logits);
    }
  }

  return Napi::Number::New(env, 0);
}

class ForwardWorker : public Napi::AsyncWorker {
 public:
  ForwardWorker(Napi::Env env, Model* model, const std::vector<int32_t>& tokens)
      : Napi::AsyncWorker(env), model_(model), tokens_(tokens),
        deferred_(Napi::Promise::Deferred::New(env)) {
    model_->Ref();
  }
  ~ForwardWorker() {
    model_->Unref();
  }

  void Execute() override {
    if (!model_->ctx_) {
      SetError("Context has been freed");
      return;
    }
    llama_memory_clear(llama_get_memory(model_->ctx_), true);

    llama_token* tokens_ptr = reinterpret_cast<llama_token*>(tokens_.data());
    int32_t n_tokens = (int32_t)tokens_.size();
    constexpr int32_t CHUNK = 2048;
    for (int32_t i = 0; i < n_tokens; i += CHUNK) {
      int32_t chunk_size = std::min(CHUNK, n_tokens - i);
      llama_batch batch = llama_batch_init(chunk_size, 0, 1);
      for (int32_t j = 0; j < chunk_size; j++) {
        batch.token[j] = tokens_ptr[i + j];
        batch.pos[j] = i + j;
        batch.n_seq_id[j] = 1;
        batch.seq_id[j][0] = 0;
        batch.logits[j] = (i + j == n_tokens - 1) ? 1 : 0;
      }
      batch.n_tokens = chunk_size;
      status_ = llama_decode(model_->ctx_, batch);
      llama_batch_free(batch);
      if (status_ != 0) { model_->n_past_ = 0; return; }
    }
    model_->n_past_ = n_tokens;
  }

  void OnOK() override {
    if (model_->ctx_ && model_->logit_processor_ && (model_->logit_processor_->HasBias() || model_->logit_processor_->HasSteering())) {
      float* logits = llama_get_logits_ith(model_->ctx_, -1);
      if (logits) {
        model_->logit_processor_->Apply(logits);
      }
    }
    deferred_.Resolve(Napi::Number::New(Env(), status_));
  }

  void OnError(const Napi::Error& e) override {
    deferred_.Reject(e.Value());
  }

  Napi::Promise GetPromise() { return deferred_.Promise(); }

 private:
  Model* model_;
  std::vector<int32_t> tokens_;
  int32_t status_ = 0;
  Napi::Promise::Deferred deferred_;
};

Napi::Value Model::ForwardAsync(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 1 || !info[0].IsTypedArray()) {
    Napi::TypeError::New(env, "forwardAsync(tokens: Int32Array)").ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Int32Array arr = info[0].As<Napi::Int32Array>();
  int32_t n_tokens = (int32_t)arr.ElementLength();
  if (n_tokens <= 0) {
    Napi::Error::New(env, "tokens must be non-empty").ThrowAsJavaScriptException();
    return env.Null();
  }
  if ((uint32_t)n_tokens > n_ctx_) {
    Napi::Error::New(env, "tokens exceeds context size").ThrowAsJavaScriptException();
    return env.Null();
  }

  std::vector<int32_t> tokens_vec(n_tokens);
  std::memcpy(tokens_vec.data(), arr.Data(), n_tokens * sizeof(int32_t));

  auto* worker = new ForwardWorker(env, this, std::move(tokens_vec));
  worker->Queue();
  return worker->GetPromise();
}

// V7: incremental decode. Append `tokens` onto the existing KV cache (seq 0)
// starting at position n_past_, WITHOUT clearing or re-prefilling. Logits are
// produced for the final appended token by default. When allLogits=true every
// token in the batch has logits computed — required for GPU speculative
// verification (getLogitsBatch on all draft positions).
Napi::Value Model::DecodeAppend(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 1 || !info[0].IsTypedArray()) {
    Napi::TypeError::New(env, "decodeAppend(tokens: Int32Array, allLogits?: boolean)").ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Int32Array arr = info[0].As<Napi::Int32Array>();
  int32_t n_tokens = (int32_t)arr.ElementLength();
  if (n_tokens <= 0) {
    Napi::Error::New(env, "tokens must be non-empty").ThrowAsJavaScriptException();
    return env.Null();
  }
  if ((uint32_t)(n_past_ + n_tokens) > n_ctx_) {
    Napi::Error::New(env, "decodeAppend would exceed context size").ThrowAsJavaScriptException();
    return env.Null();
  }
  bool all_logits = info.Length() > 1 ? info[1].ToBoolean().Value() : false;

  llama_token* tokens_ptr = reinterpret_cast<llama_token*>(arr.Data());
  constexpr int32_t CHUNK = 2048;
  for (int32_t i = 0; i < n_tokens; i += CHUNK) {
    int32_t chunk_size = std::min(CHUNK, n_tokens - i);
    llama_batch batch = llama_batch_init(chunk_size, 0, 1);
    for (int32_t j = 0; j < chunk_size; j++) {
      batch.token[j] = tokens_ptr[i + j];
      batch.pos[j] = n_past_ + i + j;
      batch.n_seq_id[j] = 1;
      batch.seq_id[j][0] = 0;
      batch.logits[j] = (all_logits || (i + j == n_tokens - 1)) ? 1 : 0;
    }
    batch.n_tokens = chunk_size;
    int32_t status = llama_decode(ctx_, batch);
    llama_batch_free(batch);
    if (status != 0) return Napi::Number::New(env, status);
  }

  n_past_ += n_tokens;

  if (logit_processor_ && (logit_processor_->HasBias() || logit_processor_->HasSteering())) {
    float* logits = llama_get_logits_ith(ctx_, -1);
    if (logits) {
      logit_processor_->Apply(logits);
    }
  }

  return Napi::Number::New(env, 0);
}

class DecodeAppendWorker : public Napi::AsyncWorker {
 public:
  DecodeAppendWorker(Napi::Env env, Model* model, const std::vector<int32_t>& tokens, bool all_logits)
      : Napi::AsyncWorker(env), model_(model), tokens_(tokens), all_logits_(all_logits),
        deferred_(Napi::Promise::Deferred::New(env)) {
    model_->Ref();
  }
  ~DecodeAppendWorker() {
    model_->Unref();
  }

  void Execute() override {
    if (!model_->ctx_) {
      SetError("Context has been freed");
      return;
    }
    llama_token* tokens_ptr = reinterpret_cast<llama_token*>(tokens_.data());
    int32_t n_tokens = (int32_t)tokens_.size();
    constexpr int32_t CHUNK = 2048;
    for (int32_t i = 0; i < n_tokens; i += CHUNK) {
      int32_t chunk_size = std::min(CHUNK, n_tokens - i);
      llama_batch batch = llama_batch_init(chunk_size, 0, 1);
      for (int32_t j = 0; j < chunk_size; j++) {
        batch.token[j] = tokens_ptr[i + j];
        batch.pos[j] = model_->n_past_ + i + j;
        batch.n_seq_id[j] = 1;
        batch.seq_id[j][0] = 0;
        batch.logits[j] = (all_logits_ || (i + j == n_tokens - 1)) ? 1 : 0;
      }
      batch.n_tokens = chunk_size;
      status_ = llama_decode(model_->ctx_, batch);
      llama_batch_free(batch);
      if (status_ != 0) return;
    }
    model_->n_past_ += n_tokens;
  }

  void OnOK() override {
    if (model_->ctx_ && model_->logit_processor_ && (model_->logit_processor_->HasBias() || model_->logit_processor_->HasSteering())) {
      float* logits = llama_get_logits_ith(model_->ctx_, -1);
      if (logits) {
        model_->logit_processor_->Apply(logits);
      }
    }
    deferred_.Resolve(Napi::Number::New(Env(), status_));
  }

  void OnError(const Napi::Error& e) override {
    deferred_.Reject(e.Value());
  }

  Napi::Promise GetPromise() { return deferred_.Promise(); }

 private:
  Model* model_;
  std::vector<int32_t> tokens_;
  bool all_logits_;
  int32_t status_ = 0;
  Napi::Promise::Deferred deferred_;
};

Napi::Value Model::DecodeAppendAsync(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 1 || !info[0].IsTypedArray()) {
    Napi::TypeError::New(env, "decodeAppendAsync(tokens: Int32Array, allLogits?: boolean)").ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Int32Array arr = info[0].As<Napi::Int32Array>();
  int32_t n_tokens = (int32_t)arr.ElementLength();
  if (n_tokens <= 0) {
    Napi::Error::New(env, "tokens must be non-empty").ThrowAsJavaScriptException();
    return env.Null();
  }
  if ((uint32_t)(n_past_ + n_tokens) > n_ctx_) {
    Napi::Error::New(env, "decodeAppendAsync would exceed context size").ThrowAsJavaScriptException();
    return env.Null();
  }
  bool all_logits = info.Length() > 1 ? info[1].ToBoolean().Value() : false;

  std::vector<int32_t> tokens_vec(n_tokens);
  std::memcpy(tokens_vec.data(), arr.Data(), n_tokens * sizeof(int32_t));

  auto* worker = new DecodeAppendWorker(env, this, std::move(tokens_vec), all_logits);
  worker->Queue();
  return worker->GetPromise();
}

// M11.3 variant (b): same shape as DecodeAppendWorker, but the seq_id and position are
// caller-supplied instead of hardcoded to seq 0 / n_past_. Requires ModelOptions.auxSeq at
// construction (n_seq_max=2, kv_unified=true) — otherwise llama_decode rejects seq_id=1 as
// out of range. Deliberately does NOT touch model_->n_past_: that tracks seq 0's position,
// and this decode runs on a DIFFERENT sequence that must not perturb it.
class DecodeAppendSeqWorker : public Napi::AsyncWorker {
 public:
  DecodeAppendSeqWorker(Napi::Env env, Model* model, std::vector<int32_t> tokens,
                        int32_t seq_id, int32_t pos_base, bool all_logits)
      : Napi::AsyncWorker(env), model_(model), tokens_(std::move(tokens)), seq_id_(seq_id),
        pos_base_(pos_base), all_logits_(all_logits),
        deferred_(Napi::Promise::Deferred::New(env)) {
    model_->Ref();
  }
  ~DecodeAppendSeqWorker() {
    model_->Unref();
  }

  void Execute() override {
    if (!model_->ctx_) {
      SetError("Context has been freed");
      return;
    }
    llama_token* tokens_ptr = reinterpret_cast<llama_token*>(tokens_.data());
    int32_t n_tokens = (int32_t)tokens_.size();
    constexpr int32_t CHUNK = 2048;
    for (int32_t i = 0; i < n_tokens; i += CHUNK) {
      int32_t chunk_size = std::min(CHUNK, n_tokens - i);
      llama_batch batch = llama_batch_init(chunk_size, 0, 1);
      for (int32_t j = 0; j < chunk_size; j++) {
        batch.token[j] = tokens_ptr[i + j];
        batch.pos[j] = pos_base_ + i + j;
        batch.n_seq_id[j] = 1;
        batch.seq_id[j][0] = seq_id_;
        batch.logits[j] = (all_logits_ || (i + j == n_tokens - 1)) ? 1 : 0;
      }
      batch.n_tokens = chunk_size;
      status_ = llama_decode(model_->ctx_, batch);
      llama_batch_free(batch);
      if (status_ != 0) return;
    }
    // n_past_ intentionally untouched — see class comment.
  }

  void OnOK() override { deferred_.Resolve(Napi::Number::New(Env(), status_)); }
  void OnError(const Napi::Error& e) override { deferred_.Reject(e.Value()); }
  Napi::Promise GetPromise() { return deferred_.Promise(); }

 private:
  Model* model_;
  std::vector<int32_t> tokens_;
  int32_t seq_id_;
  int32_t pos_base_;
  bool all_logits_;
  int32_t status_ = 0;
  Napi::Promise::Deferred deferred_;
};

Napi::Value Model::DecodeAppendSeq(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 3 || !info[0].IsTypedArray() || !info[1].IsNumber() || !info[2].IsNumber()) {
    Napi::TypeError::New(env, "decodeAppendSeq(tokens: Int32Array, seqId: number, pos: number, allLogits?: boolean)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Int32Array arr = info[0].As<Napi::Int32Array>();
  int32_t n_tokens = (int32_t)arr.ElementLength();
  if (n_tokens <= 0) {
    Napi::Error::New(env, "tokens must be non-empty").ThrowAsJavaScriptException();
    return env.Null();
  }
  int32_t seq_id = info[1].As<Napi::Number>().Int32Value();
  int32_t pos = info[2].As<Napi::Number>().Int32Value();
  if ((uint32_t)(pos + n_tokens) > n_ctx_) {
    Napi::Error::New(env, "decodeAppendSeq would exceed context size").ThrowAsJavaScriptException();
    return env.Null();
  }
  bool all_logits = info.Length() > 3 ? info[3].ToBoolean().Value() : false;

  std::vector<int32_t> tokens_vec(n_tokens);
  std::memcpy(tokens_vec.data(), arr.Data(), n_tokens * sizeof(int32_t));

  auto* worker = new DecodeAppendSeqWorker(env, this, std::move(tokens_vec), seq_id, pos, all_logits);
  worker->Queue();
  return worker->GetPromise();
}

Napi::Value Model::GetLogits(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }

  float* logits = llama_get_logits_ith(ctx_, -1);
  if (!logits) {
    Napi::Error::New(env, "Logits unavailable (call forward() first)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto buf = Napi::ArrayBuffer::New(env, n_vocab_ * sizeof(float));
  std::memcpy(buf.Data(), logits, n_vocab_ * sizeof(float));
  return Napi::Float32Array::New(env, n_vocab_, buf, 0);
}

// SUBVOCAL-PATCH (Mac port, 2026-06-28): ported verbatim from binding.cpp (cpu backend) —
// see that file for the original. Avoids forcing JS callers to sort/scan the full
// (262144-entry on Gemma4) logits vector themselves for top-k sampling every token.
Napi::Value Model::GetLogitsTopK(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  int32_t k = info.Length() > 0 && info[0].IsNumber() ? info[0].As<Napi::Number>().Int32Value() : 20;
  if (k <= 0 || k > n_vocab_) k = n_vocab_;

  float* logits = llama_get_logits_ith(ctx_, -1);
  if (!logits) {
    Napi::Error::New(env, "Logits unavailable (call forward() first)").ThrowAsJavaScriptException();
    return env.Null();
  }

  // Linear scan: O(vocab) but only copies k pairs (<100 bytes vs 600KB+ full vector).
  std::vector<std::pair<int32_t, float>> heap;
  heap.reserve(k);
  for (int32_t i = 0; i < n_vocab_; i++) {
    if ((int32_t)heap.size() < k) {
      heap.push_back({i, logits[i]});
      if ((int32_t)heap.size() == k) {
        std::make_heap(heap.begin(), heap.end(),
          [](auto& a, auto& b) { return a.second < b.second; });
      }
    } else if (logits[i] > heap.front().second) {
      std::pop_heap(heap.begin(), heap.end(),
        [](auto& a, auto& b) { return a.second < b.second; });
      heap.back() = {i, logits[i]};
      std::push_heap(heap.begin(), heap.end(),
        [](auto& a, auto& b) { return a.second < b.second; });
    }
  }

  // Sort descending by logit.
  std::sort(heap.begin(), heap.end(),
    [](auto& a, auto& b) { return a.second > b.second; });

  // Pack as [id0..idK-1, logitBits0..logitBitsK-1] (2*K int32).
  // Logit bits are IEEE 754 float reinterpreted as int32 via memcpy — safe and
  // lossless. Caller reads float half as: new Float32Array(result.buffer, k*4, k)
  const int32_t out_len = k * 2;
  Napi::Int32Array out = Napi::Int32Array::New(env, out_len);
  int32_t* ptr = reinterpret_cast<int32_t*>(out.Data());
  for (int32_t i = 0; i < k; i++) {
    ptr[i] = heap[i].first;
    std::memcpy(&ptr[k + i], &heap[i].second, sizeof(float));
  }
  return out;
}

// SUBVOCAL-PATCH (Mac port, 2026-06-28): MTP speculative decoding, proof-of-concept.
// See doc/research/mtp-speculative-decoding-scoping.md for the full design discussion.
//
// Loads the MTP drafter GGUF (e.g. mtp-gemma-4-12B-it.gguf) as a second llama_context
// that SHARES this model's KV cache via llama_context_params.ctx_other — confirmed via
// src/llama-context.cpp:104-121 (Gemma4Assistant/gemma4 archs require ctx_other) and
// src/llama-ext.h's llama_get_ctx_other(). This is NOT an independent model: the drafter's
// forward pass reads the target's "h_nextn" hidden state (llama_get_embeddings_nextn_ith)
// instead of token embeddings for its own input.
Napi::Value Model::LoadDraftModel(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_) {
    Napi::Error::New(env, "Target context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 1 || !info[0].IsString()) {
    Napi::TypeError::New(env, "loadDraftModel(path: string)").ThrowAsJavaScriptException();
    return env.Null();
  }
  std::string path = info[0].As<Napi::String>();

  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = 999;  // drafter is tiny relative to the target — offload fully
  draft_model_ = llama_model_load_from_file(path.c_str(), mparams);
  if (!draft_model_) {
    Napi::Error::New(env, "Failed to load draft model from " + path).ThrowAsJavaScriptException();
    return env.Null();
  }
  draft_n_vocab_ = llama_vocab_n_tokens(llama_model_get_vocab(draft_model_));

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = n_ctx_;
  cparams.n_threads = n_threads_;
  cparams.n_threads_batch = n_threads_batch_;
  // NOTE: deliberately NOT setting cparams.embeddings = true here. That flag means
  // "produce pooled sentence embeddings" (classic embedding-model output), a different
  // mechanism from MTP's h_nextn — the gemma4-assistant draft architecture doesn't build a
  // pooling graph, so embeddings=true crashes with "missing result_norm/result_embd tensor"
  // in build_pooling(). The nextn mechanism is enabled separately below, after context init.
  cparams.ctx_other = ctx_;  // share the target's KV cache (gemma4 MTP requires this)

  draft_ctx_ = llama_init_from_model(draft_model_, cparams);
  if (!draft_ctx_) {
    llama_model_free(draft_model_);
    draft_model_ = nullptr;
    Napi::Error::New(env, "Failed to init draft context (ctx_other wiring failed?)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  // masked=true for the target too (not the unmasked dense-by-raw-position mode a full
  // batched-verify implementation would use): get_embeddings_nextn_ith(ctx, -1) only
  // supports negative ("last") indexing through output_resolve_row(), which is the masked
  // path. Unmasked rejects negative i outright (see llama-context.cpp's get_embeddings_nextn_ith).
  // For this proof-of-concept we only ever need h_nextn at the last decoded position anyway.
  llama_set_embeddings_nextn(ctx_, true, /*masked*/ true);
  llama_set_embeddings_nextn(draft_ctx_, true, /*masked*/ true);

  draft_n_past_ = 0;
  return Napi::Boolean::New(env, true);
}

// One MTP draft step: read the target's h_nextn for its last decoded position, feed it to
// the drafter as an embedding-input batch (not tokens — see ForwardEmbedding above for the
// same batch-construction pattern), decode on draft_ctx_, return the drafter's top
// candidate token id + logit packed the same way as getLogitsTopK (k=1: [id, logitBits]).
//
// PROOF-OF-CONCEPT SCOPE: this does one unbatched draft prediction. It does NOT yet do the
// batched multi-token verify-against-target step that would make this actually faster than
// normal decodeAppend — see the scoping doc's "Decision" section. Useful today only to
// confirm the drafter loads, shares KV correctly, and produces sane (non-garbage) predictions.
Napi::Value Model::MtpDraftNext(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_ || !draft_ctx_) {
    Napi::Error::New(env, "Call loadDraftModel() first").ThrowAsJavaScriptException();
    return env.Null();
  }

  float* h_nextn = llama_get_embeddings_nextn_ith(ctx_, -1);
  if (!h_nextn) {
    Napi::Error::New(env, "h_nextn unavailable (call forward()/decodeAppend() on the "
                          "target first; embeddings_nextn must be enabled)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  // Position: since the draft context shares the target's KV cache (ctx_other), the
  // draft step's position must align with where the TARGET is, not an independent
  // counter. h_nextn comes from the target's last decoded position (n_past_ - 1); the
  // drafter predicts the token that would occupy n_past_ — same slot the target's own
  // next normal decode would use next.
  llama_batch batch = llama_batch_init(1, n_embd_, 1);
  std::memcpy(batch.embd, h_nextn, n_embd_ * sizeof(float));
  batch.pos[0] = n_past_;
  batch.n_seq_id[0] = 1;
  batch.seq_id[0][0] = 0;
  batch.logits[0] = 1;
  batch.n_tokens = 1;
  int32_t status = llama_decode(draft_ctx_, batch);
  llama_batch_free(batch);
  if (status != 0) {
    Napi::Error::New(env, "draft decode failed, status=" + std::to_string(status))
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  float* logits = llama_get_logits_ith(draft_ctx_, -1);
  if (!logits) {
    Napi::Error::New(env, "draft logits unavailable").ThrowAsJavaScriptException();
    return env.Null();
  }
  int32_t best_id = 0;
  float best_logit = logits[0];
  for (int32_t i = 1; i < draft_n_vocab_; i++) {
    if (logits[i] > best_logit) { best_logit = logits[i]; best_id = i; }
  }

  Napi::Int32Array out = Napi::Int32Array::New(env, 2);
  int32_t* ptr = reinterpret_cast<int32_t*>(out.Data());
  ptr[0] = best_id;
  std::memcpy(&ptr[1], &best_logit, sizeof(float));
  return out;
}

// Produces up to `maxTokens` draft candidates without ever advancing the shared KV position.
// Ref: common/speculative.cpp's common_speculative_impl_draft_mtp::draft(), the `is_mem_shared`
// branch (gemma4) — "with shared memory we use the same position for all draft tokens": each
// pass re-decodes draft_ctx_ at batch.pos[0] = n_past_ (never incremented), but feeds the
// PREVIOUS pass's own nextn output as the new input embedding instead of the target's h_nextn
// (only the first pass uses the target's). This lets the model effectively "look ahead"
// multiple tokens by repeated self-conditioning at one KV slot, rather than needing one KV
// slot per speculative token.
Napi::Value Model::MtpDraftChain(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_ || !draft_ctx_) {
    Napi::Error::New(env, "Call loadDraftModel() first").ThrowAsJavaScriptException();
    return env.Null();
  }
  int32_t max_tokens = info.Length() > 0 && info[0].IsNumber()
      ? info[0].As<Napi::Number>().Int32Value() : 4;
  if (max_tokens <= 0) max_tokens = 1;

  float* h_row = llama_get_embeddings_nextn_ith(ctx_, -1);
  if (!h_row) {
    Napi::Error::New(env, "h_nextn unavailable (call forward()/decodeAppend() on the target "
                          "first)").ThrowAsJavaScriptException();
    return env.Null();
  }

  std::vector<int32_t> tokens;
  std::vector<float> token_logits;
  tokens.reserve(max_tokens);
  token_logits.reserve(max_tokens);

  for (int32_t step = 0; step < max_tokens; step++) {
    llama_batch batch = llama_batch_init(1, n_embd_, 1);
    std::memcpy(batch.embd, h_row, n_embd_ * sizeof(float));
    batch.pos[0] = n_past_;  // fixed — see comment above
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;
    batch.n_tokens = 1;
    int32_t status = llama_decode(draft_ctx_, batch);
    llama_batch_free(batch);
    if (status != 0) break;  // keep whatever was drafted so far rather than failing outright

    float* logits = llama_get_logits_ith(draft_ctx_, -1);
    if (!logits) break;
    int32_t best_id = 0;
    float best_logit = logits[0];
    for (int32_t i = 1; i < draft_n_vocab_; i++) {
      if (logits[i] > best_logit) { best_logit = logits[i]; best_id = i; }
    }
    tokens.push_back(best_id);
    token_logits.push_back(best_logit);

    // Next pass conditions on THIS pass's own nextn output, not the target's.
    h_row = llama_get_embeddings_nextn_ith(draft_ctx_, -1);
    if (!h_row) break;
  }

  if (tokens.empty()) {
    Napi::Error::New(env, "draft chain produced zero tokens").ThrowAsJavaScriptException();
    return env.Null();
  }

  const int32_t k = (int32_t)tokens.size();
  Napi::Int32Array out = Napi::Int32Array::New(env, k * 2);
  int32_t* ptr = reinterpret_cast<int32_t*>(out.Data());
  for (int32_t i = 0; i < k; i++) {
    ptr[i] = tokens[i];
    std::memcpy(&ptr[k + i], &token_logits[i], sizeof(float));
  }
  return out;
}

// Verifies a chain of draft tokens against the target in ONE batched decode (this is the
// step that actually amortizes the bandwidth cost — see doc/research/m1-metal-benchmark.md).
// Accepts tokens up to the first mismatch between the target's own greedy prediction and the
// corresponding draft token, then always appends one "correction" token: the target's true
// prediction at the disagreement point (or, if every draft token was accepted, the target's
// next-token prediction following the last one). KV is truncated to discard any rejected
// positions before the correction token is decoded.
//
// Returns: Int32Array [acceptedCount, ...emittedTokenIds] where emittedTokenIds has
// acceptedCount+1 entries (the accepted draft tokens, then the correction token) — this is
// the actual generation output for the round, always at least 1 new token.
Napi::Value Model::MtpVerifyBatch(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 1 || !info[0].IsTypedArray()) {
    Napi::TypeError::New(env, "mtpVerifyBatch(draftTokens: Int32Array)").ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Int32Array draft = info[0].As<Napi::Int32Array>();
  const int32_t k = (int32_t)draft.ElementLength();
  if (k <= 0) {
    Napi::Error::New(env, "draftTokens must be non-empty").ThrowAsJavaScriptException();
    return env.Null();
  }
  if ((uint32_t)(n_past_ + k) > n_ctx_) {
    Napi::Error::New(env, "verify batch would exceed context size").ThrowAsJavaScriptException();
    return env.Null();
  }

  // "Pre-batch" prediction: what the target already predicts comes next, from its last
  // decode (before this verify batch) — this is the candidate for draft[0].
  float* pre_logits = llama_get_logits_ith(ctx_, -1);
  if (!pre_logits) {
    Napi::Error::New(env, "target logits unavailable (call forward() first)").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto argmax = [this](const float* logits) -> int32_t {
    int32_t best_id = 0;
    float best_logit = logits[0];
    for (int32_t i = 1; i < n_vocab_; i++) {
      if (logits[i] > best_logit) { best_logit = logits[i]; best_id = i; }
    }
    return best_id;
  };

  const int32_t first_target_pred = argmax(pre_logits);

  llama_token* draft_ptr = reinterpret_cast<llama_token*>(draft.Data());
  llama_batch batch = llama_batch_init(k, 0, 1);
  for (int32_t i = 0; i < k; i++) {
    batch.token[i] = draft_ptr[i];
    batch.pos[i] = n_past_ + i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = 1;  // need every position's prediction to verify the next draft token
  }
  batch.n_tokens = k;
  int32_t status = llama_decode(ctx_, batch);
  llama_batch_free(batch);
  if (status != 0) {
    Napi::Error::New(env, "verify decode failed, status=" + std::to_string(status))
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  // Accept up to the first mismatch. accepted = number of draft tokens confirmed correct.
  // pred_for(i) = the target's prediction for draft[i], using pre_logits (the target's
  // last decode before this batch) for i=0, or this batch's per-position output for i>0.
  int32_t accepted = 0;
  int32_t correction = first_target_pred;
  while (accepted < k) {
    const int32_t pred = (accepted == 0) ? first_target_pred
                                          : argmax(llama_get_logits_ith(ctx_, accepted - 1));
    if (pred != draft_ptr[accepted]) {
      correction = pred;
      break;
    }
    accepted++;
  }
  if (accepted == k) {
    // Every draft token was confirmed correct — the correction is a genuinely new token:
    // the target's prediction for what comes after the last accepted one.
    correction = argmax(llama_get_logits_ith(ctx_, k - 1));
  }
  // accepted draft tokens [0, accepted) are confirmed correct and already in the target's KV
  // (written by the verify batch above). Anything from `accepted` onward in the KV is wrong
  // and must be discarded before the correction token is decoded into that slot.
  if (accepted < k) {
    llama_memory_seq_rm(llama_get_memory(ctx_), 0, n_past_ + accepted, -1);
  }
  n_past_ += accepted;

  // Decode the correction token as a fresh single-token append at the now-truncated position.
  llama_batch cbatch = llama_batch_init(1, 0, 1);
  cbatch.token[0] = correction;
  cbatch.pos[0] = n_past_;
  cbatch.n_seq_id[0] = 1;
  cbatch.seq_id[0][0] = 0;
  cbatch.logits[0] = 1;
  cbatch.n_tokens = 1;
  status = llama_decode(ctx_, cbatch);
  llama_batch_free(cbatch);
  if (status != 0) {
    Napi::Error::New(env, "correction decode failed, status=" + std::to_string(status))
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  n_past_ += 1;

  Napi::Int32Array out = Napi::Int32Array::New(env, accepted + 2);
  int32_t* ptr = reinterpret_cast<int32_t*>(out.Data());
  ptr[0] = accepted;
  for (int32_t i = 0; i < accepted; i++) ptr[1 + i] = draft_ptr[i];
  ptr[1 + accepted] = correction;
  return out;
}

Napi::Value Model::Detokenize(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!vocab_) {
    Napi::Error::New(env, "Model has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 1 || !info[0].IsTypedArray()) {
    Napi::TypeError::New(env, "detokenize(tokens: Int32Array)").ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Int32Array arr = info[0].As<Napi::Int32Array>();
  int32_t n = (int32_t)arr.ElementLength();
  llama_token* tokens = reinterpret_cast<llama_token*>(arr.Data());

  std::string out;
  out.reserve(n * 4);
  char buf[256];
  for (int32_t i = 0; i < n; ++i) {
    int32_t len = llama_token_to_piece(vocab_, tokens[i], buf, sizeof(buf), 0, false);
    if (len > 0) {
      out.append(buf, len);
    } else if (len < 0) {
      std::vector<char> big(-len + 1);
      int32_t real = llama_token_to_piece(vocab_, tokens[i], big.data(),
                                          (int32_t)big.size(), 0, false);
      if (real > 0) out.append(big.data(), real);
    }
  }
  return Napi::String::New(env, out);
}

// ============================================================================
// V6: Zero-Copy Logits Extraction
// ============================================================================

Napi::Value Model::GetLogitsFast(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }

  float* logits = llama_get_logits_ith(ctx_, -1);
  if (!logits) {
    Napi::Error::New(env, "Logits unavailable (call forward() first)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  std::memcpy(logits_shadow_.data(), logits, n_vocab_ * sizeof(float));

  auto buf = Napi::ArrayBuffer::New(env, logits_shadow_.data(),
                                    n_vocab_ * sizeof(float),
                                    [](Napi::Env, void*) { /* no-op */ });
  return Napi::Float32Array::New(env, n_vocab_, buf, 0);
}

Napi::Value Model::GetLogitsUnsafe(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }

  float* logits = llama_get_logits_ith(ctx_, -1);
  if (!logits) {
    Napi::Error::New(env, "Logits unavailable (call forward() first)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto buf = Napi::ArrayBuffer::New(env, logits, n_vocab_ * sizeof(float),
                                    [](Napi::Env, void*) { /* no-op */ });
  return Napi::Float32Array::New(env, n_vocab_, buf, 0);
}

Napi::Value Model::GetLogitsBatch(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 1 || !info[0].IsTypedArray()) {
    Napi::TypeError::New(env, "getLogitsBatch(indices: Int32Array)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  Napi::Int32Array indices = info[0].As<Napi::Int32Array>();
  int32_t n = (int32_t)indices.ElementLength();
  Napi::Array result = Napi::Array::New(env, n);

  for (int32_t i = 0; i < n; i++) {
    int32_t ith = indices[i];
    float* logits = llama_get_logits_ith(ctx_, ith);
    if (!logits) {
      Napi::Error::New(env, "Logits unavailable for position " + std::to_string(ith))
          .ThrowAsJavaScriptException();
      return env.Null();
    }

    auto buf = Napi::ArrayBuffer::New(env, n_vocab_ * sizeof(float));
    std::memcpy(buf.Data(), logits, n_vocab_ * sizeof(float));
    result[i] = Napi::Float32Array::New(env, n_vocab_, buf, 0);
  }

  return result;
}

// ============================================================================
// V6.1: Logit Bias / Steering
// ============================================================================

Napi::Value Model::ApplyLogitBias(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 1 || !info[0].IsArray()) {
    Napi::TypeError::New(env, "applyLogitBias(biases: Array<{tokenId: number, bias: number}>)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Array arr = info[0].As<Napi::Array>();
  uint32_t len = arr.Length();
  std::vector<BiasEntry> entries;
  entries.reserve(len);
  for (uint32_t i = 0; i < len; i++) {
    Napi::Object entry = arr.Get(i).As<Napi::Object>();
    BiasEntry be;
    be.token_id = entry.Get("tokenId").As<Napi::Number>().Int32Value();
    be.bias = entry.Get("bias").As<Napi::Number>().FloatValue();
    entries.push_back(be);
  }
  logit_processor_->SetPendingBiases(entries.data(), entries.size());

  float* logits = llama_get_logits_ith(ctx_, -1);
  if (logits) {
    logit_processor_->Apply(logits);
  }
  return env.Undefined();
}

Napi::Value Model::SetPersistentBiases(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 1 || !info[0].IsArray()) {
    Napi::TypeError::New(env, "setPersistentBiases(biases: Array<{tokenId: number, bias: number}>)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Array arr = info[0].As<Napi::Array>();
  uint32_t len = arr.Length();
  std::vector<BiasEntry> entries;
  entries.reserve(len);
  for (uint32_t i = 0; i < len; i++) {
    Napi::Object entry = arr.Get(i).As<Napi::Object>();
    BiasEntry be;
    be.token_id = entry.Get("tokenId").As<Napi::Number>().Int32Value();
    be.bias = entry.Get("bias").As<Napi::Number>().FloatValue();
    entries.push_back(be);
  }
  logit_processor_->SetPersistentBiases(entries.data(), entries.size());
  return env.Undefined();
}

Napi::Value Model::ClearLogitBiases(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  logit_processor_->ClearAllBiases();
  return env.Undefined();
}

Napi::Value Model::SetSteeringVector(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 2 || !info[0].IsTypedArray() || !info[1].IsNumber()) {
    Napi::TypeError::New(env, "setSteeringVector(vector: Float32Array, strength: number)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Float32Array arr = info[0].As<Napi::Float32Array>();
  float strength = info[1].As<Napi::Number>().FloatValue();
  logit_processor_->SetSteeringVector(arr.Data(), arr.ElementLength(), strength);
  return env.Undefined();
}

Napi::Value Model::ClearSteeringVector(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  logit_processor_->ClearSteeringVector();
  return env.Undefined();
}

// ============================================================================
// V6.2: KV Cache Manipulation
// ============================================================================

Napi::Value Model::KVCacheSeqRemove(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_ || !kv_cache_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  int32_t seq_id = info[0].As<Napi::Number>().Int32Value();
  int32_t p0 = info[1].As<Napi::Number>().Int32Value();
  int32_t p1 = info[2].As<Napi::Number>().Int32Value();
  kv_cache_->SeqRemove(seq_id, p0, p1);
  return env.Undefined();
}

Napi::Value Model::KVCacheSeqCopy(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_ || !kv_cache_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  int32_t src_seq = info[0].As<Napi::Number>().Int32Value();
  int32_t dst_seq = info[1].As<Napi::Number>().Int32Value();
  int32_t p0 = info[2].As<Napi::Number>().Int32Value();
  int32_t p1 = info[3].As<Napi::Number>().Int32Value();
  kv_cache_->SeqCopy(src_seq, dst_seq, p0, p1);
  return env.Undefined();
}

Napi::Value Model::KVCacheSeqKeep(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_ || !kv_cache_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  int32_t seq_id = info[0].As<Napi::Number>().Int32Value();
  kv_cache_->SeqKeep(seq_id);
  return env.Undefined();
}

Napi::Value Model::KVCacheSeqShift(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_ || !kv_cache_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  int32_t seq_id = info[0].As<Napi::Number>().Int32Value();
  int32_t p0 = info[1].As<Napi::Number>().Int32Value();
  int32_t p1 = info[2].As<Napi::Number>().Int32Value();
  int32_t delta = info[3].As<Napi::Number>().Int32Value();
  kv_cache_->SeqShift(seq_id, p0, p1, delta);
  return env.Undefined();
}

Napi::Value Model::KVCacheClear(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_ || !kv_cache_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  kv_cache_->Clear();
  n_past_ = 0;  // V7: KV emptied → next decodeAppend starts at position 0.
  return env.Undefined();
}

// V7.1: reposition the n_past_ pointer WITHOUT touching the KV cache.
// Use after kvCacheSeqRemove() to evict tail tokens: the KV slots are gone but
// n_past_ still points past them. resetNPast(newLen) corrects the pointer so
// the next decodeAppend() deposits tokens at the right sequence position.
Napi::Value Model::ResetNPast(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "resetNPast(n: number)").ThrowAsJavaScriptException();
    return env.Null();
  }
  int32_t n = info[0].As<Napi::Number>().Int32Value();
  if (n < 0) n = 0;
  n_past_ = n;
  return env.Undefined();
}

Napi::Value Model::KVCacheFork(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_ || !kv_cache_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  int32_t src_seq = info[0].As<Napi::Number>().Int32Value();
  int32_t new_seq = kv_cache_->Fork(src_seq);
  return Napi::Number::New(env, new_seq);
}

Napi::Value Model::KVCacheEvict(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_ || !kv_cache_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  int32_t keep_last_n = info[0].As<Napi::Number>().Int32Value();
  int32_t seq_id = info.Length() > 1 ? info[1].As<Napi::Number>().Int32Value() : 0;
  kv_cache_->Evict(keep_last_n, seq_id);
  return env.Undefined();
}

Napi::Value Model::KVCacheView(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_ || !kv_cache_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  auto view = kv_cache_->GetView();
  Napi::Object result = Napi::Object::New(env);
  result.Set("maxSize", Napi::Number::New(env, view.max_size));
  result.Set("used", Napi::Number::New(env, view.used));
  Napi::Array cells = Napi::Array::New(env, view.cells.size());
  for (size_t i = 0; i < view.cells.size(); i++) {
    Napi::Object cell = Napi::Object::New(env);
    cell.Set("pos", Napi::Number::New(env, view.cells[i].pos));
    cell.Set("seqId", Napi::Number::New(env, view.cells[i].seq_id));
    cell.Set("hasValue", Napi::Boolean::New(env, view.cells[i].has_value));
    cells.Set(i, cell);
  }
  result.Set("cells", cells);
  return result;
}

Napi::Value Model::ForkContext(const Napi::CallbackInfo& info) {
  Napi::Error::New(info.Env(), "forkContext not supported on GPU backend").ThrowAsJavaScriptException();
  return info.Env().Undefined();
}

// ============================================================================
// V9: Token-to-Pointer Routing — Action Token Callbacks
// ============================================================================

Napi::Value Model::RegisterActionToken(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() < 2 || !info[0].IsNumber() || !info[1].IsFunction()) {
    Napi::TypeError::New(env, "registerActionToken(tokenId: number, callback: function)")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  int token_id = info[0].As<Napi::Number>().Int32Value();
  auto callback = info[1].As<Napi::Function>();
  auto tsfn = Napi::ThreadSafeFunction::New(
      env, callback, "action_callback", 0, 1);
  action_callbacks_[token_id] = tsfn;
  return env.Undefined();
}

Napi::Value Model::RemoveActionToken(const Napi::CallbackInfo& info) {
  int token_id = info[0].As<Napi::Number>().Int32Value();
  auto it = action_callbacks_.find(token_id);
  if (it != action_callbacks_.end()) {
    it->second.Release();
    action_callbacks_.erase(it);
  }
  return info.Env().Undefined();
}

Napi::Value Model::HandleActionToken(const Napi::CallbackInfo& info) {
  int token_id = info[0].As<Napi::Number>().Int32Value();
  auto it = action_callbacks_.find(token_id);
  if (it != action_callbacks_.end()) {
    auto cb = [](Napi::Env env, Napi::Function jsCb, int* tokenPtr) {
      jsCb.Call({Napi::Number::New(env, *tokenPtr)});
      delete tokenPtr;
    };
    int* token_copy = new int(token_id);
    it->second.NonBlockingCall(token_copy, cb);
    return Napi::Boolean::New(info.Env(), true);
  }
  return Napi::Boolean::New(info.Env(), false);
}

// ============================================================================
// V6.3: Suffix Tree for Self-Speculative Drafting
// ============================================================================

Napi::Value Model::SuffixTreeExtend(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!suffix_tree_) {
    Napi::Error::New(env, "SuffixTree has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 1 || !info[0].IsTypedArray()) {
    Napi::TypeError::New(env, "suffixTreeExtend(tokens: Int32Array)").ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Int32Array arr = info[0].As<Napi::Int32Array>();
  int32_t n = (int32_t)arr.ElementLength();
  if (n > 0) {
    suffix_tree_->extend(reinterpret_cast<llama_token*>(arr.Data()), n);
  }
  return env.Undefined();
}

Napi::Value Model::SuffixTreeClear(const Napi::CallbackInfo& info) {
  if (!suffix_tree_) {
    Napi::Env env = info.Env();
    Napi::Error::New(env, "SuffixTree has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  suffix_tree_->clear();
  return info.Env().Undefined();
}

Napi::Value Model::SuffixTreeSpeculate(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!suffix_tree_) {
    Napi::Error::New(env, "SuffixTree has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 1 || !info[0].IsTypedArray()) {
    Napi::TypeError::New(env,
      "suffixTreeSpeculate(context: Int32Array, maxSpecTokens?: number, minTokenProb?: number, minMatchCount?: number, minMatchLen?: number)"
    ).ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Int32Array context = info[0].As<Napi::Int32Array>();
  int max_spec_tokens = 7;
  float min_token_prob = 0.1f;
  int min_match_count  = 1;
  int min_match_len    = 5;

  if (info.Length() > 1 && info[1].IsNumber()) {
    max_spec_tokens = info[1].As<Napi::Number>().Int32Value();
  }
  if (info.Length() > 2 && info[2].IsNumber()) {
    min_token_prob = info[2].As<Napi::Number>().FloatValue();
  }
  if (info.Length() > 3 && info[3].IsNumber()) {
    min_match_count = info[3].As<Napi::Number>().Int32Value();
  }
  if (info.Length() > 4 && info[4].IsNumber()) {
    min_match_len = info[4].As<Napi::Number>().Int32Value();
  }

  std::vector<llama_token> draft = suffix_tree_->speculate(
    reinterpret_cast<llama_token*>(context.Data()),
    static_cast<int>(context.ElementLength()),
    max_spec_tokens,
    min_token_prob,
    min_match_count,
    min_match_len
  );

  size_t n_out = draft.size();
  if (n_out == 0) {
    return Napi::Int32Array::New(env, 0);
  }
  Napi::Int32Array out = Napi::Int32Array::New(env, n_out);
  std::memcpy(out.Data(), draft.data(), n_out * sizeof(int32_t));
  return out;
}

Napi::Value Model::SuffixTreeTokenCount(const Napi::CallbackInfo& info) {
  if (!suffix_tree_) {
    Napi::Env env = info.Env();
    Napi::Error::New(env, "SuffixTree has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  return Napi::Number::New(info.Env(), suffix_tree_->token_count());
}

Napi::Value Model::SuffixTreeMaxDepth(const Napi::CallbackInfo& info) {
  if (!suffix_tree_) {
    Napi::Env env = info.Env();
    Napi::Error::New(env, "SuffixTree has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  return Napi::Number::New(info.Env(), suffix_tree_->max_depth());
}

// ---------------------------------------------------------------------------
// Module init / unload

static bool g_backend_initialized = false;

// SUBVOCAL: quiet the very verbose ggml/llama loading + Metal-pipeline logs by default. They
// flood the TUI (kernel_mul_mm_*, flash_attn compile lines, KV cache sizes, etc.) and are only
// useful for debugging. Set SUBVOCAL_DEBUG=1 to restore full native logging; otherwise only
// genuine errors get through. Installed before llama_backend_init() so even init logs are caught.
static void subvocal_log_callback(ggml_log_level level, const char* text, void* /*user_data*/) {
  static const bool debug = std::getenv("SUBVOCAL_DEBUG") != nullptr;
  if (debug) {
    std::fputs(text, stderr);
    return;
  }
  // Non-debug: only surface real errors (still on stderr, never stdout — stdout is the TUI's).
  if (level == GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr);
}

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
  if (!g_backend_initialized) {
    llama_log_set(subvocal_log_callback, nullptr);
    ggml_log_set(subvocal_log_callback, nullptr);
    llama_backend_init();
    g_backend_initialized = true;
    napi_add_env_cleanup_hook(env, [](void*) { llama_backend_free(); }, nullptr);
  }
  exports.Set("Model", Model::Init(env));
  exports.Set("SuffixTree", SuffixTree::Init(env));
#ifdef SUBVOCAL_HAS_LIBGIT2
  // SUBVOCAL-PATCH (Mac port, 2026-06-28): this call was unguarded — git_binding.cpp is only
  // compiled in when CMake finds libgit2, so without it subvocal::InitGit was an undefined
  // symbol. With -undefined dynamic_lookup (required for Node's N-API symbols) the linker
  // doesn't catch that at link time; it resolves to a null function pointer at runtime instead,
  // crashing every require() with SIGSEGV at address 0x0. Found and fixed while bringing up the
  // Metal backend, where libgit2 isn't installed by default — same latent bug exists in
  // binding.cpp (cpu backend), fixed there too.
  subvocal::InitGit(env, exports);
#endif
  return exports;
}

NODE_API_MODULE(subvocal_ffi_gpu, InitAll)
