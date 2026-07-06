// Subvocal FFI binder — V1: load / tokenize / forward / getLogits / detokenize / free
// V3: getHiddenState() (last token, post-final-layer)
// V3.2: getHiddenStateLayer(layer) — per-layer hidden state via cb_eval callback,
//       intercepts tensors named "l_out-<layer>" during llama_decode.
//       NOT a patch to the engine: uses the public ggml_backend_sched_set_eval_callback API.
// V6: zero-copy logits extraction — getLogitsFast() (memcpy into stable shadow buffer),
//     getLogitsUnsafe() (true zero-copy, direct pointer to llama internal buffer),
//     getLogitsBatch() (multi-position logits for speculative decoding / validation).
// V6.1: Logit bias/steering — applyLogitBias(), setPersistentBiases(), setSteeringVector()
// V6.2: KV cache manipulation — kvCacheSeqRemove/Copy/Keep/Shift/Clear/Fork/Evict/View
// Backend: ik_llama.cpp (CPU). GPU backend is a follow-up task.
//
// Lifecycle: llama_backend_init() called once on module load. Model wraps both
// llama_model* and llama_context* (one ctx per model — V1 simplification).
// forward() clears the KV cache and prefills the given tokens from position 0,
// so each call is independent. Stateful KV management comes in V2.

#include <napi.h>
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <queue>
#include <limits>
#include <cmath>
#include <unordered_map>

#include "logits_processor.h"
#include "kv_cache.h"
#include "suffix_tree_wrapper.h"
#include "text_scanner.h"
#include "git_binding.h"

class Model : public Napi::ObjectWrap<Model> {
  public:
   static Napi::Function Init(Napi::Env env);
   static Napi::FunctionReference constructorRef;
   Model(const Napi::CallbackInfo& info);
  ~Model();

 private:
  Napi::Value Tokenize(const Napi::CallbackInfo& info);
  Napi::Value Forward(const Napi::CallbackInfo& info);
  // V7: incremental decode — append tokens onto existing KV (no clear/re-prefill).
  Napi::Value DecodeAppend(const Napi::CallbackInfo& info);
  Napi::Value GetLogits(const Napi::CallbackInfo& info);
  // V6: zero-copy logits extraction
  Napi::Value GetLogitsFast(const Napi::CallbackInfo& info);
  Napi::Value GetLogitsUnsafe(const Napi::CallbackInfo& info);
  Napi::Value GetLogitsBatch(const Napi::CallbackInfo& info);
  Napi::Value GetHiddenState(const Napi::CallbackInfo& info);
  Napi::Value GetHiddenStateLayer(const Napi::CallbackInfo& info);
  Napi::Value Detokenize(const Napi::CallbackInfo& info);
  Napi::Value VocabSize(const Napi::CallbackInfo& info);
  Napi::Value ContextSize(const Napi::CallbackInfo& info);
  Napi::Value EmbeddingSize(const Napi::CallbackInfo& info);
  Napi::Value LayerCount(const Napi::CallbackInfo& info);
  void Free(const Napi::CallbackInfo& info);

  // V3.2: per-layer hidden state capture via cb_eval.
  // Only captures the LAST TOKEN of each "l_out-<layer>" tensor during decode.
  // Allocated once in constructor if capture_layers=true, dim = n_layer * n_embd.
  static int LayerHiddenCaptureCb(struct ggml_tensor* t, bool ask, void* user_data);
  // V4: abort callback that fires during eval, returns true once partial-forward target is reached
  static bool PartialAbortCb(void* user_data);
  Napi::Value ForwardPartial(const Napi::CallbackInfo& info);
  // V5: KV cache snapshot/restore — agent backtracking / multi-branch exploration
  Napi::Value GetKVState(const Napi::CallbackInfo& info);
  Napi::Value SetKVState(const Napi::CallbackInfo& info);

  // V6.1: logit bias / steering
  Napi::Value ApplyLogitBias(const Napi::CallbackInfo& info);
  Napi::Value SetPersistentBiases(const Napi::CallbackInfo& info);
  Napi::Value ClearLogitBiases(const Napi::CallbackInfo& info);
  Napi::Value SetSteeringVector(const Napi::CallbackInfo& info);
  Napi::Value ClearSteeringVector(const Napi::CallbackInfo& info);

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

  // V6.2: KV cache manipulation
  Napi::Value KVCacheSeqRemove(const Napi::CallbackInfo& info);
  Napi::Value KVCacheSeqCopy(const Napi::CallbackInfo& info);
  Napi::Value KVCacheSeqKeep(const Napi::CallbackInfo& info);
  Napi::Value KVCacheSeqShift(const Napi::CallbackInfo& info);
  Napi::Value KVCacheClear(const Napi::CallbackInfo& info);
  Napi::Value KVCacheFork(const Napi::CallbackInfo& info);
  Napi::Value KVCacheEvict(const Napi::CallbackInfo& info);
  Napi::Value KVCacheView(const Napi::CallbackInfo& info);

  // V6.3: suffix tree for self-speculative drafting
  Napi::Value SuffixTreeExtend(const Napi::CallbackInfo& info);
  Napi::Value SuffixTreeClear(const Napi::CallbackInfo& info);
  Napi::Value SuffixTreeSpeculate(const Napi::CallbackInfo& info);
  Napi::Value SuffixTreeTokenCount(const Napi::CallbackInfo& info);
  Napi::Value SuffixTreeMaxDepth(const Napi::CallbackInfo& info);

  // V6.3: AST-aware logit masking
  Napi::Value SetASTTokenMask(const Napi::CallbackInfo& info);
  Napi::Value ClearASTMask(const Napi::CallbackInfo& info);

  // V6.4: Sparse logits — top-K extraction (<100 bytes instead of ~600KB)
  Napi::Value GetLogitsTopK(const Napi::CallbackInfo& info);
  Napi::Value ApplyMaskSIMD(const Napi::CallbackInfo& info);

  llama_model* model_ = nullptr;
  llama_context* ctx_ = nullptr;
  int32_t n_vocab_ = 0;
  uint32_t n_ctx_ = 0;
  int32_t n_embd_ = 0;
  int32_t n_layer_ = 0;
  bool embeddings_enabled_ = false;
  bool capture_layers_ = false;
  std::vector<std::vector<float>> layer_hidden_; // [n_layer][n_embd], last-token row of each l_out-<il>
  // V6: pre-allocated shadow buffer for getLogitsFast() — avoids per-call malloc
  std::vector<float> logits_shadow_;
  // V6.1: logit bias and steering processor
  LogitProcessor* logit_processor_ = nullptr;
  // V6.2: KV cache manager
  KVCacheManager* kv_cache_ = nullptr;
  // V7: tokens currently held in KV cache (seq 0). forward() resets to n_tokens;
  // decodeAppend() advances it; KV clears reset to 0.
  int32_t n_past_ = 0;
  uint32_t n_threads_ = 4;
  uint32_t n_threads_batch_ = 4;
  // V6.3: suffix tree for self-speculative drafting
  common_suffix_tree* suffix_tree_ = nullptr;

  // V9: token-to-pointer routing — action token callbacks
  std::unordered_map<int, Napi::ThreadSafeFunction> action_callbacks_;

  // V4 partial-forward state
  std::atomic<int32_t> partial_target_layer_{-1};   // -1 = no partial active
  std::atomic<bool> partial_target_reached_{false}; // set by cb_eval, read by abort_cb
};
Napi::FunctionReference Model::constructorRef;

Napi::Function Model::Init(Napi::Env env) {
  auto ctor = DefineClass(env, "Model", {
    InstanceMethod("tokenize", &Model::Tokenize),
    InstanceMethod("forward", &Model::Forward),
    InstanceMethod("decodeAppend", &Model::DecodeAppend),
    InstanceMethod("forwardPartial", &Model::ForwardPartial),
    InstanceMethod("getKVState", &Model::GetKVState),
    InstanceMethod("setKVState", &Model::SetKVState),
    InstanceMethod("getLogits", &Model::GetLogits),
    InstanceMethod("getLogitsFast", &Model::GetLogitsFast),
    InstanceMethod("getLogitsUnsafe", &Model::GetLogitsUnsafe),
    InstanceMethod("getLogitsBatch", &Model::GetLogitsBatch),
    InstanceMethod("getHiddenState", &Model::GetHiddenState),
    InstanceMethod("getHiddenStateLayer", &Model::GetHiddenStateLayer),
    InstanceMethod("detokenize", &Model::Detokenize),
    InstanceMethod("vocabSize", &Model::VocabSize),
    InstanceMethod("contextSize", &Model::ContextSize),
    InstanceMethod("embeddingSize", &Model::EmbeddingSize),
    InstanceMethod("layerCount", &Model::LayerCount),
    InstanceMethod("free", &Model::Free),
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
    InstanceMethod("kvCacheFork", &Model::KVCacheFork),
    InstanceMethod("kvCacheEvict", &Model::KVCacheEvict),
     InstanceMethod("kvCacheView", &Model::KVCacheView),
      InstanceMethod("resetNPast", &Model::ResetNPast),
      InstanceMethod("forkContext", &Model::ForkContext),
    // V9: token-to-pointer routing
    InstanceMethod("registerActionToken", &Model::RegisterActionToken),
    InstanceMethod("removeActionToken", &Model::RemoveActionToken),
    InstanceMethod("handleActionToken", &Model::HandleActionToken),
    // V6.3: suffix tree for self-speculative drafting
   InstanceMethod("suffixTreeExtend", &Model::SuffixTreeExtend),
   InstanceMethod("suffixTreeClear", &Model::SuffixTreeClear),
   InstanceMethod("suffixTreeSpeculate", &Model::SuffixTreeSpeculate),
   InstanceMethod("suffixTreeTokenCount", &Model::SuffixTreeTokenCount),
   InstanceMethod("suffixTreeMaxDepth", &Model::SuffixTreeMaxDepth),
// V6.3: AST-aware logit masking
    InstanceMethod("setASTTokenMask", &Model::SetASTTokenMask),
    InstanceMethod("clearASTMask", &Model::ClearASTMask),
    // V6.4: Sparse logits
     InstanceMethod("getLogitsTopK", &Model::GetLogitsTopK),
      InstanceMethod("applyMaskSIMD", &Model::ApplyMaskSIMD),
   });
   constructorRef = Napi::Persistent(ctor);
   constructorRef.SuppressDestruct();
   return ctor;
}

// V7: incremental decode. Appends `tokens` onto the KV cache left by the
// previous forward()/decodeAppend() without clearing or re-prefilling, and
// produces logits for the last appended token. Use it to drive an
// autoregressive loop in O(n) instead of O(n^2): forward(prompt) once, then
// decodeAppend([nextToken]) per step. Returns the llama_decode status (0 = ok).
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
  // When allLogits=true every token in the batch has logits computed — required
  // for GPU speculative verification (getLogitsBatch on all draft positions).
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

  // V6.3: apply AST mask + V6.1: biases/steering after each append step
  if (logit_processor_ && logit_processor_->HasASTMask()) {
    float* logits = llama_get_logits_ith(ctx_, -1);
    if (logits) logit_processor_->ApplyASTMask(logits);
  }
  if (logit_processor_ && (logit_processor_->HasBias() || logit_processor_->HasSteering())) {
    float* logits = llama_get_logits_ith(ctx_, -1);
    if (logits) logit_processor_->Apply(logits);
  }

  return Napi::Number::New(env, 0);
}

// V4: returns true to abort the in-flight llama_decode().
// Called regularly by ggml during eval. Cheap atomic load.
bool Model::PartialAbortCb(void* user_data) {
  Model* self = static_cast<Model*>(user_data);
  // No partial active → never abort (normal forward path)
  if (self->partial_target_layer_.load(std::memory_order_relaxed) < 0) return false;
  return self->partial_target_reached_.load(std::memory_order_acquire);
}

// V3.2: pure C++ callback fired by ggml_backend_sched during eval.
// Two-phase: ask=true means "do you want notification when this tensor is computed?"
// → return true to be called again with ask=false + actual data.
int Model::LayerHiddenCaptureCb(struct ggml_tensor* t, bool ask, void* user_data) {
  Model* self = static_cast<Model*>(user_data);
  if (!t || !t->name[0]) return false;

  // Match "l_out-<digits>" — the per-layer output tensor in ik_llama / llama.cpp.
  if (std::strncmp(t->name, "l_out-", 6) != 0) return false;
  char* end = nullptr;
  long layer = std::strtol(t->name + 6, &end, 10);
  if (end == t->name + 6 || layer < 0 || layer >= self->n_layer_) return false;

  if (ask) return true; // ask phase: opt-in for this tensor

  // Data phase: tensor is [n_embd, n_tokens]. We want the LAST token row.
  // ne[0] = n_embd, ne[1] = n_tokens (typically).
  if (t->ne[0] != self->n_embd_) return true; // unexpected shape, skip silently
  const int64_t n_tokens = t->ne[1];
  if (n_tokens <= 0) return true;
  const size_t row_bytes = self->n_embd_ * sizeof(float);
  const size_t offset = static_cast<size_t>(n_tokens - 1) * row_bytes;

  auto& buf = self->layer_hidden_[layer];
  ggml_backend_tensor_get(t, buf.data(), offset, row_bytes);

  // V4 partial-forward: if this layer matches the partial target, signal abort.
  // The abort_cb (called by ggml shortly after) will then short-circuit the eval.
  const int32_t target = self->partial_target_layer_.load(std::memory_order_relaxed);
  if (target >= 0 && layer >= target) {
    self->partial_target_reached_.store(true, std::memory_order_release);
  }
  return true;
}

Model::Model(const Napi::CallbackInfo& info) : Napi::ObjectWrap<Model>(info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsString()) {
    Napi::TypeError::New(env, "Model(path: string, opts?: {contextSize, threads, gpuLayers})")
        .ThrowAsJavaScriptException();
    return;
  }

  std::string path = info[0].As<Napi::String>();
  if (path == "__subvocal_fork__") return;  // fork sentinel — fields populated by ForkContext

  uint32_t n_ctx = 2048;
  uint32_t n_threads = 4;
  uint32_t n_threads_batch = 0; // 0 = same as n_threads
  int32_t n_gpu_layers = 0;
  bool embeddings = false;       // V3: opt-in to extract hidden state from forward()
  bool capture_layers = false;   // V3.2: opt-in to capture per-layer hidden states

  if (info.Length() > 1 && info[1].IsObject()) {
    auto opts = info[1].As<Napi::Object>();
    if (opts.Has("contextSize")) n_ctx = opts.Get("contextSize").As<Napi::Number>().Uint32Value();
    if (opts.Has("threads")) n_threads = opts.Get("threads").As<Napi::Number>().Uint32Value();
    if (opts.Has("threadsBatch")) n_threads_batch = opts.Get("threadsBatch").As<Napi::Number>().Uint32Value();
    if (opts.Has("gpuLayers")) n_gpu_layers = opts.Get("gpuLayers").As<Napi::Number>().Int32Value();
    if (opts.Has("embeddings")) embeddings = opts.Get("embeddings").ToBoolean().Value();
    if (opts.Has("captureLayerHidden")) capture_layers = opts.Get("captureLayerHidden").ToBoolean().Value();
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

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = n_ctx;
  // Leave n_batch / n_ubatch at library defaults (typically 2048 / 512).
  cparams.n_threads = n_threads;
  cparams.n_threads_batch = n_threads_batch;
  cparams.embeddings = embeddings;
  // V3.2: install our cb_eval to intercept l_out-<layer> tensors.
  if (capture_layers_) {
    cparams.cb_eval = &Model::LayerHiddenCaptureCb;
    cparams.cb_eval_user_data = this;
  }
  // V4: install abort_cb only if user opted into partial forward
  // (always install when capture_layers_ is on, since that's the prerequisite)
  // The cb is a cheap atomic load, harmless during normal forward (returns false).

  ctx_ = llama_init_from_model(model_, cparams);
  if (!ctx_) {
    llama_free_model(model_);
    model_ = nullptr;
    Napi::Error::New(env, "Failed to init context").ThrowAsJavaScriptException();
    return;
  }

  n_vocab_ = llama_n_vocab(model_);
  n_ctx_ = n_ctx;
  n_embd_ = llama_model_n_embd(model_);
  n_layer_ = llama_n_layer(model_); // ik_llama legacy (upstream renamed to llama_model_n_layer)
  // V6: pre-allocate shadow buffer for getLogitsFast()
  logits_shadow_.resize(n_vocab_);

  // V6.1: initialize logit processor
  logit_processor_ = new LogitProcessor(n_vocab_);

   // V6.2: initialize KV cache manager
   kv_cache_ = new KVCacheManager(ctx_, n_ctx_);

   // V6.3: initialize suffix tree for self-speculative drafting
   suffix_tree_ = new common_suffix_tree(64);  // default max depth 64
 
   if (capture_layers_) {
    layer_hidden_.assign(n_layer_, std::vector<float>(n_embd_, 0.0f));
    // V4: register abort callback (called regularly by ggml during eval).
    // partial_target_layer_ stays at -1 outside ForwardPartial, so the cb is a no-op
    // during a normal Forward(). Zero overhead in the common case.
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
  if (ctx_) {
    llama_free(ctx_);
    ctx_ = nullptr;
  }
  if (model_) {
    llama_free_model(model_);
    model_ = nullptr;
  }
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
  if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
  if (model_) { llama_free_model(model_); model_ = nullptr; }
}

Napi::Value Model::VocabSize(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), n_vocab_);
}

Napi::Value Model::ContextSize(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), n_ctx_);
}

Napi::Value Model::EmbeddingSize(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), n_embd_);
}

Napi::Value Model::LayerCount(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), n_layer_);
}

// V4: forward, but stop evaluation right after `layer_limit` is computed.
// Useful for early intent detection / small→large transfer (the deep layers add
// "continuation prediction" noise that hurts cross-prompt semantic discrimination).
// Requires { captureLayerHidden: true } at construction.
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

  llama_kv_cache_clear(ctx_);

  // Arm partial-forward state. The cb_eval will flip reached=true once l_out-<layer_limit>
  // is captured, and the abort_cb will then short-circuit eval at the next checkpoint.
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
    // status == 2 typically means "aborted by callback" — that's success for us
    if (status != 0 && status != 2) { final_status = status; break; }
  }

  // Disarm: subsequent forward() calls behave normally
  partial_target_layer_.store(-1, std::memory_order_release);
  partial_target_reached_.store(false, std::memory_order_relaxed);

  return Napi::Number::New(env, final_status);
}

// V5: snapshot the full KV cache (and other context state) as a binary blob.
// Agent use cases: branching exploration, backtracking on tool-call mistakes,
// "savepoint" before risky generation, multi-turn conversation memory without
// re-tokenizing the prompt history every turn.
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
  // Return a Uint8Array view of the actual bytes written (≤ needed).
  return Napi::Uint8Array::New(env, written, buf, 0);
}

// V5: restore a previously-saved KV state. Buffer must come from a getKVState()
// on the SAME model (same arch, same context size, same hparams). Cross-model
// restore is undefined behavior.
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
  return Napi::Number::New(env, (double)read);
}

// V3.2: returns the per-layer hidden state of the last token after forward().
// Requires constructor option { captureLayerHidden: true }.
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
  // Fallback chain: works for both pooled (embedding models: LAST/MEAN/CLS) and
  // non-pooled (causal models: get last token hidden state directly).
  //   - get_embeddings_seq(0): present if pooling_type != NONE
  //   - get_embeddings_ith(-1): present if pooling_type == NONE
  float* embd = llama_get_embeddings_seq(ctx_, 0);
  if (!embd) embd = llama_get_embeddings_ith(ctx_, -1);
  if (!embd) embd = llama_get_embeddings(ctx_);
  if (!embd) {
    Napi::Error::New(env, "Embeddings unavailable (call forward() first; ensure embeddings=true)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto buf = Napi::ArrayBuffer::New(env, n_embd_ * sizeof(float));
  std::memcpy(buf.Data(), embd, n_embd_ * sizeof(float));
  return Napi::Float32Array::New(env, n_embd_, buf, 0);
}

Napi::Value Model::Tokenize(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!model_) {
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

  // Probe: negative return is the needed buffer size
  std::vector<llama_token> tokens(text.size() + 16);
  int32_t n = llama_tokenize(model_, text.c_str(), (int32_t)text.size(),
                             tokens.data(), (int32_t)tokens.size(),
                             add_special, parse_special);
  if (n < 0) {
    tokens.resize(-n);
    n = llama_tokenize(model_, text.c_str(), (int32_t)text.size(),
                       tokens.data(), (int32_t)tokens.size(),
                       add_special, parse_special);
    if (n < 0) {
      Napi::Error::New(env, "Tokenization failed").ThrowAsJavaScriptException();
      return env.Null();
    }
  }

  // Copy into a new ArrayBuffer (owned by JS)
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

  // V1: stateless. Reset KV cache and prefill in chunks of n_batch (default 2048).
  // We use llama_batch_init + manual logits[last]=true so that both logits and
  // embeddings (when cparams.embeddings=true) are saved. The llama_batch_get_one
  // helper in ik_llama does NOT set logits flags reliably.
  llama_kv_cache_clear(ctx_);

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

  // V6.3: apply AST token mask (disallowed tokens → -Inf) before bias/steering
  if (logit_processor_ && logit_processor_->HasASTMask()) {
    float* logits = llama_get_logits_ith(ctx_, -1);
    if (logits) {
      logit_processor_->ApplyASTMask(logits);
    }
  }

  // V6.1: automatically apply biases + steering after decode if any are active
  if (logit_processor_ && (logit_processor_->HasBias() || logit_processor_->HasSteering())) {
    float* logits = llama_get_logits_ith(ctx_, -1);
    if (logits) {
      logit_processor_->Apply(logits);
    }
  }

  return Napi::Number::New(env, 0);
}

// V6: zero-alloc logits — memcpy into a pre-allocated shadow buffer that lives
// as long as the Model. Avoids per-call malloc of the ArrayBuffer backing store.
// The returned Float32Array is backed by Model-internal memory; do NOT use after
// the Model is freed.
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

  // Wrap the stable shadow buffer as an external ArrayBuffer (no-op finalizer —
  // logits_shadow_ is owned by Model and freed in the destructor).
  auto buf = Napi::ArrayBuffer::New(env, logits_shadow_.data(),
                                    n_vocab_ * sizeof(float),
                                    [](Napi::Env, void*) { /* no-op */ });
  return Napi::Float32Array::New(env, n_vocab_, buf, 0);
}

// V6: true zero-copy logits. Points directly into llama's internal logits buffer.
// @unsafe — the returned data is valid only until the next llama_decode() call.
// No memcpy, no allocation. The finalizer is a no-op because llama owns the memory.
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

  // Wrap the raw llama pointer as an external ArrayBuffer. No finalizer needed
  // since llama.cpp manages the underlying memory.
  auto buf = Napi::ArrayBuffer::New(env, logits, n_vocab_ * sizeof(float),
                                    [](Napi::Env, void*) { /* no-op */ });
  return Napi::Float32Array::New(env, n_vocab_, buf, 0);
}

// V6: multi-position logits extraction. For every index in `indices`, calls
// llama_get_logits_ith and copies the resulting row into a fresh Float32Array.
// Useful for speculative decoding verification, validation heads, or anytime
// the caller needs logits at arbitrary sequence positions.
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

Napi::Value Model::GetLogits(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }

  // -1 = last token in batch
  float* logits = llama_get_logits_ith(ctx_, -1);
  if (!logits) {
    Napi::Error::New(env, "Logits unavailable (call forward() first)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  // V1: copy. Zero-copy via Napi::ArrayBuffer::New(env, data, size) would
  // share llama's internal buffer, but ownership is tricky (re-allocated on
  // next decode). Safer to copy for V1.
  auto buf = Napi::ArrayBuffer::New(env, n_vocab_ * sizeof(float));
  std::memcpy(buf.Data(), logits, n_vocab_ * sizeof(float));
  return Napi::Float32Array::New(env, n_vocab_, buf, 0);
}

Napi::Value Model::Detokenize(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!model_) {
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
    int32_t len = llama_token_to_piece(model_, tokens[i], buf, sizeof(buf), 0, false);
    if (len > 0) {
      out.append(buf, len);
    } else if (len < 0) {
      // Token piece longer than 256 bytes — fallback (rare)
      std::vector<char> big(-len + 1);
      int32_t real = llama_token_to_piece(model_, tokens[i], big.data(),
                                          (int32_t)big.size(), 0, false);
      if (real > 0) out.append(big.data(), real);
    }
  }
  return Napi::String::New(env, out);
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

  // Apply biases immediately to the current logits if available
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
// V6.3: AST-Aware Logit Masking
// ============================================================================

Napi::Value Model::SetASTTokenMask(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 1 || !info[0].IsTypedArray()) {
    Napi::TypeError::New(env, "setASTTokenMask(allowed: Int32Array)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Int32Array arr = info[0].As<Napi::Int32Array>();
  int32_t n = (int32_t)arr.ElementLength();
  std::vector<llama_token> allowed(n);
  std::memcpy(allowed.data(), arr.Data(), n * sizeof(int32_t));
  logit_processor_->SetASTTokenMask(allowed);
  return env.Undefined();
}

Napi::Value Model::ClearASTMask(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  logit_processor_->ClearASTMask();
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

// V8: multi-context support — fork a new llama_context sharing the loaded model
// weights. The new context has its own KV cache (n_past=0) and independent
// logit processor. Caller must free() both instances separately.
Napi::Value Model::ForkContext(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = n_ctx_;
  cparams.n_threads = n_threads_;
  cparams.n_threads_batch = n_threads_batch_;
  cparams.embeddings = embeddings_enabled_;

  auto new_ctx = llama_init_from_model(model_, cparams);
  if (!new_ctx) {
    Napi::Error::New(env, "forkContext: llama_init_from_model failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  auto obj = Model::constructorRef.New({ Napi::String::New(env, "__subvocal_fork__"), Napi::Object::New(env) });
  auto* fork = Model::Unwrap(obj);
  fork->model_ = model_;
  fork->ctx_ = new_ctx;
  fork->n_vocab_ = n_vocab_;
  fork->n_ctx_ = n_ctx_;
  fork->n_embd_ = n_embd_;
  fork->n_layer_ = n_layer_;
  fork->n_threads_ = n_threads_;
  fork->n_threads_batch_ = n_threads_batch_;
  fork->embeddings_enabled_ = embeddings_enabled_;
  fork->capture_layers_ = false;
  fork->n_past_ = 0;
  fork->logits_shadow_.resize(n_vocab_);
  fork->logit_processor_ = new LogitProcessor(n_vocab_);
  fork->kv_cache_ = new KVCacheManager(new_ctx, n_ctx_);
  fork->suffix_tree_ = new common_suffix_tree(64);

  return obj;
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

// ============================================================================
// V6.4: Sparse Logits — Top-K Extraction + SIMD Mask
// ============================================================================

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

  // Linear scan: O(vocab) but only copies k pairs (<100 bytes vs 600KB full)
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

  // Sort descending by logit
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

Napi::Value Model::ApplyMaskSIMD(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!ctx_) {
    Napi::Error::New(env, "Context has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 1 || !info[0].IsTypedArray()) {
    Napi::TypeError::New(env, "applyMaskSIMD(disallowed: Int32Array)").ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Int32Array arr = info[0].As<Napi::Int32Array>();
  const int32_t n = (int32_t)arr.ElementLength();

  float* logits = llama_get_logits_ith(ctx_, -1);
  if (!logits) {
    Napi::Error::New(env, "Logits unavailable").ThrowAsJavaScriptException();
    return env.Null();
  }

  for (int32_t i = 0; i < n; i++) {
    const int32_t id = reinterpret_cast<int32_t*>(arr.Data())[i];
    if (id >= 0 && id < n_vocab_) logits[id] = -std::numeric_limits<float>::infinity();
  }
  return env.Undefined();
}

// ---------------------------------------------------------------------------
// Module init / unload

static bool g_backend_initialized = false;

// ── Substory 2.4: Zero-Copy Pinned Buffer ─────────────────────────────────────
// createPinnedBuffer(sizeInInts: number) → Int32Array
// Allocates an off-heap int32 array via malloc and exposes it to JS as an external
// ArrayBuffer (napi_create_external_arraybuffer). V8 cannot relocate this buffer,
// so the address is stable for the lifetime of the ArrayBuffer — safe to pass as a
// pointer to C++ without pinning concerns. The finalizer frees the malloc block.
Napi::Value CreatePinnedBuffer(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "createPinnedBuffer(sizeInInts: number)").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  const uint32_t n_ints = info[0].As<Napi::Number>().Uint32Value();
  if (n_ints == 0) {
    Napi::RangeError::New(env, "createPinnedBuffer: sizeInInts must be > 0").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  const size_t byte_len = static_cast<size_t>(n_ints) * sizeof(int32_t);
  void* ptr = std::malloc(byte_len);
  if (!ptr) {
    Napi::Error::New(env, "createPinnedBuffer: malloc failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  std::memset(ptr, 0, byte_len);
  // External ArrayBuffer: V8 owns the JS wrapper but NOT the memory.
  // The finalizer is called by GC when the ArrayBuffer is collected.
  auto ab = Napi::ArrayBuffer::New(
    env, ptr, byte_len,
    [](Napi::Env /*env*/, void* data) { std::free(data); }
  );
  return Napi::Int32Array::New(env, n_ints, ab, 0);
}

// ── Substory 1.4: AVX2 full-text byte scan ────────────────────────────────────
// Standalone export (no class): scanBytes(haystack: Buffer, needle: Buffer) → Int32Array
// Returns byte offsets of every occurrence of needle in haystack.
// Uses AVX2 first-byte scan + scalar tail verification (see text_scanner.cpp).
Napi::Value ScanBytesBinding(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 2) {
    Napi::TypeError::New(env, "scanBytes(haystack: Buffer, needle: Buffer)").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  // Accept both Buffer and Uint8Array for both arguments
  auto toBytes = [&](Napi::Value val, const char* argName) -> std::pair<const uint8_t*, size_t> {
    if (val.IsBuffer()) {
      auto buf = val.As<Napi::Buffer<uint8_t>>();
      return { buf.Data(), buf.Length() };
    }
    if (val.IsTypedArray()) {
      auto ta = val.As<Napi::TypedArray>();
      if (ta.TypedArrayType() != napi_uint8_array) {
        Napi::TypeError::New(env, std::string(argName) + " must be Buffer or Uint8Array").ThrowAsJavaScriptException();
        return { nullptr, 0 };
      }
      auto u8 = val.As<Napi::Uint8Array>();
      return { u8.Data(), u8.ElementLength() };
    }
    Napi::TypeError::New(env, std::string(argName) + " must be Buffer or Uint8Array").ThrowAsJavaScriptException();
    return { nullptr, 0 };
  };

  auto [hay_ptr, hay_len]       = toBytes(info[0], "haystack");
  auto [needle_ptr, needle_len] = toBytes(info[1], "needle");
  // Allow zero-length arrays (nullptr is fine for length 0 — scan_bytes handles it)
  if ((hay_len > 0 && !hay_ptr) || (needle_len > 0 && !needle_ptr)) return env.Undefined();

  std::vector<int32_t> offsets = scan_bytes(hay_ptr, hay_len, needle_ptr, needle_len);

  auto result = Napi::Int32Array::New(env, offsets.size());
  for (size_t i = 0; i < offsets.size(); ++i) result[i] = offsets[i];
  return result;
}

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
  if (!g_backend_initialized) {
    llama_backend_init();
    g_backend_initialized = true;
    napi_add_env_cleanup_hook(env, [](void*) { llama_backend_free(); }, nullptr);
  }
  exports.Set("Model", Model::Init(env));
  exports.Set("SuffixTree", SuffixTree::Init(env));
  exports.Set("createPinnedBuffer", Napi::Function::New(env, CreatePinnedBuffer));
  exports.Set("scanBytes", Napi::Function::New(env, ScanBytesBinding));
#ifdef SUBVOCAL_HAS_LIBGIT2
  // SUBVOCAL-PATCH (Mac port, 2026-06-28): see binding_gpu.cpp's InitAll for why this needs
  // guarding — unguarded, this crashes every require() when libgit2 isn't installed.
  subvocal::InitGit(env, exports);
#endif
  return exports;
}

NODE_API_MODULE(subvocal_ffi_cpu, InitAll)
