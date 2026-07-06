// Subvocal FFI — Suffix Tree (Self-Speculative Drafting)
// Wraps ik_llama.cpp common_suffix_tree for Node.js / TypeScript
//
// Usage:
//   const suffixTree = new SuffixTree(maxDepth);
//   suffixTree.extend(contextTokens);
//   const draft = suffixTree.speculate(contextTokens, maxSpecTokens);
//
// References:
//   ik_llama.cpp/common/suffix-tree.h
//   "Suffix Decoding" (Saxena et al., 2024) — arXiv:2411.04975

#pragma once
#include <napi.h>
#include "suffix-tree.h"
#include "llama.h"
#include <vector>
#include <string>

class SuffixTree : public Napi::ObjectWrap<SuffixTree> {
 public:
  static Napi::Function Init(Napi::Env env);

  SuffixTree(const Napi::CallbackInfo& info);
  ~SuffixTree();

 private:
  Napi::Value Extend(const Napi::CallbackInfo& info);
  Napi::Value Clear(const Napi::CallbackInfo& info);
  Napi::Value Speculate(const Napi::CallbackInfo& info);
  Napi::Value TokenCount(const Napi::CallbackInfo& info);
  Napi::Value MaxDepth(const Napi::CallbackInfo& info);

  common_suffix_tree* tree_ = nullptr;
};

Napi::Function SuffixTree::Init(Napi::Env env) {
  return DefineClass(env, "SuffixTree", {
    InstanceMethod("extend",    &SuffixTree::Extend),
    InstanceMethod("clear",     &SuffixTree::Clear),
    InstanceMethod("speculate", &SuffixTree::Speculate),
    InstanceMethod("tokenCount",&SuffixTree::TokenCount),
    InstanceMethod("maxDepth",  &SuffixTree::MaxDepth),
  });
}

SuffixTree::SuffixTree(const Napi::CallbackInfo& info) : Napi::ObjectWrap<SuffixTree>(info) {
  int max_depth = 64;
  if (info.Length() > 0 && info[0].IsNumber()) {
    max_depth = info[0].As<Napi::Number>().Int32Value();
    if (max_depth <= 0) max_depth = 64;
  }
  tree_ = new common_suffix_tree(max_depth);
}

SuffixTree::~SuffixTree() {
  delete tree_;
  tree_ = nullptr;
}

Napi::Value SuffixTree::Extend(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!tree_) {
    Napi::Error::New(env, "SuffixTree has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 1 || !info[0].IsTypedArray()) {
    Napi::TypeError::New(env, "extend(tokens: Int32Array)").ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::Int32Array arr = info[0].As<Napi::Int32Array>();
  int32_t n = static_cast<int32_t>(arr.ElementLength());
  if (n > 0) {
    tree_->extend(reinterpret_cast<llama_token*>(arr.Data()), n);
  }
  return env.Undefined();
}

Napi::Value SuffixTree::Clear(const Napi::CallbackInfo& info) {
  if (!tree_) {
    Napi::Env env = info.Env();
    Napi::Error::New(env, "SuffixTree has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  tree_->clear();
  return info.Env().Undefined();
}

Napi::Value SuffixTree::Speculate(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!tree_) {
    Napi::Error::New(env, "SuffixTree has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  if (info.Length() < 1 || !info[0].IsTypedArray()) {
    Napi::TypeError::New(env,
      "speculate(context: Int32Array, maxSpecTokens?: number, minTokenProb?: number, minMatchCount?: number, minMatchLen?: number)"
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

  std::vector<llama_token> draft = tree_->speculate(
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

Napi::Value SuffixTree::TokenCount(const Napi::CallbackInfo& info) {
  if (!tree_) {
    Napi::Env env = info.Env();
    Napi::Error::New(env, "SuffixTree has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  return Napi::Number::New(info.Env(), tree_->token_count());
}

Napi::Value SuffixTree::MaxDepth(const Napi::CallbackInfo& info) {
  if (!tree_) {
    Napi::Env env = info.Env();
    Napi::Error::New(env, "SuffixTree has been freed").ThrowAsJavaScriptException();
    return env.Null();
  }
  return Napi::Number::New(info.Env(), tree_->max_depth());
}
