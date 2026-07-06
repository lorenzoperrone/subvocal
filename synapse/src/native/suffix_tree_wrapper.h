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
