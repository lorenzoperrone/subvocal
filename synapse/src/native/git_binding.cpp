// Subvocal Git binding — V1: libgit2-backed in-process stage / commit / diff
// Eliminates the ~20ms process-spawn overhead of calling `git` CLI.
//
// Uses a static global git_repository* (one-at-a-time, not thread-safe).
// Single-agent usage is fine; multi-agent parallel access would need a pool.

#include "git_binding.h"

#ifdef SUBVOCAL_HAS_LIBGIT2
#include <git2.h>
#endif

#include <string>

namespace subvocal {
namespace {

#ifdef SUBVOCAL_HAS_LIBGIT2
static git_repository* g_repo = nullptr;
static bool g_libgit2_initialized = false;

std::string ensure_libgit2_init() {
    if (!g_libgit2_initialized) {
        int err = git_libgit2_init();
        if (err < 0) {
            const git_error* ge = git_error_last();
            return std::string("git_libgit2_init failed: ") + (ge ? ge->message : "unknown");
        }
        g_libgit2_initialized = true;
    }
    return "";
}

std::string check_repo() {
    if (!g_repo) return "No repo open — call gitInit first";
    return "";
}
#endif

} // anonymous namespace

// N-API wrappers

Napi::Value GitInit(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
#ifdef SUBVOCAL_HAS_LIBGIT2
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "gitInit(repoPath: string)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    std::string repo_path = info[0].As<Napi::String>();

    std::string author_name = "Subvocal";
    std::string author_email = "subvocal@local";
    if (info.Length() > 1 && info[1].IsObject()) {
        auto opts = info[1].As<Napi::Object>();
        if (opts.Has("authorName")) author_name = opts.Get("authorName").As<Napi::String>();
        if (opts.Has("authorEmail")) author_email = opts.Get("authorEmail").As<Napi::String>();
    }

    std::string err = ensure_libgit2_init();
    if (!err.empty()) {
        Napi::Error::New(env, err).ThrowAsJavaScriptException();
        return env.Undefined();
    }

    if (g_repo) {
        git_repository_free(g_repo);
        g_repo = nullptr;
    }

    int ret = git_repository_open(&g_repo, repo_path.c_str());
    if (ret != 0) {
        const git_error* ge = git_error_last();
        Napi::Error::New(env, std::string("git_repository_open failed: ") + (ge ? ge->message : "unknown")).ThrowAsJavaScriptException();
        return env.Undefined();
    }

    return Napi::Boolean::New(env, true);
#else
    Napi::Error::New(env, "libgit2 not linked (build with libgit2 installed)").ThrowAsJavaScriptException();
    return env.Undefined();
#endif
}

Napi::Value GitStageAll(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
#ifdef SUBVOCAL_HAS_LIBGIT2
    std::string err = check_repo();
    if (!err.empty()) {
        Napi::Error::New(env, err).ThrowAsJavaScriptException();
        return env.Undefined();
    }

    git_index* index = nullptr;
    int ret = git_repository_index(&index, g_repo);
    if (ret != 0) {
        Napi::Error::New(env, "Failed to get repo index").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    char* pathspec_strs[] = { (char*)"*" };
    git_strarray pathspec = { pathspec_strs, 1 };

    ret = git_index_add_all(index, &pathspec, GIT_INDEX_ADD_DEFAULT, nullptr, nullptr);
    if (ret != 0) {
        git_index_free(index);
        const git_error* ge = git_error_last();
        Napi::Error::New(env, std::string("git_index_add_all failed: ") + (ge ? ge->message : "unknown")).ThrowAsJavaScriptException();
        return env.Undefined();
    }

    ret = git_index_write(index);
    git_index_free(index);
    if (ret != 0) {
        const git_error* ge = git_error_last();
        Napi::Error::New(env, std::string("git_index_write failed: ") + (ge ? ge->message : "unknown")).ThrowAsJavaScriptException();
        return env.Undefined();
    }

    return env.Undefined();
#else
    Napi::Error::New(env, "libgit2 not linked").ThrowAsJavaScriptException();
    return env.Undefined();
#endif
}

Napi::Value GitCommit(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
#ifdef SUBVOCAL_HAS_LIBGIT2
    std::string err = check_repo();
    if (!err.empty()) {
        Napi::Error::New(env, err).ThrowAsJavaScriptException();
        return env.Undefined();
    }
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "gitCommit(message: string, authorName?: string, authorEmail?: string)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    std::string message = info[0].As<Napi::String>();
    std::string author_name = "Subvocal";
    std::string author_email = "subvocal@local";
    if (info.Length() > 1 && info[1].IsString()) author_name = info[1].As<Napi::String>();
    if (info.Length() > 2 && info[2].IsString()) author_email = info[2].As<Napi::String>();

    git_signature* sig = nullptr;
    int ret = git_signature_now(&sig, author_name.c_str(), author_email.c_str());
    if (ret != 0) {
        const git_error* ge = git_error_last();
        Napi::Error::New(env, std::string("git_signature_now failed: ") + (ge ? ge->message : "unknown")).ThrowAsJavaScriptException();
        return env.Undefined();
    }

    git_index* index = nullptr;
    ret = git_repository_index(&index, g_repo);
    if (ret != 0) {
        git_signature_free(sig);
        Napi::Error::New(env, "Failed to get repo index").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    git_oid tree_oid;
    ret = git_index_write_tree(&tree_oid, index);
    git_index_free(index);
    if (ret != 0) {
        git_signature_free(sig);
        const git_error* ge = git_error_last();
        Napi::Error::New(env, std::string("git_index_write_tree failed: ") + (ge ? ge->message : "unknown")).ThrowAsJavaScriptException();
        return env.Undefined();
    }

    git_tree* tree = nullptr;
    ret = git_tree_lookup(&tree, g_repo, &tree_oid);
    if (ret != 0) {
        git_signature_free(sig);
        const git_error* ge = git_error_last();
        Napi::Error::New(env, std::string("git_tree_lookup failed: ") + (ge ? ge->message : "unknown")).ThrowAsJavaScriptException();
        return env.Undefined();
    }

    git_commit* parent = nullptr;
    int parent_count = 0;
    git_oid head_oid;
    ret = git_reference_name_to_id(&head_oid, g_repo, "HEAD");
    if (ret == 0) {
        ret = git_commit_lookup(&parent, g_repo, &head_oid);
        if (ret == 0) parent_count = 1;
    }

    git_oid commit_oid;
    ret = git_commit_create(&commit_oid, g_repo, "HEAD", sig, sig, "UTF-8",
                            message.c_str(), tree, parent_count,
                            parent_count ? (const git_commit**)&parent : nullptr);

    if (parent) git_commit_free(parent);
    git_tree_free(tree);
    git_signature_free(sig);

    if (ret != 0) {
        const git_error* ge = git_error_last();
        Napi::Error::New(env, std::string("git_commit_create failed: ") + (ge ? ge->message : "unknown")).ThrowAsJavaScriptException();
        return env.Undefined();
    }

    char sha[GIT_OID_HEXSZ + 1] = {0};
    git_oid_fmt(sha, &commit_oid);
    sha[GIT_OID_HEXSZ] = '\0';
    return Napi::String::New(env, sha, GIT_OID_HEXSZ);
#else
    Napi::Error::New(env, "libgit2 not linked").ThrowAsJavaScriptException();
    return env.Undefined();
#endif
}

Napi::Value GitDiffStaged(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
#ifdef SUBVOCAL_HAS_LIBGIT2
    std::string err = check_repo();
    if (!err.empty()) {
        Napi::Error::New(env, err).ThrowAsJavaScriptException();
        return env.Undefined();
    }

    git_diff* diff = nullptr;
    git_diff_options diffopts = GIT_DIFF_OPTIONS_INIT;
    int ret = git_diff_index_to_workdir(&diff, g_repo, nullptr, &diffopts);
    if (ret != 0) {
        const git_error* ge = git_error_last();
        Napi::Error::New(env, std::string("git_diff_index_to_workdir failed: ") + (ge ? ge->message : "unknown")).ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Render diff as unified patch
    git_buf buf = GIT_BUF_INIT_CONST(nullptr, 0);
    ret = git_diff_to_buf(&buf, diff, GIT_DIFF_FORMAT_PATCH);
    git_diff_free(diff);
    if (ret != 0) {
        git_buf_dispose(&buf);
        const git_error* ge = git_error_last();
        Napi::Error::New(env, std::string("git_diff_to_buf failed: ") + (ge ? ge->message : "unknown")).ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string result = buf.ptr ? std::string(buf.ptr, buf.size) : "";
    git_buf_dispose(&buf);
    return Napi::String::New(env, result);
#else
    Napi::Error::New(env, "libgit2 not linked").ThrowAsJavaScriptException();
    return env.Undefined();
#endif
}

Napi::Value GitDiffWorking(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
#ifdef SUBVOCAL_HAS_LIBGIT2
    std::string err = check_repo();
    if (!err.empty()) {
        Napi::Error::New(env, err).ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Get HEAD tree
    git_oid head_oid;
    int ret = git_reference_name_to_id(&head_oid, g_repo, "HEAD");
    if (ret != 0) {
        const git_error* ge = git_error_last();
        Napi::Error::New(env, std::string("Failed to resolve HEAD: ") + (ge ? ge->message : "unknown")).ThrowAsJavaScriptException();
        return env.Undefined();
    }

    git_commit* head_commit = nullptr;
    ret = git_commit_lookup(&head_commit, g_repo, &head_oid);
    if (ret != 0) {
        const git_error* ge = git_error_last();
        Napi::Error::New(env, std::string("git_commit_lookup failed: ") + (ge ? ge->message : "unknown")).ThrowAsJavaScriptException();
        return env.Undefined();
    }

    git_tree* head_tree = nullptr;
    ret = git_commit_tree(&head_tree, head_commit);
    if (ret != 0) {
        git_commit_free(head_commit);
        const git_error* ge = git_error_last();
        Napi::Error::New(env, std::string("git_commit_tree failed: ") + (ge ? ge->message : "unknown")).ThrowAsJavaScriptException();
        return env.Undefined();
    }

    git_diff* diff = nullptr;
    git_diff_options diffopts = GIT_DIFF_OPTIONS_INIT;
    ret = git_diff_tree_to_workdir_with_index(&diff, g_repo, head_tree, &diffopts);
    git_tree_free(head_tree);
    git_commit_free(head_commit);
    if (ret != 0) {
        const git_error* ge = git_error_last();
        Napi::Error::New(env, std::string("git_diff_tree_to_workdir failed: ") + (ge ? ge->message : "unknown")).ThrowAsJavaScriptException();
        return env.Undefined();
    }

    git_buf buf = GIT_BUF_INIT_CONST(nullptr, 0);
    ret = git_diff_to_buf(&buf, diff, GIT_DIFF_FORMAT_PATCH);
    git_diff_free(diff);
    if (ret != 0) {
        git_buf_dispose(&buf);
        return Napi::String::New(env, "");
    }

    std::string result = buf.ptr ? std::string(buf.ptr, buf.size) : "";
    git_buf_dispose(&buf);
    return Napi::String::New(env, result);
#else
    Napi::Error::New(env, "libgit2 not linked").ThrowAsJavaScriptException();
    return env.Undefined();
#endif
}

Napi::Value GitFree(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
#ifdef SUBVOCAL_HAS_LIBGIT2
    if (g_repo) {
        git_repository_free(g_repo);
        g_repo = nullptr;
    }
#endif
    return env.Undefined();
}

// ---------------------------------------------------------------------------
// Module init
// ---------------------------------------------------------------------------

Napi::Object InitGit(Napi::Env env, Napi::Object exports) {
    exports.Set("gitInit", Napi::Function::New(env, GitInit));
    exports.Set("gitStageAll", Napi::Function::New(env, GitStageAll));
    exports.Set("gitCommit", Napi::Function::New(env, GitCommit));
    exports.Set("gitDiffStaged", Napi::Function::New(env, GitDiffStaged));
    exports.Set("gitDiffWorking", Napi::Function::New(env, GitDiffWorking));
    exports.Set("gitFree", Napi::Function::New(env, GitFree));
    return exports;
}

} // namespace subvocal
