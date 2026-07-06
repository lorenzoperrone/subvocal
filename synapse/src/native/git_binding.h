#pragma once
#include <napi.h>
#include <string>

namespace subvocal {

Napi::Object InitGit(Napi::Env env, Napi::Object exports);

} // namespace subvocal
