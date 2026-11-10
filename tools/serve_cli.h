#pragma once

#include "tools/cli_common.h"

namespace gai {

// `ghassan-ai serve`: an OpenAI compatible HTTP server.
//
//   POST /v1/chat/completions   (streaming and non streaming)
//   POST /v1/completions
//   GET  /v1/models
//   GET  /health
int cmd_serve(const Args& args);

}  // namespace gai
