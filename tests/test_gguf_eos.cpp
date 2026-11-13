// The GGUF EOS metadata round trip.
//
// The writer stores tokenizer.ggml.eos_token_id as a 4-byte UINT32 (see
// GGUFWriter::set_u32). The reader used to require 8 bytes and silently
// reported eos = 0 otherwise. On Granite id 0 is "!", so the generator
// treated an exclamation mark as end-of-sequence and cut every answer at
// the first "!" ("Hello" instead of "Hello! I am an AI ..."), while the
// same prompt on the safetensors checkpoint (whose EOS comes from
// generation_config.json) answered fully. This test locks the round trip:
// the id written must be the id read back, and it must never degrade to 0.

#include "core/common.h"
#include "core/dtype.h"
#include "core/tensor.h"
#include "format/gguf_codec.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace gai;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (ok) {
        printf("  ok    %s\n", what.c_str());
    } else {
        printf("  FAIL  %s\n", what.c_str());
        ++g_failures;
    }
}

}  // namespace

int main() {
    printf("== the GGUF EOS metadata ==\n");

    const std::string path = "test_gguf_eos_tmp.gguf";
    const uint32_t eos = 100257;  // <|end_of_text|> in Granite-4

    {
        GGUFWriter w(path);
        w.set_arch("ghassan");
        w.set_model_name("eos probe");
        w.set_tokenizer_eos_id(eos);
        Tensor t({4}, DType::F32, Device::CPU);
        const float values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        std::memcpy(t.f32(), values, sizeof(values));
        w.add_tensor("probe", t, GGMLType::F32);
        w.write();
    }

    GGUFReader r;
    check(r.open(path), "the probe file opens");
    const std::vector<i32> ids = r.eos_ids();
    check(ids.size() == 1, "exactly one stop id is reported");
    check(!ids.empty() && ids[0] == static_cast<i32>(eos),
          "the stop id is the one that was written (100257)");
    check(!ids.empty() && ids[0] != 0,
          "the stop id is not 0 (Granite's '!' - the bug that truncated the answers)");

    std::remove(path.c_str());

    if (g_failures == 0) {
        printf("\nPASS\n");
        return 0;
    }
    printf("\nFAIL: %d check(s) failed\n", g_failures);
    return 1;
}
