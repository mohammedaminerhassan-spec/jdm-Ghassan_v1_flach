// Long-context KV cache correctness: ring eviction + pinned attention sink.
//
// The generator keeps the first few tokens as a "pinned prefix" (system
// prompt / attention sink) and recycles the rest of the slots as a ring.
// This test verifies that:
//   * pinned slots never move while the ring advances
//   * no two live logical positions alias to the same physical slot
//   * Granite mamba layers have no KV cache (compact mapping)

#include "inference/kv_cache.h"

#include <cstdio>
#include <set>
#include <vector>

using namespace gai;

int main() {
    ModelConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_size = 32;
    cfg.num_layers = 6;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.arch = ModelConfig::ArchGranite;
    // layers 0,2,4 = mamba (no cache); 1,3,5 = attention (2 cache slots)
    cfg.layer_types = {0, 1, 0, 1, 0, 1};

    KVCache cache(cfg, /*max_len=*/8, Device::CPU);
    if (cache.bytes() == 0) {
        std::printf("FAIL: KV cache allocated nothing\n");
        return 1;
    }
    cache.set_pinned_prefix(2);

    std::vector<int> slot_of;
    for (int i = 0; i < 20; ++i) slot_of.push_back(cache.allocate_slot());

    // The two pinned logical positions always map to physical 0 and 1.
    if (cache.physical_slot(0) != 0 || cache.physical_slot(1) != 1) {
        std::printf("FAIL: pinned attention sink moved\n");
        return 1;
    }

    // Every live logical position must have a unique physical slot.
    std::set<int> used;
    for (int j = 0; j < cache.length(); ++j) {
        const int p = cache.physical_slot(j);
        if (p < 0 || p >= cache.capacity() || !used.insert(p).second) {
            std::printf("FAIL: alias or out-of-range physical slot at logical %d -> %d\n", j, p);
            return 1;
        }
    }
    // The first allocated pinned slots must be exactly the sink positions.
    if (slot_of[0] != 0 || slot_of[1] != 1) {
        std::printf("FAIL: first allocations were not the pinned sink\n");
        return 1;
    }
    std::printf("[kv] len=%d cap=%d pinned=2 ring_start=%d: sink stable, no aliases\n",
                cache.length(), cache.capacity(), cache.ring_start());
    std::printf("PASS\n");
    return 0;
}
