#include "evaluation/perplexity.h"
#include "core/common.h"

#include <cmath>
#include <limits>

namespace gai {

static PerplexityResult from_nll(double sum, i64 tokens, i64 batches) {
    PerplexityResult result;
    result.tokens = tokens;
    result.batches = batches;
    if (tokens <= 0) {
        result.mean_nll = std::numeric_limits<double>::infinity();
        result.perplexity = std::numeric_limits<double>::infinity();
        return result;
    }
    const double mean = sum / static_cast<double>(tokens);
    result.mean_nll = mean;
    result.perplexity = std::exp(mean);
    return result;
}

PerplexityResult evaluate_perplexity(Generator& generator,
                                      const std::vector<i32>& tokens,
                                      const std::vector<u8>* mask,
                                      const std::vector<i32>* segment_ids) {
    i64 count = 0;
    double nll = generator.score_tokens(tokens, mask, &count, segment_ids);
    return from_nll(nll * static_cast<double>(count), count, 1);
}

PerplexityResult evaluate_shard_perplexity(Generator& generator,
                                            const std::string& shard_dir,
                                            const std::string& prefix,
                                            int batch_size,
                                            int seq_len,
                                            int max_batches,
                                            bool pack_sequences,
                                            u64 seed) {
    GAI_CHECK(max_batches > 0, "perplexity max_batches must be > 0");
    BatchSpec spec{batch_size, seq_len, pack_sequences};
    DataLoader loader;
    loader.set_vocab_size(generator.vocab_size());
    GAI_CHECK(loader.open_glob(shard_dir, prefix, spec, seed),
              "no held-out shards found for perplexity: " + shard_dir);
    double sum = 0.0;
    i64 tokens = 0;
    i64 batches = 0;
    Batch batch;
    while (batches < max_batches && loader.next(batch)) {
        GAI_CHECK(batch.ids.size() == batch.targets.size(),
                  "perplexity: ids/targets length mismatch (dataloader contract)");
        for (int b = 0; b < batch.B; ++b) {
            const size_t off = static_cast<size_t>(b) * static_cast<size_t>(batch.T);
            std::vector<i32> row_ids(batch.ids.begin() + static_cast<std::ptrdiff_t>(off),
                                     batch.ids.begin() + static_cast<std::ptrdiff_t>(off + batch.T));
            std::vector<u8> row_mask(static_cast<size_t>(batch.T), 0);
            for (int i = 1; i < batch.T; ++i)
                row_mask[static_cast<size_t>(i)] =
                    batch.targets[off + static_cast<size_t>(i) - 1] >= 0 ? 1 : 0;
            std::vector<i32> row_segments;
            if (!batch.segment_ids.empty()) {
                row_segments.assign(batch.segment_ids.begin() + static_cast<std::ptrdiff_t>(off),
                                    batch.segment_ids.begin() + static_cast<std::ptrdiff_t>(off + batch.T));
            }
            i64 count = 0;
            double nll = generator.score_tokens(row_ids, &row_mask, &count,
                                                 row_segments.empty() ? nullptr : &row_segments);
            sum += nll * static_cast<double>(count);
            tokens += count;
        }
        ++batches;
    }
    return from_nll(sum, tokens, batches);
}

}
