#include "training/shard_builder.h"
#include "training/dataloader.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

namespace fs = std::filesystem;

namespace gai {

namespace {

// The shard file name: <prefix>_0000.gbin (the loader picks the shards by
// the prefix, so the prefix must be the one the recipe declares).
std::string shard_path(const std::string& dir, const std::string& prefix, int index) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "_%04d.gbin", index);
    return (fs::path(dir) / (prefix + buf)).string();
}

// One document per line. A line that is empty (or only whitespace) is not a
// document.
bool read_lines(const std::string& path, std::vector<std::string>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    std::string line;
    while (std::getline(f, line)) {
        // A trailing \r (a CRLF file) would end up inside the token stream.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        bool blank = true;
        for (char c : line) {
            if (c != ' ' && c != '\t') { blank = false; break; }
        }
        if (!blank) out.push_back(line);
    }
    return true;
}

}  // namespace

ShardBuilder::Stats ShardBuilder::build_file(const std::string& path,
                                             const Tokenizer& tk,
                                             const std::string& out_path,
                                             int seq_len) {
    Stats st;
    std::vector<std::string> lines;
    if (!read_lines(path, lines)) {
        log_warn("[shard] cannot read " + path);
        return st;
    }
    const int vocab = tk.vocab_size();
    ShardWriter w(out_path, vocab, /*with_loss_mask=*/false);
    for (const std::string& line : lines) {
        std::vector<i32> ids = tk.encode(line, false, false);
        if (ids.empty()) continue;
        // A token outside the vocabulary means the shard was built with a
        // DIFFERENT tokenizer than the model: the document is dropped (and
        // counted) instead of silently training on garbage ids.
        bool bad = false;
        for (i32 id : ids) {
            if (id < 0 || id >= vocab) { bad = true; break; }
        }
        if (bad) {
            ++st.skipped_docs;
            st.skipped_tokens += ids.size();
            continue;
        }
        if (seq_len > 0 && static_cast<int>(ids.size()) > static_cast<int>(seq_len) * 4) {
            // A very long document: keep the first windows of it (the loader
            // samples a random window inside a document, so a long document
            // is fine - but not an unbounded one).
            const size_t keep = static_cast<size_t>(seq_len) * 4;
            ids.resize(keep);
        }
        w.add_document(ids);
        ++st.documents;
        st.tokens += ids.size();
    }
    w.close();
    ++st.shards;
    return st;
}

ShardBuilder::Stats ShardBuilder::build_directory(const std::string& dir,
                                                  const Tokenizer& tk,
                                                  const std::string& prefix,
                                                  int seq_len,
                                                  u64 target_tokens_per_shard) {
    Stats total;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        log_error("[shard] not a directory: " + dir);
        return total;
    }
    std::vector<std::string> files;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        if (!e.is_regular_file(ec) || ec) continue;
        const std::string name = e.path().filename().string();
        if (name.size() < 4 || name.compare(name.size() - 4, 4, ".txt") != 0) continue;
        files.push_back(e.path().string());
    }
    if (ec) {
        log_error("[shard] cannot list " + dir + ": " + ec.message());
        return total;
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        log_warn("[shard] no .txt file in " + dir +
                 " (the dataset script writes one document per line)");
        return total;
    }

    const u64 per_shard = target_tokens_per_shard > 0 ? target_tokens_per_shard : 0;
    int shard_index = 0;
    std::unique_ptr<ShardWriter> writer;
    u64 shard_tokens = 0;

    auto close_shard = [&]() {
        if (writer) { writer->close(); writer.reset(); ++total.shards; }
    };
    auto open_shard = [&](int index) {
        writer = std::make_unique<ShardWriter>(
            shard_path(dir, prefix, index), tk.vocab_size(), false);
        shard_tokens = 0;
    };

    for (const std::string& f : files) {
        std::vector<std::string> lines;
        if (!read_lines(f, lines)) continue;
        for (const std::string& line : lines) {
            std::vector<i32> ids = tk.encode(line, false, false);
            if (ids.empty()) continue;
            const int vocab = tk.vocab_size();
            bool bad = false;
            for (i32 id : ids) {
                if (id < 0 || id >= vocab) { bad = true; break; }
            }
            if (bad) {
                ++total.skipped_docs;
                total.skipped_tokens += ids.size();
                continue;
            }
            if (seq_len > 0 && static_cast<int>(ids.size()) > static_cast<int>(seq_len) * 4) {
                ids.resize(static_cast<size_t>(seq_len) * 4);
            }
            if (!writer) open_shard(shard_index);
            writer->add_document(ids);
            ++total.documents;
            total.tokens += ids.size();
            shard_tokens += ids.size();
            if (per_shard > 0 && shard_tokens >= per_shard) {
                close_shard();
                ++shard_index;
            }
        }
    }
    close_shard();

    log_info(strfmt("[shard] %s: %zu shard(s), %zu document(s), %s tokens%s",
                    dir.c_str(), static_cast<size_t>(total.shards),
                    static_cast<size_t>(total.documents),
                    human_count(total.tokens).c_str(),
                    total.skipped_docs
                        ? strfmt(" (%llu document(s) skipped: token ids outside the vocabulary)",
                                 static_cast<unsigned long long>(total.skipped_docs)).c_str()
                        : ""));
    return total;
}

}  // namespace gai
