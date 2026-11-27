#include "tools/cli_common.h"
#include "tools/model_loader.h"
#include "training/dataloader.h"
#include "training/jsonl_loader.h"
#include "format/parquet.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/chat_template.h"
#include "core/common.h"
#include "core/json.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <memory>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;
using namespace gai;

namespace {

void show_usage() {
    std::cout <<
    "gai_data - High-Performance Pure C++ Dataset Preparation Tool\n\n"
    "Converts raw text, JSON/JSONL, and tabular datasets directly into\n"
    "binary tokenized .gbin shards with zero Python dependencies.\n\n"
    "usage:\n"
    "  gai_data --input <file_or_dir> --tokenizer <model.gguf|dir> --output-dir <dir>\n\n"
    "options:\n"
    "  --input, -i <path>       input file (.txt, .jsonl, .json, .csv, .tsv, .parquet) or directory\n"
    "  --tokenizer, -t <path>   tokenizer source: GGUF file or HuggingFace directory / tokenizer.json\n"
    "  --output-dir, -o <dir>   output directory for .gbin shards (default: data_prepared)\n"
    "  --mode, -m <mode>        processing mode: auto | pretrain | sft (default: auto)\n"
    "                           - pretrain: raw text -> continuous supervised tokens\n"
    "                           - sft: conversations/instructions -> assistant loss masked tokens\n"
    "  --prefix <name>          train shard prefix (default: train)\n"
    "  --val-prefix <name>      validation shard prefix (default: val)\n"
    "  --val-split <ratio>      validation split fraction (default: 0.02 = 2%, 0 = disable)\n"
    "  --shard-tokens <n>       max tokens per shard (default: 50000000 = 50M tokens, 0 = unlimited)\n"
    "  --max-seq-len <n>        max tokens per sample / window (default: 4096, 0 = unlimited)\n"
    "  --text-col <name>        column/key name containing text for CSV/TSV/JSON (default: auto)\n"
    "  --min-chars <n>          minimum document length in characters (default: 16)\n"
    "  --seed <n>               random seed for train/val split (default: 42)\n"
    "  --help, -h               show this help\n";
}

struct ShardManager {
    std::string out_dir;
    std::string prefix;
    int vocab_size = 0;
    u64 tok_fingerprint = 0;
    bool with_loss_mask = false;
    u64 max_tokens_per_shard = 50000000;
    int shard_idx = 0;
    std::unique_ptr<ShardWriter> writer;
    u64 total_tokens = 0;
    u64 total_docs = 0;

    ShardManager(std::string dir, std::string pfx, int vocab, u64 fp, bool mask, u64 max_toks)
        : out_dir(std::move(dir)), prefix(std::move(pfx)), vocab_size(vocab),
          tok_fingerprint(fp),
          with_loss_mask(mask), max_tokens_per_shard(max_toks) {}

    void add_document(const std::vector<i32>& tokens, const std::vector<u8>* mask = nullptr) {
        if (!writer || (max_tokens_per_shard > 0 && writer->token_count() >= max_tokens_per_shard)) {
            rotate();
        }
        writer->add_document(tokens, mask);
        total_tokens += tokens.size();
        total_docs++;
    }

    void rotate() {
        if (writer) {
            writer->close();
            writer.reset();
        }
        char buf[32];
        std::snprintf(buf, sizeof(buf), "_%04d.gbin", shard_idx++);
        std::string path = (fs::path(out_dir) / (prefix + buf)).string();
        writer = std::make_unique<ShardWriter>(path, vocab_size, with_loss_mask, tok_fingerprint);
    }

    void close() {
        if (writer) {
            writer->close();
            writer.reset();
        }
    }
};

static bool read_csv_record(std::istream& in, char delim, std::string& out) {
    out.clear();
    bool in_quotes = false;
    bool any = false;
    std::string line;
    while (std::getline(in, line)) {
        any = true;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!out.empty()) out.push_back('\n');
        out += line;
        for (size_t i = 0; i < line.size(); ++i) {
            if (line[i] != '"') continue;
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') { ++i; continue; }
            in_quotes = !in_quotes;
        }
        (void)delim;
        if (!in_quotes) return true;
    }
    return any;
}
bool extract_csv_text(const std::string& line, char delim, int text_col_idx, std::string& out) {
    if (text_col_idx < 0) {
        out = line;
        return true;
    }
    int cur_col = 0;
    bool in_quotes = false;
    std::string field;
    for (size_t i = 0; i <= line.size(); ++i) {
        const bool at_end = (i == line.size());
        if (!at_end && line[i] == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                if (cur_col == text_col_idx) field.push_back('"');
                ++i;
                continue;
            }
            in_quotes = !in_quotes;
            continue;
        }
        if (at_end || (!in_quotes && line[i] == delim)) {
            if (cur_col == text_col_idx) { out = field; return true; }
            cur_col++;
            field.clear();
            continue;
        }
        if (cur_col == text_col_idx) field.push_back(line[i]);
    }
    return false;
}

}

static int data_prep_main(int argc, char** argv) {
    enable_utf8_console();
    Args args(argc, argv);
    args.reject_unknown({"input","tokenizer","output-dir","prefix","val-prefix","val-split","shard-tokens","max-seq-len","text-col","min-chars","mode","seed","sft","auto","content","conversations","document","instruction","messages","prompt","response","train","val","help","verbose"});
    if (args.flag("help") || args.flag("h") || argc < 2) {
        show_usage();
        return 0;
    }

    const std::string input_path = args.str("input", args.str("i"));
    const std::string tok_path   = args.str("tokenizer", args.str("t"));
    const std::string out_dir    = args.str("output-dir", args.str("o", "data_prepared"));
    const std::string mode       = args.str("mode", args.str("m", "auto"));
    const std::string prefix     = args.str("prefix", "train");
    const std::string val_prefix = args.str("val-prefix", "val");
    const double val_split       = args.real("val-split", 0.02);
    const u64 shard_tokens       = args.num_u64("shard-tokens", 50000000ULL);
    const int max_seq_len        = args.num_int("max-seq-len", 4096);
    const int min_chars_arg      = args.num_int("min-chars", 16);
    if (min_chars_arg < 0) {
        std::cerr << "[-] Error: --min-chars must be >= 0.\n";
        return 1;
    }
    if (!(val_split >= 0.0 && val_split <= 1.0)) {
        std::cerr << "[-] Error: --val-split must be in [0, 1].\n";
        return 1;
    }
    const size_t min_chars       = static_cast<size_t>(min_chars_arg);
    const u64 seed               = args.num_u64("seed", 42ULL);
    const std::string text_col   = args.str("text-col");

    if (input_path.empty()) {
        std::cerr << "[-] Error: --input is required.\n";
        return 1;
    }
    if (tok_path.empty()) {
        std::cerr << "[-] Error: --tokenizer is required.\n";
        return 1;
    }

    std::error_code ec;
    fs::create_directories(out_dir, ec);
    if (ec) {
        std::cerr << "[-] Error creating output directory " << out_dir << ": " << ec.message() << "\n";
        return 1;
    }

    std::cout << "[*] Loading tokenizer from: " << tok_path << "...\n";
    Tokenizer tk;
    std::vector<i32> eos_ids;
    if (!load_tokenizer_auto(tok_path, tk, eos_ids)) {
        std::cerr << "[-] Failed to load tokenizer from " << tok_path << "\n";
        return 1;
    }
    const int vocab_size = tk.vocab_size();
    std::cout << "[+] Tokenizer loaded: vocab_size = " << vocab_size << "\n";

    std::vector<std::string> files;
    if (fs::is_directory(input_path)) {
        for (const auto& entry : fs::recursive_directory_iterator(input_path)) {
            if (entry.is_regular_file()) {
                const std::string ext = entry.path().extension().string();
                if (ext == ".txt" || ext == ".jsonl" || ext == ".json" || ext == ".csv" || ext == ".tsv" || ext == ".parquet") {
                    files.push_back(entry.path().string());
                }
            }
        }
        std::sort(files.begin(), files.end());
    } else if (fs::is_regular_file(input_path)) {
        files.push_back(input_path);
    } else {
        std::cerr << "[-] Error: Input path does not exist: " << input_path << "\n";
        return 1;
    }

    if (files.empty()) {
        std::cerr << "[-] Error: No matching input files found.\n";
        return 1;
    }

    std::cout << "[*] Found " << files.size() << " input file(s) to process.\n";

    bool sft_mode = (mode == "sft");
    if (mode == "auto") {
        for (const auto& f : files) {
            std::ifstream test_f(f);
            std::string line;
            if (std::getline(test_f, line)) {
                if (line.find("\"messages\"") != std::string::npos ||
                    line.find("\"conversations\"") != std::string::npos ||
                    line.find("\"instruction\"") != std::string::npos) {
                    sft_mode = true;
                    break;
                }
            }
        }
    }

    std::cout << "[*] Selected mode: " << (sft_mode ? "SFT (loss-masked chat/instruct)" : "PRETRAIN (continuous text)") << "\n";

    ShardManager train_shards(out_dir, prefix, vocab_size, tk.fingerprint(), sft_mode, shard_tokens);
    std::unique_ptr<ShardManager> val_shards;
    if (val_split > 0.0) {
        val_shards = std::make_unique<ShardManager>(out_dir, val_prefix, vocab_size, tk.fingerprint(), sft_mode, shard_tokens);
    }

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    auto start_time = std::chrono::steady_clock::now();
    u64 total_docs_processed = 0;
    u64 skipped_short = 0;
    u64 skipped_invalid = 0;
    u64 last_report_tokens = 0;
    auto last_report_time = start_time;

    for (size_t file_idx = 0; file_idx < files.size(); ++file_idx) {
        const std::string& file_path = files[file_idx];
        std::ifstream infile(file_path, std::ios::binary);
        if (!infile.good()) {
            std::cerr << "[-] Warning: cannot open " << file_path << "\n";
            continue;
        }

        std::cout << "\n[*] Processing [" << (file_idx + 1) << "/" << files.size() << "]: "
                  << fs::path(file_path).filename().string() << "...\n";

        const std::string ext = fs::path(file_path).extension().string();

        if (ext == ".parquet") {
            std::string err;
            ParquetSchema schema;
            if (!ParquetReader::inspect(file_path, schema, &err)) {
                std::cerr << "[-] Error reading parquet " << file_path << ": " << err << "\n";
                skipped_invalid++;
                continue;
            }
            std::cout << "    [parquet] " << schema.num_rows << " rows, " << schema.columns.size() << " columns\n";

            const bool parquet_ok = ParquetReader::read_rows(file_path, {}, [&](const ParquetRowMap& row) -> bool {
                bool is_val = (val_shards && dist(rng) < val_split);
                ShardManager& target_shard = is_val ? *val_shards : train_shards;

                auto get_col = [&](const std::string& name) -> std::string {
                    auto it = row.find(name);
                    if (it == row.end() || it->second.is_null) return "";
                    return it->second.as_string();
                };

                if (sft_mode) {
                    JsonlSample sample;
                    bool parsed = false;
                    std::string msgs_str = get_col("messages");
                    std::string convs_str = get_col("conversations");
                    std::string inst_str = get_col("instruction");
                    std::string out_str = get_col("output");
                    std::string prompt_str = get_col("prompt");
                    std::string resp_str = get_col("response");

                    if (!msgs_str.empty()) {
                        parsed = JsonlLoader::parse_line("{\"messages\":" + msgs_str + "}", sample);
                        if (!parsed) parsed = JsonlLoader::parse_line(msgs_str, sample);
                    } else if (!convs_str.empty()) {
                        parsed = JsonlLoader::parse_line("{\"conversations\":" + convs_str + "}", sample);
                        if (!parsed) parsed = JsonlLoader::parse_line(convs_str, sample);
                    } else if (!inst_str.empty() && !out_str.empty()) {
                        std::string user_content = inst_str;
                        std::string in_str = get_col("input");
                        if (!in_str.empty()) user_content += "\n" + in_str;
                        sample.messages.push_back({Role::User, user_content, ""});
                        sample.messages.push_back({Role::Assistant, out_str, ""});
                        parsed = true;
                    } else if (!prompt_str.empty() && !resp_str.empty()) {
                        sample.messages.push_back({Role::User, prompt_str, ""});
                        sample.messages.push_back({Role::Assistant, resp_str, ""});
                        parsed = true;
                    }

                    if (!parsed || sample.messages.empty()) {
                        skipped_invalid++;
                        return true;
                    }

                    std::vector<u8> loss_mask;
                    const ChatTemplate tmpl = chat_template_from_vocab(tk, sample.thinking);
                    std::vector<i32> tokens = encode_chat(tk, tmpl, sample.messages, false, &loss_mask);
                    if (tokens.empty()) {
                        skipped_invalid++;
                        return true;
                    }
                    if (max_seq_len > 0 && static_cast<int>(tokens.size()) > max_seq_len) {
                        tokens.resize(static_cast<size_t>(max_seq_len));
                        loss_mask.resize(static_cast<size_t>(max_seq_len));
                    }
                    target_shard.add_document(tokens, &loss_mask);
                } else {
                    std::string doc_text;
                    if (!text_col.empty()) {
                        doc_text = get_col(text_col);
                    } else {
                        doc_text = get_col("text");
                        if (doc_text.empty()) doc_text = get_col("content");
                        if (doc_text.empty()) doc_text = get_col("document");
                        if (doc_text.empty()) {
                            for (const auto& kv : row) {
                                if (!kv.second.is_null && !kv.second.as_string().empty()) {
                                    doc_text = kv.second.as_string();
                                    break;
                                }
                            }
                        }
                    }

                    if (doc_text.size() < min_chars) {
                        skipped_short++;
                        return true;
                    }

                    std::vector<i32> tokens = tk.encode(doc_text, false, false);
                    if (tokens.empty()) {
                        skipped_invalid++;
                        return true;
                    }
                    if (max_seq_len > 0 && static_cast<int>(tokens.size()) > max_seq_len) {
                        tokens.resize(static_cast<size_t>(max_seq_len));
                    }
                    target_shard.add_document(tokens, nullptr);
                }

                total_docs_processed++;
                if (total_docs_processed % 5000 == 0) {
                    auto now = std::chrono::steady_clock::now();
                    double elapsed_sec = std::chrono::duration<double>(now - last_report_time).count();
                    u64 cur_total_tokens = train_shards.total_tokens + (val_shards ? val_shards->total_tokens : 0);
                    double tok_s = elapsed_sec > 0 ? (cur_total_tokens - last_report_tokens) / elapsed_sec : 0;
                    last_report_tokens = cur_total_tokens;
                    last_report_time = now;

                    std::cout << "\r[gai_data] Docs: " << total_docs_processed
                              << " | Total: " << (cur_total_tokens / 1000000.0) << "M tokens"
                              << " | Speed: " << static_cast<u64>(tok_s / 1000.0) << "k tok/s"
                              << " (Train: " << (train_shards.total_tokens / 1000000.0) << "M"
                              << (val_shards ? (", Val: " + std::to_string(val_shards->total_tokens / 1000) + "k)") : ")")
                              << std::flush;
                }
                return true;
            }, 0, &err);
            if (!parquet_ok) {
                if (!err.empty()) std::cerr << "[-] parquet detail: " << err << "\n";
                std::cerr << "[-] Error: parquet reader stopped early on " << file_path
                          << "; dataset would be incomplete\n";
                skipped_invalid++;
            }
            continue;
        }

        const bool is_csv = (ext == ".csv");
        const bool is_tsv = (ext == ".tsv");
        const char delim = is_tsv ? '\t' : ',';
        int col_idx = -1;

        if (is_csv || is_tsv) {
            std::string header_line;
            if (std::getline(infile, header_line)) {
                std::stringstream ss(header_line);
                std::string item;
                int cur = 0;
                while (std::getline(ss, item, delim)) {
                    if (!item.empty() && item.back() == '\r') item.pop_back();
                    const bool match = text_col.empty()
                        ? (item == "text" || item == "content" || item == "document" || item == "prompt")
                        : (item == text_col || item == ("\"" + text_col + "\""));
                    if (match && col_idx < 0) col_idx = cur;
                    ++cur;
                }
                if (col_idx < 0) col_idx = 0;
            }
        }

        std::string line;
        auto next_record = [&]() -> bool {
            if (is_csv || is_tsv) return read_csv_record(infile, delim, line);
            if (!std::getline(infile, line)) return false;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            return true;
        };
        while (next_record()) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;

            bool is_val = (val_shards && dist(rng) < val_split);
            ShardManager& target_shard = is_val ? *val_shards : train_shards;

            if (sft_mode) {
                JsonlSample sample;
                if (!JsonlLoader::parse_line(line, sample) || sample.messages.empty()) {
                    skipped_invalid++;
                    continue;
                }

                std::vector<u8> loss_mask;
                const ChatTemplate tmpl = chat_template_from_vocab(tk, sample.thinking);
                std::vector<i32> tokens = encode_chat(tk, tmpl, sample.messages, false, &loss_mask);

                if (tokens.empty()) {
                    skipped_invalid++;
                    continue;
                }
                if (max_seq_len > 0 && static_cast<int>(tokens.size()) > max_seq_len) {
                    tokens.resize(static_cast<size_t>(max_seq_len));
                    loss_mask.resize(static_cast<size_t>(max_seq_len));
                }

                target_shard.add_document(tokens, &loss_mask);
            } else {
                std::string doc_text;
                if (is_csv || is_tsv) {
                    if (!extract_csv_text(line, delim, col_idx, doc_text)) {
                        skipped_invalid++;
                        continue;
                    }
                } else if (ext == ".jsonl" || ext == ".json") {
                    Json j;
                    if (Json::parse(line, j) && j.is_object()) {
                        if (!text_col.empty() && j.has(text_col)) {
                            doc_text = j.at(text_col).as_string();
                        } else if (j.has("text")) {
                            doc_text = j.at("text").as_string();
                        } else if (j.has("content")) {
                            doc_text = j.at("content").as_string();
                        } else {
                            doc_text = line;
                        }
                    } else {
                        doc_text = line;
                    }
                } else {
                    doc_text = line;
                }

                if (doc_text.size() < min_chars) {
                    skipped_short++;
                    continue;
                }

                std::vector<i32> tokens = tk.encode(doc_text, false, false);
                if (tokens.empty()) {
                    skipped_invalid++;
                    continue;
                }
                if (max_seq_len > 0 && static_cast<int>(tokens.size()) > max_seq_len) {
                    tokens.resize(static_cast<size_t>(max_seq_len));
                }

                target_shard.add_document(tokens, nullptr);
            }

            total_docs_processed++;

            if (total_docs_processed % 5000 == 0) {
                auto now = std::chrono::steady_clock::now();
                double elapsed_sec = std::chrono::duration<double>(now - last_report_time).count();
                u64 cur_total_tokens = train_shards.total_tokens + (val_shards ? val_shards->total_tokens : 0);
                double tok_s = elapsed_sec > 0 ? (cur_total_tokens - last_report_tokens) / elapsed_sec : 0;
                last_report_tokens = cur_total_tokens;
                last_report_time = now;

                std::cout << "\r[gai_data] Docs: " << total_docs_processed
                          << " | Total: " << (cur_total_tokens / 1000000.0) << "M tokens"
                          << " | Speed: " << static_cast<u64>(tok_s / 1000.0) << "k tok/s"
                          << " (Train: " << (train_shards.total_tokens / 1000000.0) << "M"
                          << (val_shards ? (", Val: " + std::to_string(val_shards->total_tokens / 1000) + "k)") : ")")
                          << std::flush;
            }
        }
    }

    train_shards.close();
    if (val_shards) val_shards->close();

    auto end_time = std::chrono::steady_clock::now();
    double total_sec = std::chrono::duration<double>(end_time - start_time).count();
    u64 final_total_tokens = train_shards.total_tokens + (val_shards ? val_shards->total_tokens : 0);

    std::cout << "\n\n========================================================\n";
    std::cout << "         Ghassan AI - Data Preparation Complete         \n";
    std::cout << "========================================================\n";
    std::cout << "  Input Files Processed : " << files.size() << "\n";
    std::cout << "  Total Documents       : " << total_docs_processed << "\n";
    std::cout << "  Skipped (< min_chars) : " << skipped_short << "\n";
    std::cout << "  Skipped (invalid/err) : " << skipped_invalid << "\n";
    std::cout << "  Train Tokens          : " << train_shards.total_tokens
              << " (" << train_shards.shard_idx << " shards)\n";
    if (val_shards) {
        std::cout << "  Validation Tokens     : " << val_shards->total_tokens
                  << " (" << val_shards->shard_idx << " shards)\n";
    }
    std::cout << "  Total Tokens Written  : " << final_total_tokens << "\n";
    std::cout << "  Elapsed Time          : " << total_sec << " seconds\n";
    std::cout << "  Average Throughput    : "
              << (total_sec > 0 ? static_cast<u64>(final_total_tokens / total_sec) : 0)
              << " tokens/second\n";
    std::cout << "  Output Directory      : " << out_dir << "\n";
    std::cout << "========================================================\n";

    return 0;
}

int main(int argc, char** argv) {
    try {
        return data_prep_main(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "[-] Fatal: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "[-] Fatal: unknown exception during data preparation\n";
        return 1;
    }
}
