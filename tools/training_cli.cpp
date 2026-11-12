#include "tools/cli_common.h"
#include "tools/model_loader.h"
#include "training/trainer.h"
#include "core/device.h"
#include "format/gguf_codec.h"
#include "format/hf_checkpoint.h"
#include "tokenizer/tokenizer.h"
#include "training/jsonl_loader.h"

#include <iostream>
#include <filesystem>
#include <cstdlib>
#include <cstdio>
#include <exception>
#include <string>
#ifdef GAI_CUDA
#include <cuda_runtime.h>
#include "cuda/cuda_ops.h"
#include "cuda/cuda_utils.h"
#endif

namespace fs = std::filesystem;
using namespace gai;

static void gai_verbose_terminate() {
    try {
        if (auto p = std::current_exception()) std::rethrow_exception(p);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[fatal] uncaught exception: %s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "[fatal] uncaught non-std exception\n");
    }
    std::fflush(stderr);
    std::abort();
}

static void usage() {
    std::cout <<
    "gai_train - Ghassan AI training\n\n"
    "usage:\n"
    "  gai_train --config <recipe.yaml>\n"
    "  gai_train --config <recipe.yaml> --max-steps 100 --batch-size 2 --seq-len 256\n"
    "  gai_train --dry-run --config <recipe.yaml>     # report shapes/memory only\n\n"
    "overrides (all optional, they win over the config file):\n"
    "  --data <dir> --checkpoint-dir <dir> --resume auto|none|<path>\n"
    "  --batch-size --seq-len --grad-accum --max-steps --lr --warmup --seed\n"
    "  --optimizer adamw|lion|muon --scheduler cosine|wsd --ckpt-segments <n>\n"
    "  --ce-chunks <n>        chunked CE row-blocks, 0/1=off (default 4)\n"
    "  --strict-config        unknown/dead config keys fail instead of warning\n"
    "  --dry-run              arithmetic-only memory/schedule estimate (allocates nothing)\n"
    "  --max-vram-mb <n>      pre-flight gate: fail if the predicted peak exceeds n MiB\n"
    "  --output-budget-mb <n> pre-flight gate: fail if the run would exceed n MiB of\n"
    "                         SAVED output (/kaggle/working on Kaggle: repo+build+shards+\n"
    "                         checkpoints+gguf). 0 = off.\n"
    "                         or leaves <10% headroom (use with --dry-run)\n"
  "  --allow-recipe-drift   resume despite rope/eps/ctx recipe drift (research only)\n"
    "  --resume-mode exact|migrate   # exact: fail closed on any recipe/state drift\n"
    "  --gemm-fp16 0|1           # force CUDA fp16 GEMMs off/on\n"
    "  --device auto|cpu|cuda   --threads <n>\n"
    "  --export <path> --export-profile q8_k|q4_0 --tokenizer <hf-dir|tokenizer.json>\n"
    "      # GGUF-only export (self-contained). --compat native (ghassan-ai) |\n"
    "      # llama (dense Ollama) | llama_moe (MoE Ollama, needs moe_shared=false)\n";
}

int main(int argc, char** argv) {
    std::set_terminate(gai_verbose_terminate);
    enable_utf8_console();
    Args args(argc, argv);
    apply_common_flags(args);
    if (args.flag("help")) { usage(); return 0; }

    try {
        std::string cfg_path = args.str("config");
        Config cfg;
        if (fs::exists(cfg_path)) {
            cfg = Config::from_file(cfg_path);
            log_info("[cfg ] " + cfg_path);
        } else {
            log_warn("[cfg ] " + cfg_path + " not found, using built-in defaults");
        }

        ModelConfig mcfg = ModelConfig::from_config(cfg, "model");
        const ModelConfig recipe_mcfg = mcfg;   // the recipe's adapter settings
        TrainerConfig tcfg = TrainerConfig::from_config(cfg, args.flag("strict-config", false));

        // --model <dir>: fine-tune a HuggingFace checkpoint (config.json +
        // safetensors + tokenizer.json) with no conversion step. The
        // architecture comes from the checkpoint, not from the recipe.
        std::unique_ptr<HFCheckpoint> hf;
        bool weights_loaded_by_cli = false;
        {
            const std::string mdir = args.str("model");
            if (!mdir.empty() && fs::is_directory(mdir)) {
                hf = std::make_unique<HFCheckpoint>();
                // A dry run is an arithmetic estimate: it must not map the
                // 13.9 GB of shards just to print a memory plan.
                GAI_CHECK(hf->open(mdir, !args.flag("dry-run", false)),
                          "not a usable HuggingFace checkpoint: " + mdir);
                mcfg = hf->config();
                // The checkpoint owns the architecture, the recipe owns the
                // adapter: keep the recipe's LoRA settings (they would
                // otherwise be dropped by the swap and the run would train
                // the full model instead of the adapter).
                mcfg.take_adapter_settings_from(recipe_mcfg);
                weights_loaded_by_cli = true;
                log_info(strfmt("[mod ] %s: %d layers, d=%d, H=%d/%d, V=%d%s%s",
                                mdir.c_str(), mcfg.num_layers, mcfg.hidden_size,
                                mcfg.num_heads, mcfg.num_kv_heads, mcfg.vocab_size,
                                mcfg.use_moe ? ", MoE" : "",
                                mcfg.lora_rank > 0
                                    ? strfmt(", LoRA r=%d", mcfg.lora_rank).c_str() : ""));
            }
        }

        // training.pretrained_checkpoint may also point at a HuggingFace
        // directory (config.json + safetensors). In that case the weights are
        // loaded directly from it and the trainer must not try to read the
        // engine's own binary checkpoint format. The architecture comes from
        // that checkpoint too: the recipe's model block is only a mirror, so
        // a stale key there must not change the model that gets trained.
        if (!hf) {
            const std::string pc = tcfg.pretrained_checkpoint;
            if (!pc.empty() && fs::is_directory(pc) &&
                HFCheckpoint::looks_like_checkpoint(pc)) {
                hf = std::make_unique<HFCheckpoint>();
                GAI_CHECK(hf->open(pc, !args.flag("dry-run", false)),
                          "not a usable HuggingFace checkpoint: " + pc);
                const ModelConfig hc = hf->config();
                std::string why;
                if (!hc.same_architecture_as(mcfg, &why)) {
                    log_warn("[sft ] the recipe's model block does not mirror the checkpoint (" + why +
                             "): the checkpoint's config.json is the architecture that gets trained");
                    mcfg = hc;
                    mcfg.take_adapter_settings_from(recipe_mcfg);
                }
                log_info(strfmt("[sft ] pretrained checkpoint: %s (HuggingFace safetensors, %d layers, d=%d)",
                                pc.c_str(), hc.num_layers, hc.hidden_size));
                // The trainer loads the weights (it owns the pretrained path).
            }
        }

        if (cfg.has("tokenizer.vocab_size")) {
            const int tv = static_cast<int>(cfg.get_int("tokenizer.vocab_size", mcfg.vocab_size));
            if (tv != mcfg.vocab_size) {
                log_warn(strfmt("[cfg ] tokenizer.vocab_size %d != model.vocab_size %d "
                                "(embedding/tokenizer mismatch; training would corrupt)",
                                tv, mcfg.vocab_size));
            }
        }

        const bool strict_args = args.flag("strict-args", false);
        if (args.has("data"))            tcfg.data_dir = args.str("data");
        if (args.has("checkpoint-dir"))  tcfg.checkpoint_dir = args.str("checkpoint-dir");
        if (args.has("resume"))          tcfg.resume = args.str("resume");
        if (args.flag("allow-recipe-drift")) tcfg.allow_recipe_drift = true;
if (args.has("resume-mode")) {
    const std::string rm = args.str("resume-mode");
    if (rm != "exact" && rm != "migrate")
        GAI_FAIL("--resume-mode must be 'exact' or 'migrate' (got '" + rm + "')");
    tcfg.resume_mode = rm;
}
        if (args.has("output-budget-mb")) tcfg.output_budget_mb = args.num_strict("output-budget-mb");

        {
            const std::string tok_cli = args.str("tokenizer", "");
            const std::string tok_yaml = cfg.get_str("tokenizer.path", "");
            std::string tok_path = !tok_cli.empty() ? tok_cli : tok_yaml;
            if (!tok_path.empty() && !fs::path(tok_path).is_absolute()) {
                fs::path cand = fs::path(cfg_path).parent_path() / tok_path;
                if (fs::exists(cand)) tok_path = cand.string();
            }
            const u64 fp = tok_path.empty() ? 0 : fingerprint_file(tok_path);
            if (tcfg.tok_fingerprint == 0) {
                tcfg.tok_fingerprint = fp;
                if (fp != 0)
                    log_info(strfmt("[tok ] fingerprint %s (%s) pinned for resume verification",
                                    fingerprint_hex(fp).c_str(), tok_path.c_str()));
                else if (!tok_path.empty())
                    log_warn(strfmt("[tok ] cannot read %s: resume will not verify tokenizer identity",
                                    tok_path.c_str()));
            }
        }
        if (args.has("batch-size"))      tcfg.batch_size = strict_args ? args.num_int_strict("batch-size") : args.num_int("batch-size");
        if (args.has("seq-len"))         tcfg.seq_len = strict_args ? args.num_int_strict("seq-len") : args.num_int("seq-len");
        if (args.has("grad-accum"))      tcfg.grad_accum = strict_args ? args.num_int_strict("grad-accum") : args.num_int("grad-accum");

        if (args.has("max-steps")) {
            tcfg.max_steps = strict_args ? args.num_strict("max-steps") : args.num("max-steps");
            if (tcfg.epochs > 0)
                log_warn("[cfg ] --max-steps overrides yaml epochs (epochs ignored)");
            tcfg.epochs = 0;
        }
        if (args.has("gemm-fp16"))       tcfg.gemm_fp16 = (strict_args ? args.num_strict("gemm-fp16") : args.num("gemm-fp16")) != 0;
        if (args.has("lr"))              tcfg.learning_rate = static_cast<float>(strict_args ? args.real_strict("lr") : args.real("lr"));
        if (args.has("warmup"))          tcfg.warmup_steps = strict_args ? args.num_strict("warmup") : args.num("warmup");
        if (args.has("optimizer")) {
            tcfg.optimizer = args.str("optimizer");
            if (tcfg.optimizer != "adamw" && tcfg.optimizer != "lion" && tcfg.optimizer != "muon")
                GAI_FAIL("unknown --optimizer (adamw|lion|muon)");

            if (tcfg.optimizer == "lion" && tcfg.beta2 == 0.95f) tcfg.beta2 = 0.99f;
        }
        if (args.has("scheduler")) {
            tcfg.scheduler = args.str("scheduler");
            if (tcfg.scheduler != "cosine" && tcfg.scheduler != "wsd")
                GAI_FAIL("unknown --scheduler (cosine|wsd)");
        }
        if (args.has("ckpt-segments")) {
            tcfg.activation_checkpointing = true;
            tcfg.ckpt_segments = strict_args ? args.num_int_strict("ckpt-segments") : args.num_int("ckpt-segments");
            if (tcfg.ckpt_segments < 1) tcfg.ckpt_segments = 1;
            if (tcfg.ckpt_segments > 8) tcfg.ckpt_segments = 8;
        }
        if (args.has("ce-chunks")) {
            tcfg.ce_chunks = strict_args ? args.num_int_strict("ce-chunks") : args.num_int("ce-chunks");
            if (tcfg.ce_chunks <= 0) tcfg.ce_chunks = 1;
            if (tcfg.ce_chunks > 32) tcfg.ce_chunks = 32;
        }
        if (args.has("seed"))            tcfg.seed = static_cast<u64>(strict_args ? args.num_strict("seed") : args.num("seed"));
        if (args.has("device"))          tcfg.device = args.str("device");
        if (args.has("log-every"))       tcfg.log_every = strict_args ? args.num_strict("log-every") : args.num("log-every");
        if (args.has("eval-every"))      tcfg.eval_every = strict_args ? args.num_strict("eval-every") : args.num("eval-every");
        if (args.has("save-every"))      tcfg.save_every = strict_args ? args.num_strict("save-every") : args.num("save-every");
        if (args.has("vocab"))           mcfg.vocab_size = strict_args ? args.num_int_strict("vocab") : args.num_int("vocab");
        if (args.has("layers"))          mcfg.num_layers = strict_args ? args.num_int_strict("layers") : args.num_int("layers");
        if (args.has("hidden"))          mcfg.hidden_size = strict_args ? args.num_int_strict("hidden") : args.num_int("hidden");

        if (args.has("heads"))           mcfg.num_heads = strict_args ? args.num_int_strict("heads") : args.num_int("heads");
        if (args.has("kv-heads"))        mcfg.num_kv_heads = strict_args ? args.num_int_strict("kv-heads") : args.num_int("kv-heads");
        mcfg.validate();

        if (strict_args || args.flag("verbose", false))
            log_info(strfmt("[cfg ] overrides: B=%d T=%d accum=%d steps=%lld lr=%.6g opt=%s",
                            tcfg.batch_size, tcfg.seq_len, tcfg.grad_accum,
                            (long long)tcfg.max_steps, (double)tcfg.learning_rate,
                            tcfg.optimizer.c_str()));

print_device_report();

         std::string tok_path = "";
         Tokenizer hf_tok;          // the tokenizer of a HuggingFace checkpoint
         bool have_hf_tok = false;
         if (args.has("tokenizer")) {
             tok_path = args.str("tokenizer");
         } else if (cfg.has("tokenizer.path")) {
             tok_path = cfg.get_str("tokenizer.path", "");

             if (!tok_path.empty() && !fs::path(tok_path).is_absolute()) {
                 fs::path cfg_dir = fs::path(cfg_path).parent_path();
                 fs::path cand = cfg_dir / tok_path;

                 if (fs::exists(cand)) tok_path = cand.string();
             }
         }
         if (hf) {
             // The checkpoint carries its own vocabulary: it is the one the
             // shards were tokenized with.
             GAI_CHECK(hf->load_tokenizer(hf_tok),
                       "cannot build the tokenizer of the checkpoint");
             have_hf_tok = true;
             GAI_CHECK(hf_tok.vocab_size() <= mcfg.vocab_size,
                       strfmt("the tokenizer has %d tokens but the model has %d",
                              hf_tok.vocab_size(), mcfg.vocab_size));
             log_info(strfmt("[tok ] the checkpoint's tokenizer: vocab=%d%s",
                             hf_tok.vocab_size(),
                             hf_tok.vocab_size() == mcfg.vocab_size
                                 ? " (matches the model)"
                                 : " (the model has more rows: they are padding)"));
         }
         // Auto-convert any JSON/JSONL instruction datasets in data_dir into
         // .gbin shards (idempotent: existing .gbin newer than the source
         // are skipped). Uses the checkpoint's tokenizer when fine-tuning a
         // HF checkpoint, otherwise the --tokenizer file.
         {
             const Tokenizer* convert_tok = nullptr;
             if (have_hf_tok) convert_tok = &hf_tok;
             Tokenizer file_tok;
             if (!have_hf_tok && !tok_path.empty() && !args.flag("dry-run")) {
                 std::vector<i32> e_ids;
                 if (load_tokenizer_auto(tok_path, file_tok, e_ids)) {
                     convert_tok = &file_tok;
                 }
             }
             if (convert_tok) {
                 const int n = JsonlLoader::process_directory(tcfg.data_dir, *convert_tok, tcfg.seq_len);
                 if (n > 0) log_info(strfmt("[data] auto-converted %d JSONL dataset(s) -> .gbin shards", n));
             }
         }
         if (!tok_path.empty() && !have_hf_tok) {
             if (args.flag("dry-run")) {
                 log_info(strfmt("[tok ] dry-run: skipping tokenizer load (%s)", tok_path.c_str()));
             } else {
                 Tokenizer tok;
                 std::vector<i32> e_ids;
                 GAI_CHECK(load_tokenizer_auto(tok_path, tok, e_ids),
                           "cannot load tokenizer: " + tok_path);
                 GAI_CHECK(tok.vocab_size() <= mcfg.vocab_size,
                           strfmt("tokenizer vocab_size %d exceeds model.vocab_size %d "
                                  "(training would produce corrupt embeddings)",
                                  tok.vocab_size(), mcfg.vocab_size));
                 log_info(strfmt("[tok ] validated %s (vocab=%d matches model)",
                                 tok_path.c_str(), tok.vocab_size()));
             }
         }

         Device dev = Device::CPU;
        if (tcfg.device == "cuda") {
            GAI_CHECK(cuda_available(), "--device cuda requested but no CUDA device is available");
            dev = Device::CUDA;
        } else if (tcfg.device == "tpu" || tcfg.device == "metal" || tcfg.device == "vulkan") {
            GAI_FAIL("--device " + tcfg.device + " is not supported: T4-only build (auto|cpu|cuda)");
        } else if (tcfg.device == "auto") {
            dev = best_device();
        }

#ifdef GAI_CUDA
        if (dev == Device::CUDA) {
            int want = 0;
            if (const char* lr = std::getenv("LOCAL_RANK")) {
                int v = std::atoi(lr);
                if (v >= 0) want = v;
            }
            if (cudaSetDevice(want) == cudaSuccess) {
                log_info(strfmt("[dev ] pinned to CUDA:%d via LOCAL_RANK before Model alloc", want));
            }
        }
#endif
        log_info(strfmt("[dev ] using %s", device_name(dev)));

        log_info(strfmt("[prec] compute precision: %s", tcfg.precision_name().c_str()));

        if (args.flag("dry-run")) {

            const Model::MemoryPlan plan = Model::plan_memory(
                mcfg, tcfg.batch_size, tcfg.seq_len, true, tcfg.ce_chunks,
                tcfg.fp16_weight_cache);
            const RecipeCost cost = price_recipe(mcfg, tcfg);
            const u64   params    = cost.params;
            const size_t act      = cost.activations;
            const size_t opt_b    = cost.opt_state;
            const size_t total    = cost.total;
            const bool   lion     = (tcfg.optimizer == "lion");
            const bool   muon     = (tcfg.optimizer == "muon");
            const size_t muon_scratch = cost.muon_scratch;

            log_info("---------------- dry run: memory estimate ----------------");
            log_info(strfmt("  parameters          : %s (%s without embeddings)",
                            human_count(params).c_str(),
                            human_count(plan.params_no_embedding).c_str()));
            log_info(strfmt("  weights (fp32)      : %s", human_bytes(cost.weights).c_str()));
            log_info(strfmt("  gradients (fp32)    : %s", human_bytes(cost.grads).c_str()));
            log_info(strfmt("  %s (fp32)    : %s",
                            muon ? "muon m+v+NS " : lion ? "lion m      " : "adamw m+v   ",
                            human_bytes(opt_b).c_str()));
            log_info(strfmt("  fp16 weight cache  : %s%s",
                            human_bytes(cost.fp16_cache).c_str(),
                            tcfg.fp16_weight_cache ? "" : " (disabled)"));
            {
                // The two fixed-size blocks of a hybrid model: the Mamba-2
                // recurrent state and the inference KV cache.
                const size_t ms = Model::mamba_state_bytes(mcfg);
                if (ms > 0)
                    log_info(strfmt("  mamba-2 states      : %s (recurrent, all layers)",
                                    human_bytes(ms).c_str()));
                const size_t kvb = Model::kv_cache_bytes(mcfg, mcfg.max_seq_len);
                if (kvb > 0)
                    log_info(strfmt("  kv cache @%d ctx  : %s (fp16 packed, inference)",
                                    mcfg.max_seq_len, human_bytes(kvb).c_str()));
                // A 7B model in fp32 masters does not fit a 16GB T4; fp16
                // masters + fp16 weights do (13.9GB). Say it out loud.
                const size_t fp16_weights = static_cast<size_t>(cost.params) * 2;
                if (fp16_weights < cost.weights && fp16_weights > (4ull << 30))
                    log_info(strfmt("  fp16 weights would be: %s (fp32 masters: %s)",
                                    human_bytes(fp16_weights).c_str(),
                                    human_bytes(cost.weights).c_str()));
            }
            log_info(strfmt("  activations b=%d t=%d%s%s : %s",
                            tcfg.batch_size, tcfg.seq_len,
                            tcfg.activation_checkpointing ? " [ckpt]" : "",
                            tcfg.ce_chunks > 1 ? strfmt(" [ce-x%d]", tcfg.ce_chunks) : "",
                            human_bytes(act).c_str()));
            if (muon)
                log_info(strfmt("  muon NS scratch     : %s", human_bytes(muon_scratch).c_str()));
            log_info(strfmt("  eval arena (est)    : %s", human_bytes(cost.eval_extra).c_str()));
            log_info(strfmt("  workspaces gemm/moe : %s + %s",
                            human_bytes(cost.gemm_ws).c_str(),
                            human_bytes(cost.moe_ws).c_str()));
            if (tcfg.ddp)
                log_info(strfmt("  nccl/ddp extra      : %s", human_bytes(cost.nccl).c_str()));
            log_info(strfmt("  sched %s | opt %s",
                            tcfg.scheduler.c_str(), tcfg.optimizer.c_str()));
            log_info(strfmt("  TOTAL               : %s", human_bytes(total).c_str()));
            log_info(strfmt("  tokens per step     : %s",
                            human_count(static_cast<u64>(tcfg.tokens_per_step())).c_str()));
            if (tcfg.max_steps > 0)
                log_info(strfmt("  total training toks : %s (steps mode)",
                                human_count(static_cast<u64>(TrainerConfig::checked_schedule_mul(
                                    tcfg.tokens_per_step(), tcfg.max_steps,
                                    "total training tokens"))).c_str()));
            else
                log_info(strfmt("  schedule            : epochs mode (%d epochs; steps from data size)",
                                tcfg.epochs));

            if (args.has("max-vram-mb")) {
                const i64 budget_mb = args.num_int("max-vram-mb", 0);
                if (budget_mb <= 0) GAI_FAIL("--max-vram-mb needs a positive MiB value");
                const size_t budget = static_cast<size_t>(budget_mb) * 1024u * 1024u;
                const double used_pct = 100.0 * static_cast<double>(total) / static_cast<double>(budget);
                const double margin_pct = 100.0 - used_pct;
                log_info(strfmt("  VRAM budget         : %s (predicted %.1f%% used, margin %.1f%%)",
                                human_bytes(budget).c_str(), used_pct, margin_pct));
                if (total > budget)
                    GAI_FAIL(strfmt("predicted peak %s exceeds the VRAM budget %s by %s; "
                                    "reduce batch/seq/chunks, enable activation checkpointing, "
                                    "or pick a smaller recipe",
                                    human_bytes(total).c_str(), human_bytes(budget).c_str(),
                                    human_bytes(total - budget).c_str()));
                if (margin_pct < 10.0)
                    GAI_FAIL(strfmt("predicted peak %s leaves only %.1f%% headroom under %s "
                                    "(need >=10%%: allocator fragmentation + NCCL + cuBLAS "
                                    "workspaces push the real peak higher)",
                                    human_bytes(total).c_str(), margin_pct,
                                    human_bytes(budget).c_str()));
            }

            {
                const i64 out_mb = args.has("output-budget-mb")
                    ? args.num_int("output-budget-mb", 0) : tcfg.output_budget_mb;
                if (out_mb > 0) {
                    const size_t per_snap = cost.snapshot;

                    const size_t proj = cost.output_projection;
                    const size_t budget = static_cast<size_t>(out_mb) * 1024u * 1024u;
                    const size_t headroom = (budget > proj) ? budget - proj : 0;
                    log_info(strfmt("  output budget       : %s (proj snapshots+gguf %s, %s left for existing data)",
                                    human_bytes(budget).c_str(), human_bytes(proj).c_str(),
                                    human_bytes(headroom).c_str()));
                    if (proj > budget)
                        GAI_FAIL(strfmt("projected output %s exceeds budget %s "
                                        "(2 snapshots %.1fGB + GGUF). Raise --output-budget-mb "
                                        "knowingly or shrink the model.",
                                        human_bytes(proj).c_str(), human_bytes(budget).c_str(),
                                        static_cast<double>(per_snap * 2) / 1e9));
                }
            }
            return 0;
        }

        Model model(mcfg, dev);
        model.print_parameter_report();

        if (weights_loaded_by_cli) {
            // Fine-tuning a real checkpoint: the weights come from the
            // checkpoint, not from a random init.
            GAI_CHECK(hf->load_model(model),
                      "cannot load the weights of the checkpoint");
            log_info("[mod ] the checkpoint weights are loaded");
            tcfg.pretrained_loaded = true;
        } else {
            model.init_weights(tcfg.seed);
        }
        model.enable_grad(true);

        Trainer trainer(model, tcfg);

        try {
            trainer.run();
        } catch (const std::exception& e) {
            log_error(std::string("training aborted: ") + e.what());
            return 1;
        }

        if (args.has("export")) {
            // A LoRA run trains adapters; the exported model must be a plain
            // dense checkpoint, so the adapters are folded into the base
            // weights first.
            if (model.has_lora()) {
                log_info("[expt] merging the LoRA adapters into the base weights");
                model.merge_lora();
            }
            int my_rank = 0;
            if (const char* rank_env = std::getenv("RANK")) {
                const int r = std::atoi(rank_env);
                if (r > 0) my_rank = r;
            }
            if (my_rank != 0) {
                log_info("[export] rank " + std::to_string(my_rank) +
                         ": skipping GGUF export (rank 0 owns the artifact)");
            } else {
                std::string prof = args.str("export-profile", "q8_k");

                std::string tok_path = args.str("tokenizer", cfg.get_str("tokenizer.path", ""));
                Tokenizer etok;
                if (!have_hf_tok && !tok_path.empty()) {
                    std::vector<i32> e_ids;
                    GAI_CHECK(load_tokenizer_auto(tok_path, etok, e_ids),
                              "cannot load tokenizer for export: " + tok_path);
                }
                const Tokenizer& effective_tok = have_hf_tok ? hf_tok : etok;
                GAI_CHECK(have_hf_tok || !tok_path.empty(),
                          "GGUF export needs a tokenizer: pass --tokenizer <hf-dir|tokenizer.json>");
                export_model_gguf(args.str("export"), model, effective_tok,
                                  gguf_profile_for(prof),
                                  {{"stage", cfg.get_str("training.stage", "pretrain")},
                                   {"steps", std::to_string(trainer.state().step)}},
                                  args.str("compat", "native"));
            }
        }

#ifdef GAI_CUDA
        if (model.device() == Device::CUDA) {
            cuda_ops::free_workspace();
            cuda::shutdown();
        }
#endif
        return 0;

    } catch (const std::exception& e) {
        log_error(e.what());
        return 1;
    }
}
