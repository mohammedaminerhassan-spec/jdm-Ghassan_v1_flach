#include "training/vision_trainer.h"
#include "core/json.h"
#include "core/ops.h"
#include "core/device.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <memory>
#include <vector>

namespace gai {

std::vector<VisionSample> load_vision_jsonl(const std::string& path) {
    std::ifstream f(path);
    GAI_CHECK(f.good(), "cannot open vision JSONL dataset: " + path);

    std::vector<VisionSample> samples;
    std::string line;
    int line_no = 0;
    while (std::getline(f, line)) {
        ++line_no;
        if (line.empty()) continue;
        size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;

        Json j;
        std::string err;
        GAI_CHECK(Json::parse(line, j, &err),
                  strfmt("vision JSONL parse error at %s:%d: %s",
                         path.c_str(), line_no, err.c_str()));
        GAI_CHECK(j.is_object(),
                  strfmt("vision JSONL line %d must be a JSON object", line_no));

        VisionSample s;
        if (j.has("image")) s.image_path = j.at("image").as_string();
        if (j.has("video")) s.video_path = j.at("video").as_string();

        if (j.has("prompt"))        s.prompt = j.at("prompt").as_string();
        else if (j.has("question")) s.prompt = j.at("question").as_string();
        else if (j.has("user"))     s.prompt = j.at("user").as_string();

        if (j.has("response"))       s.response = j.at("response").as_string();
        else if (j.has("answer"))    s.response = j.at("answer").as_string();
        else if (j.has("assistant")) s.response = j.at("assistant").as_string();
        else if (j.has("text"))      s.response = j.at("text").as_string();

        GAI_CHECK(!s.response.empty(),
                  strfmt("vision JSONL line %d has empty response/text", line_no));
        samples.push_back(std::move(s));
    }
    GAI_CHECK(!samples.empty(), "vision JSONL dataset is empty: " + path);
    return samples;
}

VisionTrainStats run_vision_training(Model& model,
                                     VisionTower& vision,
                                     const Tokenizer& tok,
                                     const VisionTrainConfig& cfg) {
    GAI_CHECK(vision.valid(), "run_vision_training: VisionTower is not valid");
    GAI_CHECK(cfg.max_steps > 0, "run_vision_training: max_steps must be > 0");
    GAI_CHECK(cfg.seq_len > 8, "run_vision_training: seq_len must be > 8");

    const Device dev = model.device();
    if (vision.device() != dev) {
        vision.to(dev);
    }

    const std::vector<VisionSample> samples = load_vision_jsonl(cfg.jsonl_path);
    const VisionConfig& vcfg = vision.config();
    const int d = model.config().hidden_size;
    GAI_CHECK(vcfg.out_dim == d,
              strfmt("VisionTower out_dim (%d) != model hidden_size (%d)", vcfg.out_dim, d));

    const int inter = (vcfg.proj_hidden > 0) ? vcfg.proj_hidden : vcfg.out_dim;

    model.enable_grad(true);

    std::unique_ptr<AdamW> llm_opt;
    bool has_trainable_llm = false;
    if (cfg.train_llm) {
        for (Parameter* p : model.parameters()) {
            if (!p->frozen && p->g.defined()) {
                has_trainable_llm = true;
                break;
            }
        }
        if (has_trainable_llm) {
            AdamWConfig ocfg;
            ocfg.lr = cfg.llm_lr;
            ocfg.weight_decay = cfg.weight_decay;
            llm_opt = std::make_unique<AdamW>(model, ocfg);
        }
    }

    ProjectorGradients proj_grads;
    ProjectorAdamWState proj_opt;

    Activations act = model.make_activations(1, cfg.seq_len, true);

    const i32 pad_id = std::max<i32>(0, tok.pad_id());
    const i32 bos_id = tok.bos_id();
    const i32 eos_id = tok.eos_id();

    Tensor ids_t = Tensor::empty({1, cfg.seq_len}, DType::I32, dev);
    Tensor tgt_t = Tensor::empty({1, cfg.seq_len}, DType::I32, dev);
    Tensor vis_embeds;
    Tensor saved_h1;
    Tensor saved_act;
    Tensor d_vis_out;
    int prev_n_vis = 0;

    VisionTrainStats stats;
    auto t0 = std::chrono::steady_clock::now();

    for (int step = 1; step <= cfg.max_steps; ++step) {
        const VisionSample& s = samples[static_cast<size_t>((step - 1) % static_cast<int>(samples.size()) )];

        Tensor pooled;
        if (!s.video_path.empty()) {
            VideoBuffer vid = load_video_any(s.video_path, cfg.max_frames, vcfg.image_size, vcfg.image_size);
            GAI_CHECK(vid.valid && !vid.frames.empty(),
                      "failed to decode video: " + s.video_path);
            std::vector<const float*> ptrs;
            ptrs.reserve(vid.frames.size());
            for (const auto& fr : vid.frames) ptrs.push_back(fr.data());
            vision.encode_video_pooled_features(ptrs, pooled);
        } else if (!s.image_path.empty()) {
            ImageBuffer img = load_image_any(s.image_path, vcfg.image_size, vcfg.image_size);
            GAI_CHECK(img.valid, "failed to decode image: " + s.image_path);
            vision.encode_pooled_features(img.chw_data.data(), pooled);
        } else {
            GAI_FAIL("vision sample must specify either 'image' or 'video'");
        }

        const int n_vis = static_cast<int>(pooled.shape()[0]);
        GAI_CHECK(n_vis + 4 < cfg.seq_len,
                  strfmt("seq_len (%d) too small for %d visual tokens", cfg.seq_len, n_vis));

        if (n_vis != prev_n_vis || !vis_embeds.defined()) {
            vis_embeds = Tensor::empty({static_cast<i64>(n_vis), vcfg.out_dim}, DType::F32, dev);
            saved_h1   = Tensor::empty({static_cast<i64>(n_vis), inter}, DType::F32, dev);
            saved_act  = Tensor::empty({static_cast<i64>(n_vis), inter}, DType::F32, dev);
            d_vis_out  = Tensor::empty({static_cast<i64>(n_vis), d}, DType::F32, dev);
            prev_n_vis = n_vis;
        }
        vision.projector_forward_train_dev(pooled.f32(), n_vis,
                                           vis_embeds.f32(), saved_h1.f32(), saved_act.f32());

        std::vector<i32> full_ids;
        std::vector<i32> full_targets;
        full_ids.reserve(static_cast<size_t>(cfg.seq_len) + 1);
        full_targets.reserve(static_cast<size_t>(cfg.seq_len) + 1);

        if (tok.add_bos() && bos_id >= 0) {
            full_ids.push_back(bos_id);
        }
        const int vis_pos = static_cast<int>(full_ids.size());
        for (int i = 0; i < n_vis; ++i) {
            full_ids.push_back(pad_id);
        }

        const std::vector<i32> p_ids = tok.encode(s.prompt, false, false);
        const std::vector<i32> r_ids = tok.encode(s.response, false, eos_id >= 0);

        const int prompt_end = vis_pos + n_vis + static_cast<int>(p_ids.size());
        full_ids.insert(full_ids.end(), p_ids.begin(), p_ids.end());
        full_ids.insert(full_ids.end(), r_ids.begin(), r_ids.end());

        if (static_cast<int>(full_ids.size()) > cfg.seq_len + 1) {
            full_ids.resize(static_cast<size_t>(cfg.seq_len + 1));
        }

        std::vector<i32> h_ids(static_cast<size_t>(cfg.seq_len), pad_id);
        std::vector<i32> h_tgt(static_cast<size_t>(cfg.seq_len), -100);
        const int avail = static_cast<int>(full_ids.size());
        for (int t = 0; t < cfg.seq_len && t < avail; ++t) {
            h_ids[static_cast<size_t>(t)] = full_ids[static_cast<size_t>(t)];
            if (t + 1 < avail && t + 1 >= prompt_end) {
                h_tgt[static_cast<size_t>(t)] = full_ids[static_cast<size_t>(t + 1)];
            }
        }

        device_copy(ids_t.ptr<i32>(), dev, h_ids.data(), Device::CPU,
                    sizeof(i32) * static_cast<size_t>(cfg.seq_len));
        device_copy(tgt_t.ptr<i32>(), dev, h_tgt.data(), Device::CPU,
                    sizeof(i32) * static_cast<size_t>(cfg.seq_len));

        if (llm_opt) model.zero_grad();

        i64 ntok = 0;
        double loss = model.forward_backward(
            ids_t.ptr<i32>(), tgt_t.ptr<i32>(), 1, cfg.seq_len, act, &ntok,
            1.0f, false, nullptr, h_tgt.data(),
            vis_embeds.f32(), vis_pos, n_vis, d_vis_out.f32());

        if (step == 1) stats.first_loss = loss;
        stats.last_loss  = loss;
        stats.steps_done = step;

        vision.projector_backward_dev(pooled.f32(), saved_h1.f32(), saved_act.f32(),
                                      d_vis_out.f32(), n_vis, proj_grads);
        vision.projector_adamw_step(proj_grads, proj_opt,
                                    cfg.vision_lr, 0.9f, 0.999f, 1e-8f, cfg.weight_decay);

        if (llm_opt) {
            llm_opt->step(cfg.llm_lr, 1.0f);
        }

        if (cfg.log_every > 0 && (step == 1 || step % cfg.log_every == 0 || step == cfg.max_steps)) {
            auto now = std::chrono::steady_clock::now();
            double sec = std::chrono::duration<double>(now - t0).count();
            log_info(strfmt("[vision-train] step %4d/%d | loss %.4f | vis_tok=%d | %.2fs",
                            step, cfg.max_steps, loss, n_vis, sec));
        }
        if (!cfg.save_mmproj.empty() && cfg.save_every > 0 && step % cfg.save_every == 0) {
            vision.save_projector_checkpoint(cfg.save_mmproj);
        }
    }

    if (!cfg.save_mmproj.empty()) {
        GAI_CHECK(vision.save_projector_checkpoint(cfg.save_mmproj),
                  "failed to save trained projector checkpoint to " + cfg.save_mmproj);
        log_info("[vision-train] saved projector checkpoint: " + cfg.save_mmproj);
    }

    return stats;
}

}
