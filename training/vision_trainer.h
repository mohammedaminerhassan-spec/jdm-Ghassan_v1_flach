#pragma once

#include "model/model.h"
#include "projector/projector.h"
#include "tokenizer/tokenizer.h"
#include "training/optimizer.h"

#include <string>
#include <vector>

namespace gai {

struct VisionSample {
    std::string image_path;
    std::string video_path;
    std::string prompt;
    std::string response;
};

struct VisionTrainConfig {
    std::string jsonl_path;
    std::string save_mmproj;
    int   max_steps   = 100;
    int   seq_len     = 512;
    int   max_frames  = 8;
    int   log_every   = 5;
    int   save_every  = 50;
    float vision_lr   = 1e-4f;
    float llm_lr      = 2e-5f;
    float weight_decay = 0.01f;
    bool  train_llm   = true;
    u64   seed        = 42;
};

struct VisionTrainStats {
    int    steps_done = 0;
    double first_loss = 0.0;
    double last_loss  = 0.0;
};

std::vector<VisionSample> load_vision_jsonl(const std::string& path);

VisionTrainStats run_vision_training(Model& model,
                                     VisionTower& vision,
                                     const Tokenizer& tok,
                                     const VisionTrainConfig& cfg);

}
