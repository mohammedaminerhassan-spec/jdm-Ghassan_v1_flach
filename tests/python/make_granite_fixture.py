"""Builds a multi-layer Granite fixture with the SAME layer interleaving as
granite-4.0-h-tiny (mamba layers with an attention layer at the same
positions), so the whole model (the residual stream, the multipliers, the
MoE, the Mamba-2 state across the layers) can be checked against a complete
independent reference.
"""
import json, os, struct, shutil, random

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SRC = os.path.join(ROOT, "granite-documentation", "granit filles confiques")
DST = os.path.join(ROOT, "data", "granite_tiny")
os.makedirs(DST, exist_ok=True)

rnd = random.Random(31337)

# The same interleaving as Granite-4.0-H-Tiny: the attention layers are at
# 5, 15, 25, 35. We keep the first 6 layers -> the attention at 5.
N_LAYERS = 6
LAYER_TYPES = ["mamba"] * N_LAYERS
LAYER_TYPES[5] = "attention"


def vals(shape, scale=0.05):
    n = 1
    for s in shape:
        n *= s
    return [rnd.gauss(0.0, scale) for _ in range(n)]


def build_safetensors(path, tensors):
    header, offset, blobs = {}, 0, []
    for name, (shape, v) in tensors.items():
        b = struct.pack("<%df" % len(v), *v)
        header[name] = {"dtype": "F32", "shape": shape,
                        "data_offsets": [offset, offset + len(b)]}
        offset += len(b)
        blobs.append(b)
    hdr = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hdr += b" " * ((8 - (len(hdr) % 8)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        for b in blobs:
            f.write(b)


V, D = 100352, 32
H, KV = 2, 1
HD = D // H
QD, KVD = H * HD, KV * HD
E, NE, K = 16, 2, 1
SH = 24
MH, MDH, MDS, MG, MK, ME = 2, 32, 16, 1, 4, 2
D_INNER = ME * D
CONV_DIM = D_INNER + 2 * MG * MDS
PROJ = D_INNER + CONV_DIM + MH

tensors = {
    "model.embed_tokens.weight": ((V, D), vals((V, D), 0.004)),
    "model.norm.weight":         ((D,), vals((D,))),
}
for l in range(N_LAYERS):
    p = "model.layers.%d." % l
    tensors[p + "input_layernorm.weight"] = ((D,), vals((D,)))
    tensors[p + "post_attention_layernorm.weight"] = ((D,), vals((D,)))
    if LAYER_TYPES[l] == "attention":
        tensors[p + "self_attn.q_proj.weight"] = ((QD, D), vals((QD, D)))
        tensors[p + "self_attn.k_proj.weight"] = ((KVD, D), vals((KVD, D)))
        tensors[p + "self_attn.v_proj.weight"] = ((KVD, D), vals((KVD, D)))
        tensors[p + "self_attn.o_proj.weight"] = ((D, QD), vals((D, QD)))
    else:
        tensors[p + "mamba.in_proj.weight"] = ((PROJ, D), vals((PROJ, D)))
        tensors[p + "mamba.conv1d.weight"] = ((CONV_DIM, 1, MK), vals((CONV_DIM, 1, MK), 0.3))
        tensors[p + "mamba.conv1d.bias"] = ((CONV_DIM,), vals((CONV_DIM,), 0.1))
        tensors[p + "mamba.dt_bias"] = ((MH,), vals((MH,), 0.1))
        tensors[p + "mamba.A_log"] = ((MH,), vals((MH,), 0.2))
        tensors[p + "mamba.D"] = ((MH,), vals((MH,), 0.2))
        tensors[p + "mamba.norm.weight"] = ((D_INNER,), vals((D_INNER,)))
        tensors[p + "mamba.out_proj.weight"] = ((D, D_INNER), vals((D, D_INNER)))
    tensors[p + "block_sparse_moe.router.layer.weight"] = ((NE, D), vals((NE, D)))
    # The REAL checkpoint stores the experts in 3D: [ne, 2E, d] and
    # [ne, d, E]. The engine has a branch for each layout, so the
    # fixture must use the real one.
    tensors[p + "block_sparse_moe.input_linear.weight"] = ((NE, 2 * E, D), vals((NE, 2 * E, D)))
    tensors[p + "block_sparse_moe.output_linear.weight"] = ((NE, D, E), vals((NE, D, E)))
    tensors[p + "shared_mlp.input_linear.weight"] = ((2 * SH, D), vals((2 * SH, D)))
    tensors[p + "shared_mlp.output_linear.weight"] = ((D, SH), vals((D, SH)))

build_safetensors(os.path.join(DST, "model.safetensors"), tensors)
with open(os.path.join(DST, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
    json.dump({"metadata": {"total_size": 0},
               "weight_map": {k: "model.safetensors" for k in tensors}}, f)

with open(os.path.join(SRC, "config.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)
# The official config is the template (the multipliers, the mamba defaults,
# the tokenizer settings); the geometry is the tiny one.
cfg["hidden_size"] = D
cfg["vocab_size"] = V
cfg["num_hidden_layers"] = N_LAYERS
cfg["layer_types"] = LAYER_TYPES
cfg["intermediate_size"] = E
cfg["shared_intermediate_size"] = SH
cfg["num_local_experts"] = NE
cfg["num_experts_per_tok"] = K
cfg["num_attention_heads"] = H
cfg["num_key_value_heads"] = KV
cfg["mamba_n_heads"] = MH
cfg["mamba_d_head"] = MDH
cfg["mamba_d_state"] = MDS
cfg["mamba_n_groups"] = MG
cfg["mamba_d_conv"] = MK
cfg["mamba_expand"] = ME
cfg["residual_multiplier"] = 1.0
with open(os.path.join(DST, "config.json"), "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, sort_keys=True)

for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "vocab.json", "merges.txt", "generation_config.json", "chat_template.jinja"):
    s = os.path.join(SRC, name)
    if os.path.exists(s):
        shutil.copy2(s, os.path.join(DST, name))

print("granite_tiny written:", DST)
print("the layers:", LAYER_TYPES)
