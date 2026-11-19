"""A COMPLETE independent reference for the Granite-4.0 forward pass
(multi-layer, Mamba-2 + attention + MoE + the multipliers), written from the
reference modelling code.

It is compared against the engine's logits on data/granite_tiny (the same
layer interleaving as granite-4.0-h-tiny).
"""
import json, os, struct, sys
import numpy as np

mdir = sys.argv[1]
eng_file = sys.argv[2]


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        out = {}
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            b, e = info["data_offsets"]
            f.seek(8 + n + b)
            out[name] = np.frombuffer(f.read(e - b), dtype="<f4").reshape(info["shape"]).copy()
    return out


cfg = json.load(open(os.path.join(mdir, "config.json"), encoding="utf-8"))
st = load_safetensors(os.path.join(mdir, "model.safetensors"))
ids = [int(v) for v in open(os.path.join(mdir, "ids.txt")).read().split()]
T = len(ids)

eps = cfg["rms_norm_eps"]
d = cfg["hidden_size"]
H = cfg["num_attention_heads"]
KV = cfg["num_key_value_heads"]
hd = d // H
E = cfg["intermediate_size"]
SH = cfg.get("shared_intermediate_size", E)
ne = cfg["num_local_experts"]
K = cfg["num_experts_per_tok"]
attn_mult = cfg.get("attention_multiplier", 0.0)
emb_mult = cfg.get("embedding_multiplier", 1.0)
res_mult = cfg.get("residual_multiplier", 1.0)
logit_scale = cfg.get("logits_scaling", 1.0)
nope = cfg.get("position_embedding_type", "rope") == "nope"
mh = cfg["mamba_n_heads"]
mdh = cfg["mamba_d_head"]
mds = cfg["mamba_d_state"]
mk = cfg["mamba_d_conv"]
d_inner = cfg["mamba_expand"] * d
conv_dim = d_inner + 2 * cfg["mamba_n_groups"] * mds
proj_size = d_inner + conv_dim + mh


def rmsnorm(x, w, eps):
    inv = 1.0 / np.sqrt((x.astype(np.float64) ** 2).mean(axis=-1, keepdims=True) + eps)
    return (x * inv * w).astype(np.float32)


def silu(x):
    return x / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def mamba_block(x, p, prefix):
    """x: [T, d] -> [T, d]"""
    w_in = st[prefix + "mamba.in_proj.weight"]
    proj = x @ w_in.T                                   # [T, proj_size]
    gate = proj[:, :d_inner]
    bc_raw = proj[:, d_inner:d_inner + conv_dim]
    dt_in = proj[:, d_inner + conv_dim:]

    conv_w = st[prefix + "mamba.conv1d.weight"]
    conv_b = st[prefix + "mamba.conv1d.bias"]
    bc = np.zeros_like(bc_raw)
    for c in range(conv_dim):
        w = conv_w[c, 0, :]
        b = conv_b[c]
        for t in range(T):
            acc = b
            for j in range(mk):
                idx = t - (mk - 1 - j)
                v = bc_raw[idx, c] if idx >= 0 else 0.0
                acc += w[j] * v
            bc[t, c] = acc / (1.0 + np.exp(-acc))

    xp = bc[:, :d_inner]
    Bm = bc[:, d_inner:d_inner + mds]
    Cm = bc[:, d_inner + mds:]
    dt = softplus(dt_in + st[prefix + "mamba.dt_bias"][None, :])
    A = -np.exp(st[prefix + "mamba.A_log"])
    D = st[prefix + "mamba.D"]

    y = np.zeros((T, d_inner), dtype=np.float32)
    for h in range(mh):
        state = np.zeros((mdh, mds), dtype=np.float64)
        for t in range(T):
            dA = np.exp(A[h] * dt[t, h])
            xv = xp[t, h * mdh:(h + 1) * mdh]
            state = dA * state + (dt[t, h] * Bm[t])[None, :] * xv[:, None]
            y[t, h * mdh:(h + 1) * mdh] = (Cm[t] @ state.T) + D[h] * xv

    yg = rmsnorm(y * silu(gate), st[prefix + "mamba.norm.weight"], eps)
    return yg @ st[prefix + "mamba.out_proj.weight"].T


def attention_block(x, p, prefix):
    q = x @ st[prefix + "self_attn.q_proj.weight"].T
    k = x @ st[prefix + "self_attn.k_proj.weight"].T
    v = x @ st[prefix + "self_attn.v_proj.weight"].T
    qh = q.reshape(T, H, hd)
    kh = k.reshape(T, KV, hd)
    vh = v.reshape(T, KV, hd)
    rep = H // KV
    kh = np.repeat(kh, rep, axis=1)
    vh = np.repeat(vh, rep, axis=1)
    scale = attn_mult if attn_mult > 0 else 1.0 / np.sqrt(hd)
    scores = np.einsum("thd,shd->hts", qh, kh) * np.float32(scale)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mask[None, :, :], -np.inf, scores)
    probs = softmax(scores, axis=-1)
    attn = np.einsum("hts,shd->thd", probs, vh).reshape(T, H * hd)
    return attn @ st[prefix + "self_attn.o_proj.weight"].T


def moe_block(x, prefix):
    """The routed experts (top-k of ne) plus the shared expert."""
    # The layout follows the checkpoint: [ne, 2E, d] and [ne, d, E] (the
    # real granite) or the flat 2-D variants. Both are normalised here.
    inl = st[prefix + "block_sparse_moe.input_linear.weight"]
    outl = st[prefix + "block_sparse_moe.output_linear.weight"]
    if inl.ndim == 3:
        inl = inl.reshape(ne * 2 * E, d)
    if outl.ndim == 3:
        outl = outl
        # [ne, d, E] -> per expert the down matrix is [d, E]; the reference
        # wants [ne*E, d] (the expert's E rows, transposed).
        outl = np.stack([outl[e].T for e in range(ne)], axis=0).reshape(ne * E, d)
    router = st[prefix + "block_sparse_moe.router.layer.weight"]   # [ne, d]
    logits = x @ router.T
    probs = softmax(logits, axis=-1)
    k = min(K, ne)
    top = np.argsort(-probs, axis=-1)[:, :k]
    w = np.take_along_axis(probs, top, axis=-1)
    w = w / w.sum(axis=-1, keepdims=True)

    out = np.zeros((T, d), dtype=np.float32)
    for t in range(T):
        for j in range(k):
            e = int(top[t, j])
            gate_w = inl[e * 2 * E: e * 2 * E + E]
            up_w = inl[e * 2 * E + E: e * 2 * E + 2 * E]
            down_w = outl[e * E: (e + 1) * E].T                     # [d, E]
            act = silu(x[t] @ gate_w.T) * (x[t] @ up_w.T)
            out[t] += w[t, j] * (act @ down_w.T)

    if SH:
        sg = st[prefix + "shared_mlp.input_linear.weight"]         # [2SH, d]
        sd = st[prefix + "shared_mlp.output_linear.weight"]        # [d, SH]
        act = silu(x @ sg[:SH].T) * (x @ sg[SH:2 * SH].T)
        out += act @ sd.T
    return out


# ---- the forward pass
emb = st["model.embed_tokens.weight"]
x = emb[ids] * np.float32(emb_mult)

for l in range(cfg["num_hidden_layers"]):
    p = "model.layers.%d." % l
    xb = rmsnorm(x, st[p + "input_layernorm.weight"], eps)
    if cfg["layer_types"][l] == "attention":
        branch = attention_block(xb, p, p)
    else:
        branch = mamba_block(xb, p, p)
    x = x + np.float32(res_mult) * branch

    xb2 = rmsnorm(x, st[p + "post_attention_layernorm.weight"], eps)
    x = x + np.float32(res_mult) * moe_block(xb2, p)

xf = rmsnorm(x, st["model.norm.weight"], eps)
logits = (xf @ emb.T) / np.float32(logit_scale)

eng = np.fromfile(eng_file, dtype="<f4").reshape(logits.shape)
diff = np.abs(logits.astype(np.float64) - eng.astype(np.float64))
scale = max(1e-9, float(np.abs(logits).max()))
print("tokens:", T, "layers:", cfg["num_hidden_layers"], cfg["layer_types"])
print("max |engine - reference| =", diff.max())
print("mean |engine - reference| =", diff.mean())
print("relative max =", diff.max() / scale)
ok = diff.max() / scale < 1e-3
print("MATCH" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
