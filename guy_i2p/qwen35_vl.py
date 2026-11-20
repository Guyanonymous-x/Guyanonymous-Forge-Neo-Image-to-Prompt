# Guyanonymous' Image to Prompt
# Qwen3.5-9B as a vision-language model: its vision tower reads the picture, its language tower (the same one MOSS-Image
# encodes prompts with) writes text about it or scores given wordings against it.
#
# Plain torch, no transformers (Forge Neo's transformers 4.57 has no Qwen3.5). Layer maths follows the MOSS-Image
# extension's qwen35.py (ported from ComfyUI) and Forge Neo's backend/nn/llm/qwen35.py vision tower, with what those
# leave out added here: the KV cache and recurrent Gated DeltaNet state for decoding, interleaved MRoPE positions for
# image tokens, the lm_head and sampling.
#
# Weights: the INT8 file (int8_tensorwise) is used as it is; a bf16/fp16 file has its big text-layer weights turned
# into per-row INT8 while loading, so either fits a 16 GB card. The token embedding table stays in system RAM.

import math

import torch
import torch.nn.functional as F
from safetensors import safe_open

# Qwen3.5-9B
V_HIDDEN, V_DEPTH, V_HEADS, V_PATCH, V_TPS, V_MERGE, V_NPOS = 1152, 27, 16, 16, 2, 2, 2304
T_LAYERS, T_HIDDEN = 32, 4096
A_HEADS, A_KV, A_DIM, ROT_DIM, THETA = 16, 4, 256, 64, 10000000.0
MROPE_SECTION = (11, 11, 10)
D_KHEADS, D_VHEADS, D_KDIM, D_VDIM, CONV_K = 16, 32, 128, 128, 4
EPS = 1e-6

IM_START, IM_END, ENDOFTEXT = 248045, 248046, 248044
VISION_START, VISION_END, IMAGE_PAD = 248053, 248054, 248056
THINK_END = 248069
STOP_IDS = {IM_END, ENDOFTEXT}

DTYPE = torch.bfloat16


# ---------------------------------------------------------------- small maths

def rms(x, w, add=True, eps=EPS):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * ((1.0 + w) if add else w)).to(x.dtype)


def l2norm(x):
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def partial_rope(x, cos, sin):
    d = cos.shape[-1]
    rot, rest = x[..., :d].float(), x[..., d:]
    return torch.cat(((rot * cos + rotate_half(rot) * sin).to(x.dtype), rest), dim=-1)


def chunk_gated_delta(query, key, value, g, beta, state=None, chunk_size=64):
    """Gated delta rule over T tokens, from `state` (B, H, Dk, Dv) or zeros. Inputs (B, T, H, D) / (B, T, H).
    Returns (B, T, H, Dv) float32 and the final state."""
    batch, length = key.shape[:2]
    heads, v_dim = value.shape[-2:]
    k_dim = key.shape[-1]
    query, key, value, beta, decay = (t.transpose(1, 2).float().contiguous() for t in (query, key, value, beta, g))
    query = l2norm(query) * k_dim ** -0.5
    key = l2norm(key)
    pad = (chunk_size - length % chunk_size) % chunk_size
    query, key, value = (F.pad(t, (0, 0, 0, pad)) for t in (query, key, value))
    beta, decay = (F.pad(t, (0, pad)) for t in (beta, decay))
    chunks = (length + pad) // chunk_size
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = (t.reshape(batch, heads, chunks, chunk_size, -1) for t in (query, key, k_beta, v_beta))
    cum = decay.reshape(batch, heads, chunks, chunk_size).cumsum(dim=3)
    upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    pair = (cum.unsqueeze(4) - cum.unsqueeze(3)).masked_fill(upper, float("-inf")).exp()
    ut = (k_beta @ key.transpose(-1, -2)) * pair
    intra = (query @ key.transpose(-1, -2)) * pair
    new_v = torch.linalg.solve_triangular(ut, v_beta, upper=False, unitriangular=True)
    k_cum = torch.linalg.solve_triangular(ut, k_beta * cum.exp().unsqueeze(-1), upper=False, unitriangular=True)
    query = query * cum.exp().unsqueeze(-1)
    key = key * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    chunk_decay = cum[..., -1].exp()[..., None, None]
    if state is None:
        state = query.new_zeros(batch, heads, k_dim, v_dim)
    out = torch.empty_like(new_v)
    for i in range(chunks):
        vn = new_v[:, :, i] - k_cum[:, :, i] @ state
        out[:, :, i] = query[:, :, i] @ state + intra[:, :, i] @ vn
        state = state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ vn
    return out.reshape(batch, heads, -1, v_dim)[:, :, :length].transpose(1, 2), state


def recurrent_gated_delta(query, key, value, g, beta, state):
    """One step at a time (decoding), same maths as transformers' torch_recurrent_gated_delta_rule."""
    batch, length = key.shape[:2]
    k_dim = key.shape[-1]
    query, key, value, beta, g = (t.transpose(1, 2).float() for t in (query, key, value, beta, g))
    query = l2norm(query) * k_dim ** -0.5
    key = l2norm(key)
    if state is None:
        state = query.new_zeros(batch, query.shape[1], k_dim, value.shape[-1])
    out = torch.empty_like(value)
    for i in range(length):
        state = state * g[:, :, i].exp()[..., None, None]
        k_t = key[:, :, i]
        kv = (state * k_t[..., None]).sum(-2)
        delta = (value[:, :, i] - kv) * beta[:, :, i][..., None]
        state = state + k_t[..., None] * delta[..., None, :]
        out[:, :, i] = (state * query[:, :, i][..., None]).sum(-2)
    return out.transpose(1, 2), state


# ---------------------------------------------------------------- image preparation

def smart_resize(h, w, factor=32, min_pixels=64 * 1024, max_pixels=1024 * 1024):
    hb = max(factor, round(h / factor) * factor)
    wb = max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb = max(factor, math.floor(h / beta / factor) * factor)
        wb = max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return hb, wb


def image_patches(img, max_tokens=1024):
    """PIL image -> (patches (N, 1536) float32, grid (1, gh, gw)), the Qwen3-VL image processor's layout
    (mean and std 0.5, bicubic resize to multiples of 32, two identical frames)."""
    from PIL import Image
    import numpy as np

    img = img.convert("RGB")
    w, h = img.size
    hb, wb = smart_resize(h, w, max_pixels=max_tokens * 1024)
    img = img.resize((wb, hb), Image.BICUBIC)
    a = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    a = (a - 0.5) / 0.5
    frames = a[None].repeat(V_TPS, 1, 1, 1)  # (T=2, C, H, W)
    gh, gw = hb // V_PATCH, wb // V_PATCH
    m, p = V_MERGE, V_PATCH
    x = frames.reshape(1, V_TPS, 3, gh // m, m, p, gw // m, m, p)
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return x.reshape(gh * gw, 3 * V_TPS * p * p), (1, gh, gw)


# ---------------------------------------------------------------- cache

class Cache:
    def __init__(self):
        self.layers = [dict() for _ in range(T_LAYERS)]
        self.next_pos = 0

    def clone(self):
        c = Cache()
        c.next_pos = self.next_pos
        c.layers = [{k: v.clone() for k, v in d.items()} for d in self.layers]
        return c


# ---------------------------------------------------------------- model

class Qwen35VL:
    def __init__(self, path, device="cuda", progress=None):
        self.device = torch.device(device)
        self.path = path
        self.w = {}
        self._load(path, progress)

    # ---- loading
    def _load(self, path, progress):
        w = self.w
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            if not any(k.startswith("model.visual.") for k in keys):
                raise ValueError("This file has no vision part (no model.visual.* weights), so it cannot read images.")
            if "lm_head.weight" not in keys:
                raise ValueError("This file has no lm_head, so it cannot write text.")
            wanted = [k for k in keys if (k.startswith(("model.language_model.", "model.visual.")) or k == "lm_head.weight")
                      and not k.endswith((".comfy_quant", ".input_scale", ".weight_scale"))]
            for n, k in enumerate(wanted):
                if progress and n % 50 == 0:
                    progress(n / len(wanted), f"Loading weights {n}/{len(wanted)}")
                name = k.replace("model.language_model.", "lm.").replace("model.visual.", "vis.")
                t = f.get_tensor(k)
                if name == "lm.embed_tokens.weight":
                    w[name] = t.to(DTYPE)  # stays in system RAM; looked up a few tokens at a time
                elif t.dtype == torch.int8:
                    s = f.get_tensor(k[: -len("weight")] + "weight_scale").float()
                    w[name] = (t.to(self.device), s.to(self.device))
                elif t.ndim < 2 or "norm" in k or "conv1d" in k or k.endswith(("A_log", "dt_bias")):
                    w[name] = t.float().to(self.device)
                elif name.startswith("lm.layers.") and t.ndim == 2 and t.numel() >= (1 << 20):
                    g = t.to(self.device).float()
                    s = (g.abs().amax(1, keepdim=True) / 127.0).clamp(min=1e-10)
                    w[name] = ((g / s).round().clamp(-127, 127).to(torch.int8), s)
                    del g
                else:
                    w[name] = t.to(self.device, DTYPE)
                del t
        w["vis.patch_embed.proj.weight"] = w["vis.patch_embed.proj.weight"].reshape(V_HIDDEN, -1)
        torch.cuda.empty_cache()

    def unload(self):
        self.w.clear()
        torch.cuda.empty_cache()

    # ---- helpers
    def lin(self, x, name):
        wt = self.w[name + ".weight"]
        if isinstance(wt, tuple):
            q, s = wt
            wt = q.to(x.dtype) * s.to(x.dtype)
        b = self.w.get(name + ".bias")
        return F.linear(x, wt, None if b is None else b.to(x.dtype))

    def ln(self, x, name):
        return F.layer_norm(x.float(), (x.shape[-1],), self.w[name + ".weight"], self.w[name + ".bias"], 1e-6).to(x.dtype)

    def embed(self, ids):
        return self.w["lm.embed_tokens.weight"][torch.as_tensor(ids, dtype=torch.long)].to(self.device)[None]

    # ---- vision tower
    def _pos_interp(self, gh, gw):
        n = int(V_NPOS ** 0.5)
        pe = self.w["vis.pos_embed.weight"]
        dev = pe.device
        hi = torch.linspace(0, n - 1, gh, device=dev)
        wi = torch.linspace(0, n - 1, gw, device=dev)
        hf, wf = hi.int(), wi.int()
        hc, wc = (hf + 1).clip(max=n - 1), (wf + 1).clip(max=n - 1)
        dh, dw = (hi - hf)[:, None], (wi - wf)[None]
        out = (pe[(hf[:, None] * n + wf[None]).flatten()] * ((1 - dh) * (1 - dw)).flatten()[:, None].to(pe.dtype)
               + pe[(hf[:, None] * n + wc[None]).flatten()] * ((1 - dh) * dw).flatten()[:, None].to(pe.dtype)
               + pe[(hc[:, None] * n + wf[None]).flatten()] * (dh * (1 - dw)).flatten()[:, None].to(pe.dtype)
               + pe[(hc[:, None] * n + wc[None]).flatten()] * (dh * dw).flatten()[:, None].to(pe.dtype))
        m = V_MERGE
        return out.view(gh // m, m, gw // m, m, -1).permute(0, 2, 1, 3, 4).reshape(gh * gw, -1)

    def _vis_rope(self, gh, gw):
        dim = V_HIDDEN // V_HEADS // 2
        inv = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32, device=self.device) / dim))
        m = V_MERGE
        rows = (torch.arange(gh // m, device=self.device)[:, None, None, None] * m
                + torch.arange(m, device=self.device)[None, None, :, None]).expand(gh // m, gw // m, m, m).reshape(-1)
        cols = (torch.arange(gw // m, device=self.device)[None, :, None, None] * m
                + torch.arange(m, device=self.device)[None, None, None, :]).expand(gh // m, gw // m, m, m).reshape(-1)
        emb = torch.cat((rows[:, None].float() * inv, cols[:, None].float() * inv), dim=-1)
        emb = torch.cat((emb, emb), dim=-1)
        return emb.cos(), emb.sin()

    @torch.inference_mode()
    def vision(self, patches, grid):
        _, gh, gw = grid
        x = F.linear(patches.to(self.device, DTYPE), self.w["vis.patch_embed.proj.weight"],
                     self.w["vis.patch_embed.proj.bias"].to(DTYPE))
        x = x + self._pos_interp(gh, gw).to(x.dtype)
        cos, sin = self._vis_rope(gh, gw)
        n = x.shape[0]
        hd = V_HIDDEN // V_HEADS
        for i in range(V_DEPTH):
            p = f"vis.blocks.{i}."
            h = self.ln(x, p + "norm1")
            q, k, v = self.lin(h, p + "attn.qkv").view(n, 3, V_HEADS, hd).permute(1, 2, 0, 3).unbind(0)
            q = (q.float() * cos + rotate_half(q.float()) * sin).to(x.dtype)
            k = (k.float() * cos + rotate_half(k.float()) * sin).to(x.dtype)
            o = F.scaled_dot_product_attention(q[None], k[None], v[None])[0]
            x = x + self.lin(o.permute(1, 0, 2).reshape(n, -1), p + "attn.proj")
            h = self.ln(x, p + "norm2")
            x = x + self.lin(F.gelu(self.lin(h, p + "mlp.linear_fc1"), approximate="tanh"), p + "mlp.linear_fc2")
        x = self.ln(x, "vis.merger.norm").view(-1, V_HIDDEN * V_MERGE * V_MERGE)
        return self.lin(F.gelu(self.lin(x, "vis.merger.linear_fc1")), "vis.merger.linear_fc2")

    # ---- language tower
    def _rope(self, pos3):
        inv = 1.0 / (THETA ** (torch.arange(0, ROT_DIM, 2, dtype=torch.float32, device=self.device) / ROT_DIM))
        fr = pos3.to(self.device).float()[..., None] * inv  # (3, T, 32)
        ft = fr[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            idx = slice(offset, MROPE_SECTION[axis] * 3, 3)
            ft[..., idx] = fr[axis, ..., idx]
        emb = torch.cat((ft, ft), dim=-1)
        return emb.cos(), emb.sin()

    def _attn(self, x, p, cos, sin, c):
        b, t, _ = x.shape
        q, gate = self.lin(x, p + "q_proj").view(b, t, A_HEADS, 2 * A_DIM).chunk(2, dim=-1)
        gate = gate.reshape(b, t, -1)
        q = rms(q, self.w[p + "q_norm.weight"]).transpose(1, 2)
        k = rms(self.lin(x, p + "k_proj").view(b, t, A_KV, A_DIM), self.w[p + "k_norm.weight"]).transpose(1, 2)
        v = self.lin(x, p + "v_proj").view(b, t, A_KV, A_DIM).transpose(1, 2)
        q, k = partial_rope(q, cos, sin), partial_rope(k, cos, sin)
        if "k" in c:
            k = torch.cat((c["k"], k), dim=2)
            v = torch.cat((c["v"], v), dim=2)
        c["k"], c["v"] = k, v
        past = k.shape[2] - t
        mask, causal = None, False
        if t > 1 and past == 0:
            causal = True
        elif t > 1:
            mask = (torch.arange(past + t, device=x.device)[None] <= (past + torch.arange(t, device=x.device))[:, None])
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, enable_gqa=True)
        o = o.transpose(1, 2).reshape(b, t, -1) * gate.sigmoid()
        return self.lin(o, p + "o_proj")

    def _delta(self, x, p, c):
        b, t, _ = x.shape
        kd, vd = D_KHEADS * D_KDIM, D_VHEADS * D_VDIM
        z = self.lin(x, p + "in_proj_z")
        beta = self.lin(x, p + "in_proj_b").float().sigmoid()
        a = self.lin(x, p + "in_proj_a").float()
        mixed = self.lin(x, p + "in_proj_qkv").transpose(1, 2).float()  # (B, C, T)
        prev = c.get("conv")
        if prev is None:
            prev = mixed.new_zeros(b, mixed.shape[1], CONV_K - 1)
        inp = torch.cat((prev, mixed), dim=-1)
        c["conv"] = inp[:, :, -(CONV_K - 1):].clone()
        qkv = F.silu(F.conv1d(inp, self.w[p + "conv1d.weight"], groups=inp.shape[1])).transpose(1, 2).to(x.dtype)
        q, k, v = qkv.split([kd, kd, vd], dim=-1)
        rep = D_VHEADS // D_KHEADS
        q = q.reshape(b, t, D_KHEADS, D_KDIM).repeat_interleave(rep, dim=2)
        k = k.reshape(b, t, D_KHEADS, D_KDIM).repeat_interleave(rep, dim=2)
        v = v.reshape(b, t, D_VHEADS, D_VDIM)
        g = -self.w[p + "A_log"].exp() * F.softplus(a + self.w[p + "dt_bias"])
        state = c.get("state")
        if t == 1 and state is not None:
            out, state = recurrent_gated_delta(q, k, v, g, beta, state)
        else:
            out, state = chunk_gated_delta(q, k, v, g, beta, state)
        c["state"] = state
        out = rms(out.reshape(-1, D_VDIM), self.w[p + "norm.weight"], add=False).float()
        out = (out * F.silu(z.reshape(-1, D_VDIM).float())).to(x.dtype)
        return self.lin(out.reshape(b, t, vd), p + "out_proj")

    @torch.inference_mode()
    def forward(self, x, pos3, cache, all_logits=False):
        """x (1, T, 4096) embeddings, pos3 (3, T) positions. Returns float logits (T, V) or (1, V)."""
        x = x.to(self.device, DTYPE)
        cos, sin = self._rope(pos3)
        for i in range(T_LAYERS):
            p = f"lm.layers.{i}."
            h = rms(x, self.w[p + "input_layernorm.weight"])
            if (i + 1) % 4 == 0:
                h = self._attn(h, p + "self_attn.", cos, sin, cache.layers[i])
            else:
                h = self._delta(h, p + "linear_attn.", cache.layers[i])
            x = x + h
            h = rms(x, self.w[p + "post_attention_layernorm.weight"])
            x = x + self.lin(F.silu(self.lin(h, p + "mlp.gate_proj")) * self.lin(h, p + "mlp.up_proj"), p + "mlp.down_proj")
        x = rms(x, self.w["lm.norm.weight"])
        if not all_logits:
            x = x[:, -1:]
        return F.linear(x, self.w["lm_head.weight"])[0].float()

    # ---- prompts
    def prefill(self, tokenizer, image, text, max_tokens=1024, thinking=False):
        """Runs '<user>[image]text<assistant>' through the model. Returns (cache, last logits)."""
        enc = lambda s: tokenizer.encode(s, add_special_tokens=False).ids
        head = [IM_START] + enc("user\n")
        tail = enc(text) + [IM_END] + enc("\n") + [IM_START] + enc("assistant\n")
        tail += enc("<think>\n") if thinking else enc("<think>\n\n</think>\n\n")
        pieces, positions = [], []
        pos = 0
        if image is not None:
            patches, grid = image_patches(image, max_tokens)
            vis = self.vision(patches, grid)
            ids = head + [VISION_START]
            pieces.append(self.embed(ids))
            positions.append(torch.arange(pos, pos + len(ids)).expand(3, -1))
            pos += len(ids)
            _, gh, gw = grid
            lh, lw = gh // V_MERGE, gw // V_MERGE
            pieces.append(vis[None])
            hh = torch.arange(lh).view(-1, 1).expand(lh, lw).reshape(-1)
            ww = torch.arange(lw).view(1, -1).expand(lh, lw).reshape(-1)
            positions.append(torch.stack((torch.zeros_like(hh), hh, ww)) + pos)
            pos += max(lh, lw)
            ids = [VISION_END] + tail
        else:
            ids = head + tail
        pieces.append(self.embed(ids))
        positions.append(torch.arange(pos, pos + len(ids)).expand(3, -1))
        pos += len(ids)
        cache = Cache()
        logits = self.forward(torch.cat(pieces, dim=1), torch.cat(positions, dim=1), cache)
        cache.next_pos = pos
        return cache, logits[-1]

    def step(self, ids, cache, all_logits=False):
        n = len(ids)
        pos3 = torch.arange(cache.next_pos, cache.next_pos + n).expand(3, -1)
        cache.next_pos += n
        return self.forward(self.embed(ids), pos3, cache, all_logits)

    @torch.inference_mode()
    def generate(self, tokenizer, image, text, max_new=400, temperature=0.7, top_p=0.8, repetition_penalty=1.05,
                 seed=-1, max_tokens=1024, thinking=False, should_stop=None, on_text=None):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed if seed is not None and seed >= 0 else torch.seed())
        cache, logits = self.prefill(tokenizer, image, text, max_tokens, thinking)
        out = []
        for i in range(max_new):
            if should_stop and should_stop():
                break
            tok = sample(logits, out, temperature, top_p, repetition_penalty, gen)
            if tok in STOP_IDS:
                break
            out.append(tok)
            if on_text and i % 16 == 0:
                on_text(tokenizer.decode(out, skip_special_tokens=True))
            logits = self.step([tok], cache)[-1]
        del cache
        if THINK_END in out:  # keep only the answer after the model's thinking
            out = out[out.index(THINK_END) + 1:]
        return tokenizer.decode(out, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def score(self, tokenizer, image, question, wordings, max_tokens=1024, should_stop=None):
        """For each wording: (mean log-probability per token, total, token count) of the model answering `question`
        about the image with exactly that wording."""
        base, last = self.prefill(tokenizer, image, question, max_tokens)
        res = []
        for wording in wordings:
            if should_stop and should_stop():
                break
            ids = tokenizer.encode(wording, add_special_tokens=False).ids
            if not ids:
                res.append((wording, float("nan"), float("nan"), 0))
                continue
            c = base.clone()
            logits = self.step(ids, c, all_logits=True)
            lp = torch.log_softmax(torch.cat((last[None], logits[:-1])), dim=-1)
            tok_lp = lp[torch.arange(len(ids), device=lp.device), torch.tensor(ids, device=lp.device)]
            res.append((wording, float(tok_lp.mean()), float(tok_lp.sum()), len(ids)))
            del c, logits, lp
        del base
        return res


def sample(logits, previous, temperature, top_p, repetition_penalty, gen):
    lg = logits.float().clone()
    if repetition_penalty and repetition_penalty != 1.0 and previous:
        idx = torch.tensor(sorted(set(previous)), device=lg.device)
        vals = lg[idx]
        lg[idx] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
    if not temperature or temperature <= 0:
        return int(lg.argmax())
    probs = torch.softmax(lg / temperature, dim=-1)
    sp, si = probs.sort(descending=True)
    keep = sp.cumsum(0) - sp < top_p
    sp = sp * keep
    choice = torch.multinomial((sp / sp.sum()).cpu(), 1, generator=gen)
    return int(si[choice.to(si.device)])
