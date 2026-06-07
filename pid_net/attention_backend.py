# Attention backend dispatch for the PiD net.
#
# PiD's two attention sites (RotaryAttention, MMDiTJointAttention) call
# F.scaled_dot_product_attention with q/k/v in [B, H, S, D] layout. At inference
# the attn_mask is always None (the decode loop never passes a text-pad mask), so
# flash-attn and sage-attn are *exact drop-ins* — unlike EasyControl, which needs
# an LSE decomposition because it carries a float `b_cond` bias that the flash
# SDPA backend refuses.
#
# Backend selection mirrors ComfyUI's own convention: the launch flags
# --use-sage-attention / --use-flash-attention (surfaced via model_management),
# sage taking precedence — same priority comfy's optimized_attention uses. A node
# widget may override per-decode via set_attention_backend().
#
# Note on flash vs sdpa: PyTorch's SDPA already auto-dispatches to its *built-in*
# flash-attention-v2 kernel here (bf16, no mask, head_dim 64/72), so the explicit
# flash path is numerically identical and only marginally faster (skips a little
# dispatch overhead). Sage (INT8-quantized) is the path with a real speedup.

from typing import Optional

import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func as _flash_attn_func
except ImportError:
    _flash_attn_func = None

try:
    from sageattention import sageattn as _sageattn
except ImportError:
    _sageattn = None

# Head dims sage handles natively. PiD's pixel attention is 72 (1152/16) — not in
# the set — so it transparently falls back to SDPA; patch + joint attention (64)
# are the long-sequence, compute-heavy ones sage actually accelerates.
_SAGE_HEAD_DIMS = (64, 96, 128)

# Per-decode override set by the node: "sdpa" / "flash" / "sage", or None = honor
# the ComfyUI launch flags.
_OVERRIDE: Optional[str] = None
_LAUNCH_CACHE: Optional[str] = None


def set_attention_backend(mode: Optional[str]) -> None:
    """Force a backend for subsequent decodes. mode in {"sdpa","flash","sage"};
    "auto"/None defers to ComfyUI's --use-flash-attention / --use-sage-attention
    launch flags."""
    global _OVERRIDE
    _OVERRIDE = mode if mode in ("sdpa", "flash", "sage") else None


def _launch_backend() -> str:
    """Backend implied by ComfyUI's launch flags (cached — flags can't change
    mid-process). Sage wins over flash, matching comfy's optimized_attention."""
    global _LAUNCH_CACHE
    if _LAUNCH_CACHE is None:
        be = "sdpa"
        try:
            from comfy import model_management as mm

            if _sageattn is not None and mm.sage_attention_enabled():
                be = "sage"
            elif _flash_attn_func is not None and mm.flash_attention_enabled():
                be = "flash"
        except Exception:  # noqa: BLE001 — outside comfy (tests) → plain SDPA
            be = "sdpa"
        if be != "sdpa":
            print(f"[AnimaPiD] attention backend: {be} (from launch flag)")
        _LAUNCH_CACHE = be
    return _LAUNCH_CACHE


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask=None) -> torch.Tensor:
    """Drop-in for F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p=0).

    q/k/v: [B, H, S, D] (SDPA layout). Returns [B, H, S, D]. Both default
    sm_scale = 1/sqrt(D) and bidirectional (no causal) match SDPA, so flash/sage
    are numerically faithful (sage to INT8 precision)."""
    backend = _OVERRIDE or _launch_backend()

    # flash/sage only cover the unmasked, half-precision, CUDA case — exactly what
    # PiD inference hits. Anything else (a mask, fp32, CPU) falls back to SDPA so
    # the helper stays correct everywhere (e.g. masked training, CPU smoke tests).
    fast_ok = attn_mask is None and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)

    if backend == "sage" and fast_ok and _sageattn is not None and q.shape[-1] in _SAGE_HEAD_DIMS:
        # sage "HND" layout == [B, H, S, D]: true drop-in, no transpose.
        return _sageattn(q, k, v, tensor_layout="HND", is_causal=False)

    if backend == "flash" and fast_ok and _flash_attn_func is not None:
        # flash_attn_func wants [B, S, H, D]; transpose in and back.
        out = _flash_attn_func(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=0.0, causal=False,
        )
        return out.transpose(1, 2)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
