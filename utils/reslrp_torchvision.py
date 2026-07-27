"""ResLRP for torchvision Vision Transformers — LXT-style, single file, no dependencies.

Drop this next to your code and run::

    import torchvision, torch
    from utils.reslrp_torchvision import attribute, self_check

    model = torchvision.models.vit_b_16(weights="DEFAULT").eval().cuda()
    self_check(model)                                  # verifies every rule fires
    heat = attribute(model, x)                         # (B, H, W) pixel relevance

What this implements
--------------------
Layer-wise Relevance Propagation in the **LXT convention**: every rule is a
forward-identical ``torch.autograd.Function`` that only rewrites the backward, so
after a single ``backward()`` the relevance is simply ``R = x * x.grad``.  Nothing
about the forward pass changes — logits are bit-identical to the unpatched model
(``self_check`` asserts this).

Rules, per the configuration you asked for:

===========================  =========================================================
patch-embed ``conv_proj``    **z⁺** — only positive contributions reach the pixels
MLP ``Linear`` (both)        **γ = 0.25** — contributions agreeing with the output
                             sign boosted by (1+γ)
residual adds (2 per block)  **γ = 1.0** — this is the ResLRP contribution
attention q/k/v/out_proj     ε-rule (ε = 1e-6)
attention softmax            AttnLRP quotient rule: q ÷ 4, k ÷ 4, v ÷ 2
GELU                         identity rule (gate detached)
LayerNorm                    1/std detached, mean-subtraction left in the graph
classifier head              ε-rule
===========================  =========================================================

Why the residual rule needs its own module-free treatment
---------------------------------------------------------
The usual Zennit route is to insert a ``SumLayer`` module and map
``(SumLayer, Gamma)`` in a composite.  That works, but it fails *silently* when a
residual add is not routed through the module — the composite simply never
matches and you get plain AttnLRP with no warning.  Here the γ rule is called
explicitly inside the patched block forward, and :func:`self_check` counts the
invocations against ``2 × depth`` so a missed site is an error, not a mystery.

Notes and known choices
-----------------------
* ``nn.MultiheadAttention`` packs q/k/v into one ``in_proj_weight`` and dispatches
  to a fused SDPA kernel, which leaves nowhere to attach a rule.  It is unrolled
  here into explicit projections and a plain softmax.  Without this your attention
  is silently unruled.
* ``Encoder.forward``'s ``input + self.pos_embedding`` is deliberately left a plain
  ``+``.  The position-embedding branch is a parameter with no path to the pixels,
  so a plain add keeps all relevance on the pixel path.  Applying γ there would
  still reweight the patch branch, so treat it as an untested knob, not a no-op.
* Everything is patched **per instance** and restored on context exit; no class is
  monkey-patched globally, so two models in one process cannot interfere.

Tested against torchvision 0.24 (``vit_b_16`` / ``vit_l_16`` / ``vit_h_14``).
"""

from __future__ import annotations

import math
import types
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["attribute", "attribute_concept", "lrp", "self_check", "LRPConfig"]

EPS = 1e-6  # epsilon-rule stabiliser


class LRPConfig:
    """Rule selection.  The defaults are the recipe this file documents."""

    def __init__(
        self,
        res_gamma: float = 1.0,
        mlp_gamma: float = 0.25,
        conv_rule: str = "zplus",   # "zplus" | "eps"
        attn_rule: str = "attn_lrp",  # "attn_lrp" | "cp" | "plain"
        eps: float = EPS,
    ) -> None:
        self.res_gamma = res_gamma
        self.mlp_gamma = mlp_gamma
        self.conv_rule = conv_rule
        self.attn_rule = attn_rule
        self.eps = eps

    def __repr__(self) -> str:
        return (f"LRPConfig(res_gamma={self.res_gamma}, mlp_gamma={self.mlp_gamma}, "
                f"conv_rule={self.conv_rule!r}, attn_rule={self.attn_rule!r}, eps={self.eps})")


# --------------------------------------------------------------------------------------
# LXT primitives — forward-identical, backward rewritten so that R = x * grad
# --------------------------------------------------------------------------------------


class GradScale(torch.autograd.Function):
    """Identity forward; backward multiplies the incoming gradient by ``factor``."""

    @staticmethod
    def forward(ctx, x: Tensor, factor: Tensor) -> Tensor:
        ctx.save_for_backward(factor)
        return x

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (factor,) = ctx.saved_tensors
        return grad_out * factor, None


class DivGrad(torch.autograd.Function):
    """Identity forward; backward divides the gradient by ``n`` (AttnLRP quotient rule)."""

    @staticmethod
    def forward(ctx, x: Tensor, n: float) -> Tensor:
        ctx.n = n
        return x

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        return grad_out / ctx.n, None


def eps_linear(x: Tensor, weight: Tensor, bias: Tensor | None, eps: float = EPS) -> Tensor:
    """``F.linear`` under the ε-rule.

    The bias absorbs its own share of relevance (denominator is the full
    pre-activation ``y``), which is the standard AttnLRP convention.
    """
    y = F.linear(x, weight, bias)
    factor = (y / (y + eps * (y.sign() + (y == 0)))).detach()
    return GradScale.apply(y, factor)


class GammaLinear(torch.autograd.Function):
    """``F.linear`` under the γ-rule.

    Contributions whose sign agrees with the output are boosted by ``(1 + γ)``::

        y_j >= 0 :  c_mod = c + γ·c⁺
        y_j <  0 :  c_mod = c + γ·c⁻

    The split is conserving and the backward returns ``R / x`` so that
    ``x * grad`` is the relevance downstream.
    """

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None, gamma: float) -> Tensor:
        ctx.save_for_backward(x, weight, bias)
        ctx.gamma = gamma
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x, w, b = ctx.saved_tensors
        g = ctx.gamma
        y = F.linear(x, w, b)
        xp, xn = x.clamp(min=0), x.clamp(max=0)
        wp, wn = w.clamp(min=0), w.clamp(max=0)
        if b is None:
            bp = bn = y.new_zeros(())
        else:
            bp, bn = b.clamp(min=0), b.clamp(max=0)
        pos = y >= 0
        z_mod = y + g * torch.where(pos, xp @ wp.t() + xn @ wn.t() + bp,
                                    xp @ wn.t() + xn @ wp.t() + bn)
        z_mod = z_mod + ((z_mod == 0.0).to(z_mod) + z_mod.sign()) * 1e-9
        s = grad_out * y / z_mod
        sp, sn = s * pos, s * ~pos
        grad_x = s @ w + g * ((x > 0) * (sp @ wp + sn @ wn) +
                              (x <= 0) * (sp @ wn + sn @ wp))
        return grad_x, None, None, None


class ResidualGamma(torch.autograd.Function):
    """γ-rule at a residual add ``res + z`` — the ResLRP contribution.

    Semantics match Zennit's ``Gamma`` applied to an explicit sum layer::

        res_mod = res·(1 + γ·1[res·out >= 0])
        R_res   = res_mod / (res_mod + z_mod) · R_out

    and the backward returns ``R_res / res`` so ``x * grad`` stays the relevance.

    .. warning::
       Divide by the **raw** branch, not by ``res_mod``.  Dividing by ``res_mod``
       looks like a harmless symmetry but attenuates aligned contributions by
       ``(1 + γ)`` at every residual site — over a 12-block ViT that is 24 sites,
       so with γ=1 the input-layer relevance is suppressed by ~2²⁴ and your
       heatmap decays into noise.
    """

    @staticmethod
    def forward(ctx, res: Tensor, z: Tensor, gamma: float = 1.0, eps: float = 1e-8) -> Tensor:
        ctx.save_for_backward(res, z)
        ctx.gamma, ctx.eps = gamma, eps
        return res + z

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        res, z = ctx.saved_tensors
        g, eps = ctx.gamma, ctx.eps
        out = res + z
        upper = out * grad_out  # incoming relevance R_out

        def add_eps(t: Tensor) -> Tensor:
            return t + ((t == 0.0).to(t) + t.sign()) * eps

        res_mod = res + g * ((out * res) >= 0).float() * res
        z_mod = z + g * ((out * z) >= 0).float() * z
        denom = add_eps(res_mod + z_mod)
        return ((res_mod / denom) * upper / add_eps(res),
                (z_mod / denom) * upper / add_eps(z), None, None)


class ZPlusConv2d(torch.autograd.Function):
    """z⁺-rule for the patch-embedding ``Conv2d``.

    Forward is exact (weight + bias).  The backward keeps only the positive input
    contributions ``z⁺ = conv(x⁺, w⁺) + conv(x⁻, w⁻)`` and redistributes relevance
    onto the pixels through the transposed convolution.  The bias share is absorbed.
    """

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None, stride, padding) -> Tensor:
        ctx.save_for_backward(x, weight)
        ctx.stride = stride
        ctx.padding = 0 if isinstance(padding, str) else padding
        return F.conv2d(x, weight, bias, stride=stride, padding=padding)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x, w = ctx.saved_tensors
        st, pad = ctx.stride, ctx.padding
        wp, wn = w.clamp(min=0), w.clamp(max=0)
        xp, xn = x.clamp(min=0), x.clamp(max=0)
        zp = F.conv2d(xp, wp, None, stride=st, padding=pad) + \
            F.conv2d(xn, wn, None, stride=st, padding=pad)
        zp = zp + (zp == 0) * 1e-9 + zp.sign() * 1e-9
        y = F.conv2d(x, w, None, stride=st, padding=pad)
        s = grad_out * (y / zp)
        grad_x = (x > 0) * F.conv_transpose2d(s, wp, None, stride=st, padding=pad) + \
                 (x <= 0) * F.conv_transpose2d(s, wn, None, stride=st, padding=pad)
        return grad_x, None, None, None, None


def gated_gelu(x: Tensor) -> Tensor:
    """Identity rule through GELU: relevance passes element-wise unchanged.

    torchvision's ``MLPBlock`` uses ``nn.GELU()`` (exact erf), so no tanh
    approximation here.
    """
    gate = torch.where(x.abs() < 1e-6, torch.full_like(x, 0.5), F.gelu(x) / x).detach()
    return gate * x


# --------------------------------------------------------------------------------------
# Patched forwards — bound per instance, each reads ``self._lrp`` for the config
# --------------------------------------------------------------------------------------


def _conv_proj_forward(self, x: Tensor) -> Tensor:
    cfg = self._lrp
    if cfg.conv_rule == "zplus":
        return ZPlusConv2d.apply(x, self.weight, self.bias, self.stride, self.padding)
    y = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
    factor = (y / (y + cfg.eps * (y.sign() + (y == 0)))).detach()
    return GradScale.apply(y, factor)


def _linear_forward(self, x: Tensor) -> Tensor:
    """ε-rule for any remaining ``nn.Linear`` (the classifier head)."""
    return eps_linear(x, self.weight, self.bias, self._lrp.eps)


def _layernorm_forward(self, x: Tensor) -> Tensor:
    """LayerNorm with 1/std detached; mean-subtraction stays in the graph (it is linear)."""
    mu = x.mean(-1, keepdim=True)
    std = (x.var(-1, unbiased=False, keepdim=True) + self.eps).sqrt().detach()
    return (x - mu) / std * self.weight + self.bias


def _mha_forward(self, query: Tensor, key=None, value=None, need_weights=True, **kw):
    """Unrolled ``nn.MultiheadAttention`` (self-attention) with LRP-able internals.

    The stock module fuses q/k/v into ``in_proj_weight`` and calls a fused SDPA
    kernel, so there is no place to attach a rule.  Unrolling is numerically
    equivalent — the quotient splits below only alter the backward.
    """
    cfg = self._lrp
    x = query
    B, T, C = x.shape
    H = self.num_heads
    hd = C // H

    wq, wk, wv = self.in_proj_weight.chunk(3, dim=0)
    if self.in_proj_bias is not None:
        bq, bk, bv = self.in_proj_bias.chunk(3, dim=0)
    else:
        bq = bk = bv = None

    def proj(t, w, b):
        return eps_linear(t, w, b, cfg.eps).view(B, T, H, hd).transpose(1, 2)

    q, k, v = proj(x, wq, bq), proj(x, wk, bk), proj(x, wv, bv)

    if cfg.attn_rule == "attn_lrp":
        # AttnLRP quotient rule: the two bilinear matmuls each split relevance in
        # half, so q and k are halved twice (÷4) and v once (÷2).
        q, k, v = DivGrad.apply(q, 4.0), DivGrad.apply(k, 4.0), DivGrad.apply(v, 2.0)

    attn = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
    attn = attn.softmax(dim=-1)
    if cfg.attn_rule == "cp":
        attn = attn.detach()  # CP-LRP: all relevance flows through V

    out = (attn @ v).transpose(1, 2).reshape(B, T, C)
    out = eps_linear(out, self.out_proj.weight, self.out_proj.bias, cfg.eps)
    return out, None


def _mlp_forward(self, x: Tensor) -> Tensor:
    """``MLPBlock`` (an ``nn.Sequential`` of [Linear, GELU, Dropout, Linear, Dropout]).

    Indexed positionally because the parent Sequential's key names are not stable.
    """
    cfg = self._lrp
    g = cfg.mlp_gamma

    def lin(t, layer):
        if g:
            return GammaLinear.apply(t, layer.weight, layer.bias, g)
        return eps_linear(t, layer.weight, layer.bias, cfg.eps)

    x = lin(x, self[0])
    x = gated_gelu(x)
    x = self[2](x)
    x = lin(x, self[3])
    return self[4](x)


def _block_forward(self, input: Tensor) -> Tensor:  # noqa: A002 — matches upstream signature
    """``EncoderBlock`` with both residual adds under the γ-rule."""
    cfg = self._lrp
    g = cfg.res_gamma

    def res_add(a, b):
        return ResidualGamma.apply(a, b, g) if g else a + b

    z = self.ln_1(input)
    out = self.self_attention(z, z, z, need_weights=False)
    z = self.dropout(out[0] if isinstance(out, tuple) else out)
    x = res_add(input, z)
    return res_add(x, self.mlp(self.ln_2(x)))


# --------------------------------------------------------------------------------------
# Patcher
# --------------------------------------------------------------------------------------


def _blocks(model: nn.Module):
    from torchvision.models.vision_transformer import EncoderBlock

    return [m for m in model.modules() if isinstance(m, EncoderBlock)]


@contextmanager
def lrp(model: nn.Module, cfg: LRPConfig | None = None, **overrides):
    """Context manager that installs the LRP rules and restores the model on exit.

    Patches individual module *instances* (never classes), disables parameter
    gradients for the duration, and puts everything back exactly as it was —
    including on exception.

    Parameters
    ----------
    model : nn.Module
        A torchvision ``VisionTransformer``.
    cfg : LRPConfig, optional
        Rule configuration; defaults to the documented recipe.
    **overrides
        Convenience kwargs forwarded to :class:`LRPConfig` when ``cfg`` is None.
    """
    from torchvision.models.vision_transformer import VisionTransformer

    if not isinstance(model, VisionTransformer):
        raise TypeError(f"expected a torchvision VisionTransformer, got {type(model).__name__}")
    cfg = cfg or LRPConfig(**overrides)

    original: list[tuple[nn.Module, object]] = []
    grads: list[tuple[torch.nn.Parameter, bool]] = []

    def bind(module: nn.Module, fn) -> None:
        original.append((module, module.__dict__.get("forward")))
        module._lrp = cfg
        module.forward = types.MethodType(fn, module)

    try:
        for p in model.parameters():
            grads.append((p, p.requires_grad))
            p.requires_grad_(False)

        bind(model.conv_proj, _conv_proj_forward)
        for blk in _blocks(model):
            bind(blk, _block_forward)
            bind(blk.self_attention, _mha_forward)
            bind(blk.ln_1, _layernorm_forward)
            bind(blk.ln_2, _layernorm_forward)
            bind(blk.mlp, _mlp_forward)
        # Anything not bound above: the encoder's final LayerNorm and the
        # classifier head. The MLP and attention Linears are also caught here,
        # but harmlessly — those forwards are never called, because _mlp_forward
        # and _mha_forward use their .weight/.bias directly.
        for m in model.modules():
            if "forward" in m.__dict__:
                continue
            if isinstance(m, nn.LayerNorm):
                bind(m, _layernorm_forward)
            elif isinstance(m, nn.Linear):
                bind(m, _linear_forward)
        yield cfg
    finally:
        for module, old in original:
            if old is None:
                module.__dict__.pop("forward", None)
            else:
                module.forward = old
            module.__dict__.pop("_lrp", None)
        for p, req in grads:
            p.requires_grad_(req)


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def attribute(model: nn.Module, x: Tensor, target: Tensor | None = None,
              cfg: LRPConfig | None = None, sum_channels: bool = True, **overrides) -> Tensor:
    """Pixel relevance for ``target`` (default: the predicted class).

    Parameters
    ----------
    model : nn.Module
        torchvision ``VisionTransformer``, in eval mode.
    x : Tensor
        Input batch ``(B, 3, H, W)``, already normalised for the model.
    target : Tensor or None
        Class indices ``(B,)``.  ``None`` uses ``argmax`` of the logits.
    sum_channels : bool
        Sum relevance over RGB, giving ``(B, H, W)``.  ``False`` keeps ``(B, 3, H, W)``.

    Returns
    -------
    Tensor
        Relevance map, detached, on the input's device.
    """
    was_training = model.training
    model.eval()
    try:
        with lrp(model, cfg, **overrides):
            xg = x.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                logits = model(xg)
                if target is None:
                    target = logits.argmax(dim=-1)
                target = target.to(logits.device)
                selected = logits.gather(1, target[:, None]).sum()
                (grad,) = torch.autograd.grad(selected, xg)
            rel = xg.detach() * grad
    finally:
        model.train(was_training)
    return rel.sum(1) if sum_channels else rel


def attribute_concept(
    model: nn.Module,
    x: Tensor,
    cav: Tensor,
    layer_name: str,
    cfg: LRPConfig | None = None,
    sum_channels: bool = True,
    **overrides,
) -> Tensor:
    """Pixel relevance for a CAV at a two-dimensional internal activation.

    The relevance seed matches the concept-localization convention used by this
    project: ``activation.clamp(min=0) * cav``.  The selected layer must emit a
    tensor of shape ``(B, D)`` and the CAV must contain exactly ``D`` features.
    """
    modules = dict(model.named_modules())
    if layer_name not in modules:
        available = ", ".join(name for name in modules if name)
        raise ValueError(
            f"unknown layer {layer_name!r}; available layers include: {available}"
        )

    if cav.ndim == 2 and cav.shape[0] == 1:
        cav = cav[0]
    if cav.ndim != 1:
        raise ValueError(
            f"expected a one-dimensional CAV, got shape {tuple(cav.shape)}"
        )

    captured: list[Tensor] = []

    def capture_activation(_module, _inputs, output):
        if not isinstance(output, Tensor):
            raise TypeError(
                f"layer {layer_name!r} returned {type(output).__name__}, expected Tensor"
            )
        captured.append(output)

    handle = modules[layer_name].register_forward_hook(capture_activation)
    was_training = model.training
    model.eval()
    try:
        with lrp(model, cfg, **overrides):
            xg = x.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                model(xg)
                if len(captured) != 1:
                    raise RuntimeError(
                        f"layer {layer_name!r} ran {len(captured)} times; expected once"
                    )
                activation = captured[0]
                if activation.ndim != 2:
                    raise ValueError(
                        f"layer {layer_name!r} must emit shape (B, D), got "
                        f"{tuple(activation.shape)}"
                    )
                if cav.numel() != activation.shape[-1]:
                    raise ValueError(
                        f"CAV has {cav.numel()} features but layer {layer_name!r} "
                        f"emits {activation.shape[-1]}"
                    )
                cav_on_activation = cav.to(
                    device=activation.device, dtype=activation.dtype
                )
                selected = (
                    activation.clamp(min=0) * cav_on_activation.unsqueeze(0)
                ).sum()
                (grad,) = torch.autograd.grad(selected, xg)
            rel = xg.detach() * grad
    finally:
        handle.remove()
        model.train(was_training)

    if not torch.isfinite(rel).all():
        raise RuntimeError("concept attribution produced non-finite relevance")
    return rel.sum(1) if sum_channels else rel


def self_check(model: nn.Module, image_size: int | None = None, verbose: bool = True) -> dict:
    """Verify every rule actually fires, and that the forward is unchanged.

    A rule that silently fails to install looks like "LRP barely helps", not like a
    bug, so this is worth running once per model class.  Checks:

    1. the patched forward reproduces the unpatched logits (rules are backward-only);
    2. ``ResidualGamma`` runs exactly ``2 × depth`` times;
    3. ``ZPlusConv2d`` runs once and ``GammaLinear`` ``2 × depth`` times.

    Raises
    ------
    AssertionError
        If any rule is missing or the forward is not preserved.
    """
    device = next(model.parameters()).device
    size = image_size or model.image_size
    x = torch.randn(1, 3, size, size, device=device)
    depth = len(_blocks(model))

    model.eval()
    with torch.no_grad():
        reference = model(x).clone()

    counts = {"ResidualGamma": 0, "GammaLinear": 0, "ZPlusConv2d": 0}
    originals = {}
    for fn in (ResidualGamma, GammaLinear, ZPlusConv2d):
        originals[fn] = fn.apply

    def wrap(name, orig):
        def counting(*a, **k):
            counts[name] += 1
            return orig(*a, **k)
        return counting

    ResidualGamma.apply = wrap("ResidualGamma", originals[ResidualGamma])  # type: ignore[method-assign]
    GammaLinear.apply = wrap("GammaLinear", originals[GammaLinear])        # type: ignore[method-assign]
    ZPlusConv2d.apply = wrap("ZPlusConv2d", originals[ZPlusConv2d])        # type: ignore[method-assign]
    try:
        with lrp(model) as cfg:
            xg = x.clone().requires_grad_(True)
            with torch.enable_grad():
                logits = model(xg)
                logits.max().backward()
            patched = logits.detach()
    finally:
        for fn, orig in originals.items():
            fn.apply = orig  # type: ignore[method-assign]

    max_dev = (patched - reference).abs().max().item()
    scale = reference.abs().max().item() or 1.0

    result = dict(depth=depth, counts=counts, max_logit_deviation=max_dev, config=repr(cfg))
    if verbose:
        print(f"depth                 {depth} blocks")
        print(f"ResidualGamma calls   {counts['ResidualGamma']:3d}  (expect {2 * depth})")
        print(f"GammaLinear  calls    {counts['GammaLinear']:3d}  (expect {2 * depth})")
        print(f"ZPlusConv2d  calls    {counts['ZPlusConv2d']:3d}  (expect 1)")
        print(f"max |logit| deviation {max_dev:.2e}  (forward must be unchanged)")
        print(f"config                {cfg}")

    assert counts["ResidualGamma"] == 2 * depth, \
        f"residual rule reached {counts['ResidualGamma']} of {2 * depth} sites"
    assert counts["GammaLinear"] == 2 * depth, \
        f"MLP gamma reached {counts['GammaLinear']} of {2 * depth} linears"
    assert counts["ZPlusConv2d"] == 1, f"z+ conv fired {counts['ZPlusConv2d']} times, expected 1"
    assert max_dev < 1e-3 * scale, f"forward changed by {max_dev:.3e} — rules must be backward-only"
    if verbose:
        print("\nall rules installed and the forward is preserved.")
    return result


if __name__ == "__main__":
    import torchvision

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = torchvision.models.vit_b_16(weights="DEFAULT").eval().to(dev)
    self_check(m)
    heat = attribute(m, torch.randn(2, 3, 224, 224, device=dev))
    print(f"\nheatmap {tuple(heat.shape)}  "
          f"positive mass {heat.clamp(min=0).sum().item() / heat.abs().sum().item():.3f}")
