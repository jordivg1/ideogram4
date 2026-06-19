#!/usr/bin/env python
"""Ideogram 4 en Apple Silicon (Metal/MPS), optimizado para 24 GB.

Por qué este script y no run_inference.py:
  * PyTorch MPS no soporta el dtype float8_e4m3fn -> aquí los pesos fp8 se
    guardan como uint8 y se descuantizan con una LUT (e4m3->bf16) en el forward.
  * El modelo completo son ~26 GB de pesos (2 DiTs + encoder) y no cabe en 24 GB.
    Aquí se carga por fases liberando memoria: encoder -> (libera) -> DiT.
  * Por defecto NO usa CFG (un solo DiT) para caber con holgura. CFG=1 carga
    también el unconditional (va MUY justo de RAM en 24 GB).

Uso:
  ./.venv/bin/python run_metal.py --prompt "..." [--size 512] [--preset V4_TURBO_12]
"""
from __future__ import annotations

import argparse
import gc
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Shim: hacer que la ruta fp8 (weight-only e4m3) funcione en MPS via LUT uint8
# --------------------------------------------------------------------------
from ideogram4 import quantized_loading as ql

FP8 = ql.FP8_WEIGHT_DTYPE  # torch.float8_e4m3fn
_LUT: dict = {}


def _lut(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
  """Tabla de 256 entradas: byte e4m3 -> valor en `dtype`. Construida en CPU."""
  key = (str(device), dtype)
  if key not in _LUT:
    table = torch.arange(256, dtype=torch.uint8).view(FP8).to(torch.float32)
    _LUT[key] = table.to(device=device, dtype=dtype)
  return _LUT[key]


def _fp8_init(self, in_features, out_features, bias, compute_dtype):
  nn.Module.__init__(self)
  self.in_features = in_features
  self.out_features = out_features
  self.compute_dtype = compute_dtype
  # peso guardado como uint8 (los bytes crudos del e4m3): MPS sí lo soporta
  self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.uint8))
  self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32))
  if bias:
    self.register_buffer("bias", torch.empty(out_features, dtype=compute_dtype))
  else:
    self.bias = None


def _fp8_forward(self, x):
  lut = _lut(self.weight.device, x.dtype)
  # gather: uint8 -> bf16 (descuantiza una capa cada vez, transitorio)
  w = lut[self.weight.long()] * self.weight_scale.to(x.dtype).unsqueeze(1)
  bias = self.bias.to(x.dtype) if self.bias is not None else None
  return F.linear(x, w, bias)


ql.Fp8Linear.__init__ = _fp8_init
ql.Fp8Linear.forward = _fp8_forward


def _patched_load(model, state_dict, device, dtype, *, assign=False, strict=True):
  """Como load_fp8_state_dict original, pero convierte fp8 -> uint8 (para MPS)."""
  import warnings

  prepared = {}
  for k, v in state_dict.items():
    if v.dtype == FP8:
      prepared[k] = v.view(torch.uint8).to(device=device)  # bytes crudos a MPS
    elif k.endswith(ql.FP8_SCALE_SUFFIX):
      prepared[k] = v.to(device=device, dtype=torch.float32)
    elif v.is_floating_point():
      prepared[k] = v.to(device=device, dtype=dtype)
    else:
      prepared[k] = v.to(device=device)
  missing, unexpected = model.load_state_dict(prepared, strict=False, assign=assign)
  if unexpected:
    raise RuntimeError(f"unexpected keys after fp8 load: {unexpected[:10]}")
  if missing:
    if strict:
      raise RuntimeError(f"missing keys after fp8 load: {missing[:10]}")
    warnings.warn(f"missing keys after fp8 load: {missing[:10]}", stacklevel=2)
  model.to(device)


ql.load_fp8_state_dict = _patched_load
# pipeline importó el nombre por valor -> parchear también ahí
import ideogram4.pipeline_ideogram4 as P  # noqa: E402

P.load_fp8_state_dict = _patched_load

# --------------------------------------------------------------------------
from ideogram4 import PRESETS  # noqa: E402
from ideogram4.caption_verifier import CaptionVerifier  # noqa: E402
from ideogram4.latent_norm import get_latent_norm  # noqa: E402
from ideogram4.modeling_ideogram4 import Ideogram4Config  # noqa: E402
from ideogram4.pipeline_ideogram4 import (  # noqa: E402
  Ideogram4Pipeline,
  Ideogram4PipelineConfig,
  _build_transformer,
  _load_autoencoder,
  _load_indexed_or_single_state_dict,
  _load_qwen3_vl,
)
from ideogram4.scheduler import (  # noqa: E402
  get_schedule_for_resolution,
  make_step_intervals,
)
from huggingface_hub import hf_hub_download  # noqa: E402


def _free():
  gc.collect()
  if torch.backends.mps.is_available():
    torch.mps.empty_cache()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--prompt", required=True)
  ap.add_argument("--output", default="out.png")
  ap.add_argument("--size", type=int, default=512)
  ap.add_argument("--preset", choices=sorted(PRESETS), default="V4_TURBO_12")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--cfg", action="store_true", help="usa los 2 DiTs (MUY justo en 24GB)")
  args = ap.parse_args()

  device = torch.device("mps")
  dtype = torch.bfloat16
  h = w = args.size
  cfg = Ideogram4PipelineConfig(weights_repo="ideogram-ai/ideogram-4-fp8")
  tcfg = Ideogram4Config()
  preset = PRESETS[args.preset]
  t0 = time.time()

  # ---- Fase A: text encoder (una sola pasada), luego se libera ----
  print("[fase A] cargando text encoder...", flush=True)
  tok, enc = _load_qwen3_vl(
    cfg.weights_repo,
    device,
    dtype,
    tokenizer_subfolder=cfg.tokenizer_subfolder,
    text_encoder_subfolder=cfg.text_encoder_subfolder,
  )
  pipe = Ideogram4Pipeline.__new__(Ideogram4Pipeline)
  pipe.config = cfg
  pipe.device = device
  pipe.dtype = dtype
  pipe.text_encoder = enc
  pipe.text_tokenizer = tok
  pipe.caption_verifier = CaptionVerifier()
  shift, scale = get_latent_norm()
  pipe.latent_shift = shift.to(device)
  pipe.latent_scale = scale.to(device)

  inputs = pipe._build_inputs([args.prompt], height=h, width=w)
  print("[fase A] codificando prompt...", flush=True)
  llm_features = pipe._encode_text(
    inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"]
  ).detach()

  pipe.text_encoder = None
  del enc
  _free()
  print(f"[fase A] hecho ({time.time() - t0:.0f}s). Encoder liberado.", flush=True)

  # ---- Fase B: DiT(s) y bucle de denoising ----
  num_steps = preset.num_steps
  schedule = get_schedule_for_resolution((h, w), known_mean=preset.mu, std=preset.std)
  # OJO: el schedule usa float64/ndtri (no soportados en MPS) -> dejar en CPU.
  # t_val/s_val son escalares Python, así que no hace falta que estén en device.
  step_intervals = make_step_intervals(num_steps)
  gw = torch.as_tensor(preset.guidance_schedule, dtype=torch.float32, device=device)

  num_image_tokens = inputs["num_image_tokens"]
  grid_h, grid_w = inputs["grid_h"], inputs["grid_w"]
  max_text = inputs["max_text_tokens"]
  latent_dim = tcfg.in_channels

  print("[fase B] cargando DiT condicional...", flush=True)
  cond_sd = _load_indexed_or_single_state_dict(cfg.weights_repo, cfg.conditional_index_filename)
  cond = _build_transformer(tcfg, cond_sd, device, dtype)
  del cond_sd
  _free()

  uncond = None
  neg_pos = neg_seg = neg_ind = neg_feat = None
  if args.cfg:
    print("[fase B] cargando DiT incondicional (CFG, memoria al límite)...", flush=True)
    uncond_sd = _load_indexed_or_single_state_dict(
      cfg.weights_repo, cfg.unconditional_index_filename
    )
    uncond = _build_transformer(tcfg, uncond_sd, device, dtype)
    del uncond_sd
    _free()
    neg_pos = inputs["position_ids"][:, max_text:]
    neg_seg = inputs["segment_ids"][:, max_text:]
    neg_ind = inputs["indicator"][:, max_text:]
    neg_feat = torch.zeros(
      1, num_image_tokens, llm_features.shape[-1], dtype=llm_features.dtype, device=device
    )

  g = torch.Generator()  # en CPU para evitar rarezas del generador MPS
  g.manual_seed(args.seed)
  z = torch.randn(1, num_image_tokens, latent_dim, dtype=torch.float32, generator=g).to(device)
  text_z_padding = torch.zeros(1, max_text, latent_dim, dtype=torch.float32, device=device)

  print(f"[fase B] denoising: {num_steps} pasos (CFG={'on' if args.cfg else 'off'})", flush=True)
  with torch.no_grad():
    for i in range(num_steps - 1, -1, -1):
      ts = time.time()
      t_val = float(schedule(step_intervals[i + 1].unsqueeze(0)).item())
      s_val = float(schedule(step_intervals[i].unsqueeze(0)).item())
      t = torch.full((1,), t_val, dtype=torch.float32, device=device)

      pos_z = torch.cat([text_z_padding, z], dim=1)
      pos_out = cond(
        llm_features=llm_features,
        x=pos_z,
        t=t,
        position_ids=inputs["position_ids"],
        segment_ids=inputs["segment_ids"],
        indicator=inputs["indicator"],
      )
      pos_v = pos_out[:, max_text:]

      if args.cfg:
        neg_v = uncond(
          llm_features=neg_feat,
          x=z,
          t=t,
          position_ids=neg_pos,
          segment_ids=neg_seg,
          indicator=neg_ind,
        )
        gw_i = gw[i]
        v = gw_i * pos_v + (1.0 - gw_i) * neg_v
      else:
        v = pos_v  # sin CFG: velocidad condicional cruda

      z = z + v * (s_val - t_val)
      print(f"  paso {num_steps - i}/{num_steps}  ({time.time() - ts:.1f}s)", flush=True)

  del cond, uncond
  _free()

  # ---- Fase C: decode con el VAE ----
  print("[fase C] decodificando con VAE...", flush=True)
  ae_path = hf_hub_download(repo_id=cfg.weights_repo, filename=cfg.autoencoder_filename)
  pipe.autoencoder = _load_autoencoder(ae_path, device, dtype)
  imgs = pipe._decode(z, grid_h=grid_h, grid_w=grid_w)
  imgs[0].save(args.output)
  print(f"OK -> {args.output}  (total {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
  main()
