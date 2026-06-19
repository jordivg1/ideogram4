#!/usr/bin/env bash
# Prueba rápida de Ideogram 4 en Mac. Uso:
#   ./try.sh "tu prompt aqui"
# Variables (opcionales):
#   SIZE=512        resolución (múltiplo de 16)
#   PRESET=...      V4_TURBO_12 | V4_DEFAULT_20 | V4_QUALITY_48
#   DEVICE=cpu      cpu funciona con fp8; mps NO soporta float8_e4m3fn
#   MAGIC=1         activa magic prompt (necesita IDEOGRAM_API_KEY en .env)
set -euo pipefail
cd "$(dirname "$0")"

# Carga credenciales desde .env si existe (HF_TOKEN, IDEOGRAM_API_KEY)
if [ -f .env ]; then set -a; . ./.env; set +a; fi

PROMPT="${1:-a ginger cat wearing a tiny wizard hat reading a spellbook}"
SIZE="${SIZE:-512}"
PRESET="${PRESET:-V4_TURBO_12}"
DEVICE="${DEVICE:-cpu}"
MAGIC="${MAGIC:-0}"

# Con MAGIC=1 expandimos el prompt a JSON (API hosted de Ideogram); si no, prompt plano.
if [ "$MAGIC" = "1" ]; then
  MAGIC_ARGS=(--magic-prompt)
else
  MAGIC_ARGS=(--no-magic-prompt --warn-on-caption-issues)
fi

./.venv/bin/python run_inference.py \
  --prompt "$PROMPT" \
  --output out.png \
  --quantization fp8 \
  --device "$DEVICE" \
  --height "$SIZE" --width "$SIZE" \
  --sampler-preset "$PRESET" \
  "${MAGIC_ARGS[@]}"

echo "Hecho -> $(pwd)/out.png"
