#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_PATH="${ROOT_DIR}/data/raw/dsprites/dsprite_train.npz"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Missing dSprites archive at ${DATASET_PATH}" >&2
  echo "Run: python scripts/setup/download_dsprites.py" >&2
  exit 1
fi

cd "${ROOT_DIR}/scripts"

"${PYTHON_BIN}" -m experiments.disent with \
  dataset.dsprites \
  dataset.condition=extrp \
  dataset.variant=blank_side \
  dataset.modifiers='["sparse_posX"]' \
  model.kim \
  training.beta \
  training.epochs=1 \
  training.batch_size=2048 \
  training.num_workers=0 \
  epoch_length=4 \
  validation_epoch_length=2 \
  no_cuda=True \
  save_folder='../data/sims/smoke/disent'

"${PYTHON_BIN}" -m experiments.composition with \
  dataset.dsprites \
  dataset.condition=extrp \
  dataset.variant=blank_side \
  dataset.modifiers='["sparse_posX"]' \
  model.abdi \
  model.composition_op=linear \
  training.epochs=1 \
  training.batch_size=512 \
  training.num_workers=0 \
  training.lr=0.0001 \
  epoch_length=4 \
  validation_epoch_length=2 \
  no_cuda=True \
  save_folder='../data/sims/smoke/composition'
