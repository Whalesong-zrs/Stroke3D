#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${MV_DIR:?Set MV_DIR to the directory containing per-object SKDream outputs}"
: "${COARSE_DIR:?Set COARSE_DIR for reconstructed meshes}"
: "${REFINE_DIR:?Set REFINE_DIR for refined textured meshes}"

DATA_DIR="${DATA_DIR:-${REPO_DIR}/objsk_eval2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
NUM_VIEWS="${NUM_VIEWS:-4}"
REPEAT_NUM="${REPEAT_NUM:-1}"
DISTANCE="${DISTANCE:-4.0}"
RUN_RECONSTRUCTION="${RUN_RECONSTRUCTION:-1}"
RUN_TILING="${RUN_TILING:-1}"
RUN_REFINEMENT="${RUN_REFINEMENT:-1}"
INSTANTMESH_CONFIG="${INSTANTMESH_CONFIG:-${REPO_DIR}/config/instant-mesh-large.yaml}"
REFINE_CONFIG="${REFINE_CONFIG:-${REPO_DIR}/config/refine.json}"

cd "${REPO_DIR}"

if [[ "${RUN_RECONSTRUCTION}" == "1" ]]; then
  "${PYTHON_BIN}" "${REPO_DIR}/infer_rec.py" \
    "${INSTANTMESH_CONFIG}" "${MV_DIR}" \
    --output_path "${COARSE_DIR}" \
    --num_view "${NUM_VIEWS}" \
    --repeat_num "${REPEAT_NUM}" \
    --distance "${DISTANCE}" \
    --export_texmap \
    --gpu "${GPU}"
fi

if [[ "${RUN_TILING}" == "1" ]]; then
  "${PYTHON_BIN}" "${REPO_DIR}/infer_tile.py" \
    --data-dir "${DATA_DIR}" \
    --image-dir "${MV_DIR}" \
    --num-views "${NUM_VIEWS}" \
    --repeat-num "${REPEAT_NUM}" \
    --gpu "${GPU}"
fi

if [[ "${RUN_REFINEMENT}" == "1" ]]; then
  "${PYTHON_BIN}" "${REPO_DIR}/infer_refine.py" \
    --config "${REFINE_CONFIG}" \
    --mesh-dir "${COARSE_DIR}" \
    --tile-dir "${MV_DIR}" \
    --save-dir "${REFINE_DIR}" \
    --repeat-num "${REPEAT_NUM}"
fi
