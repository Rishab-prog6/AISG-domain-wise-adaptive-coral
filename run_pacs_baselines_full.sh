#!/bin/bash
set -euo pipefail

DATA_DIR=./data
DATASET=PACS
STEPS=3000
CKPT=500
HPARAMS='{"resnet18": true, "resnet50_augmix": false}'

ALGORITHMS=("ERM" "CORAL" "GroupDRO" "IRM" "Mixup")
TEST_ENVS=(0 1 2 3)

echo "Running PACS baselines"
echo "Algorithms: ${ALGORITHMS[@]}"
echo "Test envs: ${TEST_ENVS[@]}"
echo "Steps per run: ${STEPS}"
echo "Checkpoint frequency: ${CKPT}"
echo "HParams: ${HPARAMS}"

for ALG in "${ALGORITHMS[@]}"; do
  for ENV in "${TEST_ENVS[@]}"; do
    OUT=./outputs/${DATASET}_${ALG}_env${ENV}_steps${STEPS}

    echo "============================================================"
    echo "Running ${DATASET} | ${ALG} | test_env=${ENV}"
    echo "Output: ${OUT}"
    echo "============================================================"

    if [ -f "${OUT}/done" ]; then
      echo "Found ${OUT}/done, skipping."
      continue
    fi

    rm -rf "${OUT}"

    python -u -m domainbed.scripts.train \
      --data_dir "${DATA_DIR}" \
      --algorithm "${ALG}" \
      --dataset "${DATASET}" \
      --test_env "${ENV}" \
      --steps "${STEPS}" \
      --checkpoint_freq "${CKPT}" \
      --skip_model_save \
      --hparams "${HPARAMS}" \
      --output_dir "${OUT}"

    echo "Finished ${DATASET} | ${ALG} | test_env=${ENV}"
  done
done

echo "All PACS baseline runs finished."
