#!/bin/bash
set -euo pipefail

# Full PACS sweep for DWA_CORAL_EMA (test_env 0,1,2,3 | 3000 steps).
# Uses the custom dwa_train_v2 entry point so original DomainBed files are
# left untouched.

DATA_DIR=./data
DATASET=PACS
ALG=DWA_CORAL_EMA
STEPS=3000
CKPT=500
HPARAMS='{"resnet18": true, "resnet50_augmix": false, "dwa_coral_lambda": 0.3, "dwa_coral_beta": 0.0, "dwa_coral_tau": 0.5, "dwa_ema_alpha": 0.9}'

TEST_ENVS=(0 1 2 3)

echo "Running PACS | ${ALG}"
echo "Test envs: ${TEST_ENVS[@]}"
echo "Steps per run: ${STEPS}"
echo "Checkpoint frequency: ${CKPT}"
echo "HParams: ${HPARAMS}"

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

  python -u -m domainbed.scripts.dwa_train_v2 \
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

echo "All ${ALG} runs finished."
