# AISG-domain-wise-adaptive-coral

# AISG OOD Domain Generalization Project

This repository contains our final project for AISG, focusing on **Out-of-Distribution (OOD) Generalization and Stable Learning** in image classification.

We study domain generalization under visual distribution shifts, where a model is trained on several source domains and evaluated on an unseen target domain. Our experiments are based on the DomainBed framework and mainly use the PACS and VLCS datasets.

## Project Topic

**Out-of-Distribution Generalization & Stable Learning**

In real-world image classification, the test distribution may differ from the training distribution. For example, a model trained on real photos may need to classify sketches, cartoons, or artistic paintings. Although the semantic labels remain the same, changes in texture, style, color, and background can significantly reduce model performance.

This project studies how different domain generalization algorithms perform under such distribution shifts.

## Datasets

We plan to use two standard domain generalization benchmarks:

### PACS

PACS contains four visual domains:

- Art Painting
- Cartoon
- Photo
- Sketch

The task is multi-class image classification with the same label space across domains.

### VLCS

VLCS contains four image domains:

- VOC2007
- LabelMe
- Caltech101
- SUN09

VLCS is used as a second benchmark to evaluate whether the observed trends generalize beyond PACS.

## Baselines

We reproduce the following baselines using DomainBed:

| Method | Description |
|---|---|
| ERM | Empirical Risk Minimization. The standard supervised learning baseline. |
| IRM | Invariant Risk Minimization. Encourages invariant predictors across environments. |
| GroupDRO | Group Distributionally Robust Optimization. Focuses on high-risk domains. |
| CORAL | Correlation Alignment. Aligns feature covariance statistics across domains. |
| Mixup | Data augmentation method based on interpolating samples and labels. |

## Proposed Method

We plan to implement a CORAL-based extension:

### DWA-CORAL: Domain-wise Adaptive CORAL

Standard CORAL aligns feature covariance statistics across source domains, but it treats source domains relatively uniformly. In contrast, DWA-CORAL assigns larger alignment weights to source domains with higher classification loss, so that the model pays more attention to difficult or high-risk domains.

The expected objective is:

```text
loss = weighted classification loss
     + lambda * weighted CORAL alignment loss
     + beta * domain risk variance penalty
```

----

## Installation

We recommend using Python 3.10.

```bash
conda create -n domainbed python=3.10 -y
conda activate domainbed
```

- Install PyTorch separately according to your GPU and CUDA version.
- For our RTX 5090 environment, we used the CUDA 12.8 PyTorch wheels:
``
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
``

- Then install the remaining dependencies:
``
pip install -r requirements.txt
``

Check the installation:

``
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
``

Expected output should show CUDA is available and the correct GPU name, for example:

``
2.11.0+cu128
12.8
True
NVIDIA GeForce RTX 5090
Note on DomainBed DataLoader
``

- The original DomainBed dataloader caused hanging issues in our RTX 5090 + PyTorch 2.11 environment.
We replaced domainbed/lib/fast_data_loader.py with a safer implementation based on standard PyTorch DataLoader.

- This modification is used consistently for all baselines and proposed methods in our experiments. Therefore, the comparisons between methods remain fair under our experimental setup, although the numbers should be interpreted as results from a modified DomainBed environment rather than an exact reproduction of the original DomainBed benchmark.

## Usage

All commands below should be run from the repository root directory.

```bash
cd DomainBed
conda activate domainbed
```

---

### 1. Sanity Check on Debug28

Before running real experiments, we first check whether the training pipeline works correctly.

```bash
python -u -m domainbed.scripts.train \
  --data_dir ./data \
  --algorithm ERM \
  --dataset Debug28 \
  --test_env 0 \
  --steps 10 \
  --checkpoint_freq 5 \
  --skip_model_save \
  --hparams '{"resnet18": true, "resnet50_augmix": false}' \
  --output_dir ./outputs/debug_erm_steps10
```

A successful run should generate:

```text
outputs/debug_erm_steps10/
├── done
├── out.txt
├── err.txt
└── results.jsonl
```

---

### 2. Prepare PACS

The PACS dataset is not included in this repository.

After downloading and extracting the raw PACS images, organize the directory as:

```text
data/PACS/
├── art_painting/
├── cartoon/
├── photo/
└── sketch/
```

If the extracted directory is:

```text
data/PACS_tmp/kfold/
```

move it to the expected location:

```bash
mv ./data/PACS_tmp/kfold ./data/PACS
rm -rf ./data/PACS_tmp
```

Check the structure:

```bash
find ./data/PACS -maxdepth 2 -type d
find ./data/PACS -type f | wc -l
```

The number of image files should be close to 9991.

---

### 3. PACS ERM Sanity Check

Run a short ERM experiment on PACS:

```bash
python -u -m domainbed.scripts.train \
  --data_dir ./data \
  --algorithm ERM \
  --dataset PACS \
  --test_env 0 \
  --steps 100 \
  --checkpoint_freq 50 \
  --skip_model_save \
  --hparams '{"resnet18": true, "resnet50_augmix": false}' \
  --output_dir ./outputs/PACS_ERM_env0_steps100
```

If successful, the log should show evaluation results at step 0, step 50, and step 99.

---

### 4. Run PACS Baselines

Run all PACS baseline experiments:

```bash
./run_pacs_baselines_full.sh
```

This script runs:

```text
Algorithms: ERM, CORAL, GroupDRO, IRM, Mixup
Test environments: 0, 1, 2, 3
Steps per run: 3000
```

In total, this corresponds to:

```text
5 algorithms × 4 test environments = 20 runs
```

To run the script in the background:

```bash
nohup ./run_pacs_baselines_full.sh > pacs_baselines_full.log 2>&1 &
```

To monitor progress:

```bash
tail -f pacs_baselines_full.log
```

To check how many runs have finished:

```bash
find ./outputs -maxdepth 2 -name done | grep PACS | wc -l
```

A complete PACS baseline run should output:

```text
20
```

---

### 5. Collect PACS Results

After all PACS baseline runs finish, collect the results:

```bash
mkdir -p results

python collect_domainbed_results.py \
  --dataset PACS \
  --steps 3000 \
  --algorithms ERM CORAL GroupDRO IRM Mixup \
  --select final \
  --csv results/pacs_results_final_steps3000.csv \
  --md results/pacs_results_final_steps3000.md
```

This generates:

```text
results/pacs_results_final_steps3000.csv
results/pacs_results_final_steps3000.md
```

The resulting table contains:

```text
Method | Art | Cartoon | Photo | Sketch | Avg | Worst
```

For each `test_env`, we use the corresponding held-out domain accuracy:

```text
test_env=0 -> env0_out_acc
test_env=1 -> env1_out_acc
test_env=2 -> env2_out_acc
test_env=3 -> env3_out_acc
```

---

### 6. Prepare VLCS

VLCS is used as the second dataset.

The expected structure is:

```text
data/VLCS/
├── VOC2007/
├── LABELME/
├── CALTECH/
└── SUN09/
```

The exact folder names should match the environment names used in `domainbed/datasets.py`.

To inspect the expected VLCS environments:

```bash
grep -n "class VLCS" -A80 domainbed/datasets.py
```

After preparing VLCS, run a short sanity check:

```bash
python -u -m domainbed.scripts.train \
  --data_dir ./data \
  --algorithm ERM \
  --dataset VLCS \
  --test_env 0 \
  --steps 100 \
  --checkpoint_freq 50 \
  --skip_model_save \
  --hparams '{"resnet18": true, "resnet50_augmix": false}' \
  --output_dir ./outputs/VLCS_ERM_env0_steps100
```

---

### 7. Run VLCS Baselines

After VLCS is prepared and the sanity check succeeds, run the VLCS baseline script:

```bash
./run_vlcs_baselines_full.sh
```

Or run it in the background:

```bash
nohup ./run_vlcs_baselines_full.sh > vlcs_baselines_full.log 2>&1 &
```

Collect VLCS results:

```bash
python collect_domainbed_results.py \
  --dataset VLCS \
  --steps 3000 \
  --algorithms ERM CORAL GroupDRO IRM Mixup \
  --select final \
  --csv results/vlcs_results_final_steps3000.csv \
  --md results/vlcs_results_final_steps3000.md
```

---

### 8. Run the Proposed Method

After implementing `DWA_CORAL`, first run a short sanity check:

```bash
python -u -m domainbed.scripts.train \
  --data_dir ./data \
  --algorithm DWA_CORAL \
  --dataset Debug28 \
  --test_env 0 \
  --steps 10 \
  --checkpoint_freq 5 \
  --skip_model_save \
  --hparams '{"resnet18": true, "resnet50_augmix": false}' \
  --output_dir ./outputs/debug_DWA_CORAL_steps10
```

Then test it on PACS:

```bash
python -u -m domainbed.scripts.train \
  --data_dir ./data \
  --algorithm DWA_CORAL \
  --dataset PACS \
  --test_env 0 \
  --steps 100 \
  --checkpoint_freq 50 \
  --skip_model_save \
  --hparams '{"resnet18": true, "resnet50_augmix": false}' \
  --output_dir ./outputs/PACS_DWA_CORAL_env0_steps100
```

If the sanity checks pass, run the full PACS and VLCS experiments for the proposed method.

---

## Notes

- Raw datasets are not included in this repository.
- Full training outputs under `outputs/` are not committed.
- Only processed result tables under `results/` are committed.
- All baselines and proposed methods should use the same dataloader, backbone, training steps, and evaluation protocol for fair comparison.
