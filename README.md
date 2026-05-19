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
