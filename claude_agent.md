PACS data isn't downloaded locally, so I can only verify Debug28 actually runs. The implementation is complete. Here's a summary of what I created.

Files created (existing files untouched)
domainbed/dwa_algorithms_v2.py — improved DWA_CORAL
domainbed/dwa_hparams_registry_v2.py — retuned defaults
domainbed/scripts/dwa_train_v2.py — new entry point
Why v1 scored ~20%
coral_loss = sum_e w_e * ||C_e - C_bar||_F^2 was unnormalized. With 512-dim ResNet18 features, ||C - C_bar||_F^2 is on the order of 10^4–10^6, so even with lambda=1e-3 the CORAL term dwarfed the cross-entropy (~2.0). The optimizer mostly minimized covariance distance, not labels.

Fixes in v2
Standard CORAL normalization — divide by 4 * d^2 (dwa_algorithms_v2.py:75-79). The smoke test confirms coral_loss is now ~1e-7 instead of ~1e5.
Numerically stable softmax — subtract det.max() before scaling by tau (dwa_algorithms_v2.py:51-53).
Warmup phase — uniform weights for the first dwa_coral_warmup steps so the classifier sees every domain equally before adaptive weighting kicks in (dwa_algorithms_v2.py:45-49).
Retuned defaults — lambda=1.0, beta=1e-2, tau=0.5, warmup=100 (dwa_hparams_registry_v2.py:25-28).
update() returns the six fields you specified: loss, cls_loss, coral_loss, risk_var, min_domain_loss, max_domain_loss.

Commands
Since the new code lives in v2 files, use the v2 entry point (the existing dwa_train.py still points at the original v1):


python -u -m domainbed.scripts.dwa_train_v2 \
  --data_dir ./data --algorithm DWA_CORAL --dataset Debug28 \
  --test_env 0 --steps 10 --checkpoint_freq 5 --skip_model_save \
  --hparams '{"resnet18": true, "resnet50_augmix": false}' \
  --output_dir ./outputs/debug_DWA_CORAL_v2_steps10

python -u -m domainbed.scripts.dwa_train_v2 \
  --data_dir ./data --algorithm DWA_CORAL --dataset PACS \
  --test_env 0 --steps 100 --checkpoint_freq 50 --skip_model_save \
  --hparams '{"resnet18": true, "resnet50_augmix": false}' \
  --output_dir ./outputs/PACS_DWA_CORAL_v2_env0_steps100
Debug28 smoke test ran successfully (10 steps, loss decreased from 0.69 → 0.14). PACS run wasn't verified — no ./data directory present locally.