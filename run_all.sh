#!/usr/bin/env bash
# End-to-end reproduction of every number and figure in results/ (CPU only; roughly 4-6 hours in total on a laptop).
# Order = dependency order: data -> GAN -> FID -> classifier experiments -> figures/tables.
set -euo pipefail  # stop at the first error

python -m pytest -q tests                                  # sanity checks first
python -m ganaug.explore                                   # dataset statistics (issue #1)

python -m ganaug.train_gan   --dataset breastmnist --epochs 300 --early-epoch 30           # issue #3/#4, ~15 min
python -m ganaug.fid         --dataset breastmnist                                        # issue #6
python -m ganaug.experiments --dataset breastmnist --seeds 0 1 2 3 4 --epochs 40           # issue #5/#7, 9 settings x 5 seeds
python -m ganaug.report      --dataset breastmnist                                        # figures + tables

python -m ganaug.train_gan   --dataset pneumoniamnist --epochs 60 --early-epoch 6 --sample-every 10
python -m ganaug.fid         --dataset pneumoniamnist
python -m ganaug.experiments --dataset pneumoniamnist --seeds 0 1 2 --epochs 15 \
    --settings real classical gan_final_50 gan_final_100 gan_final_200 gan_early_100 gan_final_balance   # 7 key settings (CPU budget)
python -m ganaug.report      --dataset pneumoniamnist
