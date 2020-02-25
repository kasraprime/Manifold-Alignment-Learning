# Manifold-Alignment-Learning

Implementing different manifold alignment approaches
Approaches including:
1. CCA
2. Deep CCA by Galen Adrew

In order to run the code for training phase:
```bash
python train.py --wandb_track 0 --experiment_name deepCCA --epochs 50 --task uw
```
Training took 6:43:25 to complete on a single GPU GeForce GTX 750 Ti.

To run the code for test:
```bash
python test.py --experiment_name deepCCA --task uw
```
