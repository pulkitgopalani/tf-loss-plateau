# tf-loss-plateau

Code for the paper "What Happens During the Loss Plateau? Understanding Abrupt Learning in Transformers" (NeurIPS 2025) [arXiv: 2506.13688](https://arxiv.org/abs/2506.13688).

## Setup

See `env.yml` for requirements; also add your W&B key to `train_*.py` files for tracking metrics. 

Jupyter notebook `mws_notebook.ipynb` can be used for minimal implementation of main results for MWS task.

In the `src/` directory, run the following for each figure in the paper:

| **Experiment Description** | **Command / Steps** |
|----------------------------|---------------------|
| MWS training (Fig. 2) | `python train.py --config configs/mws.yaml` |
| MWS: biasing attention map (Fig. 3, 4) | `python train_mws_att_scale.py --config configs/mws_att_scale_{0_2,0_5,2_5,10}.yaml` |
| MWS: optimal initialization (Fig. 5) | `python train_mws_optimal_init.py --config configs/mws.yaml` |
| Prefix sum training (Fig. 6, 11, 12) | `python train_prefix.py --config configs/prefix.yaml` |
| Repeat_{1,2,4} training (Fig. 7, 21, 22) | `python train_repeat.py --config configs/repeat{1,2,4}.yaml` |
| Pythia/OLMO inference (Fig. 8, 23-25) | `python {pythia,olmo}.py`, `python plot_llm.py` |
| Addition training (Fig. 9, 10) | `python train_add.py --config configs/add.yaml` |
| Permutation training (Fig. 13, 14) | `python train_permute.py --config configs/permute.yaml` |
| Histogram training (Fig. 15, 16) | `python train_hist.py --config configs/hist.yaml` |
| Reverse training (Fig. 17) | `python train_copy_reverse.py --config configs/reverse.yaml` |
| Copy training (Fig. 18) | `python train_copy_reverse.py --config configs/copy.yaml` |
| Concat training (Fig. 19) | `python train_mws_concat.py --config configs/mws_concat.yaml` |
| Attention output probing (Fig. 20) | `python train_mws_probe.py --config configs/mws_probe.yaml` |
| MWS training with varying configurations (Fig. 26-29) | Change respective hyperparameter in `python train_mws.py --config configs/mws.yaml` |
| Softmax attention training (Fig. 30; set `num_steps=2000` in config file) | `python train_mws_softmax.py --config configs/mws.yaml` |
| SGD training (Fig. 31) | `python train_mws_sgd.py --config configs/mws.yaml` |
| Muon training (Fig. 32, 33) | `python train_mws_muon.py --config configs/mws_muon.yaml` |
| Learning rate schedule (Fig. 34, 35) | `python train_mws_lr_schedule.py --config configs/mws_lr_schedule.yaml` |
| Other optimization setups (Fig. 36-40) | Change respective hyperparameter / optimizer in `python train_mws.py --config configs/mws.yaml` |
| Representation collapse at various model layers (Fig. 41, 42) | `python train_mws_all_locs.py --config configs/mws.yaml` |
| MWS: biasing attention map at different train steps (Fig. 43) | `python train_mws_att_scale.py --config configs/mws_att_scale_10_{25,50,75,100}.yaml` |

