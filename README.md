# A Few Moments Please: Scalable Graphon Learning via Moment Matching

This repository contains the official code for our paper "A Few Moments Please: Scalable Graphon Learning via Moment Matching," accepted at **NeurIPS 2025**.

Our work proposes **MomentNet**, a novel and scalable graphon estimator that directly recovers the graphon by leveraging subgraph counts (graph moments) and implicit neural representations (INRs). This approach bypasses the need for latent variable modeling and costly Gromov-Wasserstein optimization.

We also introduce **MomentMixup**, a data augmentation technique that operates by interpolating graph moments in the moment space to enhance graphon-based learning tasks.

<img width="948" height="376" alt="Method Diagram" src="https://github.com/user-attachments/assets/1925a8fb-379d-41c9-9970-65bb06c1713c" />

## Installation

```bash
conda create -n momentnet python=3.12
conda activate momentnet
pip install -r requirements.txt
```

This repository does not vendor ORCA. MomentNet uses ORCA for graphlet/motif
counting, so clone and build it locally before running the experiments:

```bash
git clone https://github.com/thocevar/orca.git orca
```

Build ORCA using a compiler suitable for your operating system. The executable should be available at `orca/orca`. If you already have ORCA
installed elsewhere, place or symlink the executable there.

## Recreate the Datasets

Datasets are not stored in this repository. Recreate the synthetic graphon
benchmark files with:

```bash
python DatasetGen.py --dataset-id 1
```

This writes `dataset/graphon_0.pkl` through `dataset/graphon_12.pkl`.

The scalability datasets from the paper can be regenerated with:

```bash
python DatasetGen.py --dataset-id 2
python DatasetGen.py --dataset-id 3
```

## Reproduce Graphon Experiments

```bash
python experiments/run.py --graphons all --overwrite
```

This runs the MomentNet graphon benchmarks, records GW distance and runtime,
resumes partial results, and keeps centrality metrics disabled by default.
Graphon 9 always uses the parallel path. Other graphons run sequentially by
default, and can be parallelized by trial with `--parallel`. Results are
written to `results/`. To run a subset:

```bash
python experiments/run.py --graphons 0,1,9,11,12 --overwrite
```

To parallelize every requested graphon:

```bash
python experiments/run.py --graphons all --parallel --gpus 0,1,3 --overwrite
```

For long runs:

```bash
bash experiments/run_background.sh momentnet_all \
  experiments/run.py --graphons all --overwrite
```

Centrality metrics are disabled by default because they are slower than GW
evaluation. Enable them explicitly with:

```bash
python experiments/run.py --centrality
```


## Acknowledgments

Some of the code in this repository was inspired by or adapted from the following outstanding projects. We thank the original authors for making their work public.

* **SIGL:** [github.com/aliaaz99/SIGL](https://github.com/aliaaz99/SIGL)
* **IGNR:** [github.com/Mishne-Lab/IGNR](https://github.com/Mishne-Lab/IGNR)
* **ORCA:** [github.com/thocevar/orca](https://github.com/thocevar/orca)
* **G-Mixup:** [github.com/ahxt/g-mixup](https://github.com/ahxt/g-mixup)

## Cite Our Paper

If you use MomentNet, MomentMixup, or this code in your research, please cite our paper.

```bibtex
@article{ramezanpour2025few,
  title={A Few Moments Please: Scalable Graphon Learning via Moment Matching},
  author={Ramezanpour, Reza and Tenorio, Victor M and Marques, Antonio G and Sabharwal, Ashutosh and Segarra, Santiago},
  journal={arXiv preprint arXiv:2506.04206},
  year={2025}
}
```

For graphon-mixture learning, where the data generator is a mixture of Graphons, please also see our new work:

```bibtex
@misc{azizpour2026momentsmodelsgraphonmixturelearning,
  title={From Moments to Models: Graphon-Mixture Learning for Mixup and Contrastive Learning},
  author={Ali Azizpour and Reza Ramezanpour and Santiago Segarra},
  year={2026},
  eprint={2510.03690},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2510.03690},
}
```
