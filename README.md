# A Few Moments Please: Scalable Graphon Learning via Moment Matching

This repository contains the official code for our paper "A Few Moments Please: Scalable Graphon Learning via Moment Matching," accepted at **NeurIPS 2025**.

Our work proposes **MomentNet**, a novel and scalable graphon estimator that directly recovers the graphon by leveraging subgraph counts (graph moments) and implicit neural representations (INRs). This approach bypasses the need for latent variable modeling and costly Gromov-Wasserstein optimization.

We also introduce **MomentMixup**, a data augmentation technique that operates by interpolating graph moments in the moment space to enhance graphon-based learning tasks.

<img width="948" height="376" alt="Method Diagram" src="https://github.com/user-attachments/assets/1925a8fb-379d-41c9-9970-65bb06c1713c" />


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

