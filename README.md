# SINGLE REMOTE SENSING IMAGE DEHAZING USING A DUAL-STEP CASCADED RESIDUAL DENSE NETWORK [ICIP 2021]

## Abstract
Remote sensing (RS) dehazing is an extremely challenging task since the non-uniform distribution of haze and fog severely degrade the image and difficult to extract features. To address these issues, we propose an end to end Dual-step Cascaded Residual Dense Network called DCRD-Net, which can exactly remove haze from the hazy RS image and precisely restoring the details. The architecture of network contains two cascaded task-driven subnetworks, in order to deal with the coarse and fine haze-relevant features separately. Besides, the Residual Dense Enhancement Block (RDEB) is involved in the network to guide the feature extraction, so that multi-scale information can be used to estimate the local and global feature. Further, the Squeeze and Excitation (SE) Block is employed to optimize the RDEB, for the purpose to get the contextual feature and reduce the computation complex. Quantitative and qualitative results illustrate that the designed framework outperform the recent outstanding dehazing methods on promoting haze remove and restoring detailed in the synthetic and real-world RS images under various scenes. To encourage more comparisons, we will publicly release our codes on Github: https://github.com/cxtalk/DCRD-Net.

## Requirements
- CUDA 9.0
- Python 3.7
- Pytorch 1.4.0
- Torchvision 0.2.0

## Acknowledgments

Codes are heavily borrowed from [GridDehazeNet](https://github.com/proteus1991/GridDehazeNet).
