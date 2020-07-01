# MS-Net: Multi-Site Network for Improving Prostate Segmentation with Heterogeneous MRI Data
by [Quande Liu](https://github.com/liuquande), [Qi Dou](http://www.cse.cuhk.edu.hk/~qdou/), [Lequan Yu](https://yulequan.github.io/), [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/). 

### Introduction

The Tensorflow implementation for our TMI 2020 paper '[Multi-Site Network for Improving Prostate Segmentation with Heterogeneous MRI Data](https://arxiv.org/abs/2002.03366)'. 

![](assets/overview.png)

### Prerequisites

```
python==2.7.17
numpy==1.16.6
scipy==1.2.1
tensorflow-gpu==1.12.0
tensorboard==1.12.2
SimpleITK==1.2.0
```
### Usage

1. Train the model:
  First, you need to specify the training configurations (can simply use the default setting) in main.py.
  Then run:
   ```shell
   python main.py --phase=train
   ```

2. Evaluate the model:

    Run:
   ```shell
   python main.py --phase=test --restore_model='xxxx'
   ```
   You will see the output results in the folder `./output/`.

### Citation
If this repository is useful for your research, please cite:

```
@article{liu2020ms,
  title={Ms-net: Multi-site network for improving prostate segmentation with heterogeneous mri data},
  author={Liu, Quande and Dou, Qi and Yu, Lequan and Heng, Pheng Ann},
  journal={IEEE Transactions on Medical Imaging},
  year={2020},
  publisher={IEEE}
}
```
### Questions

Please contact 'qdliu@cse.cuhk.edu.hk'

