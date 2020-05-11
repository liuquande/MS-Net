# MS-Net: Multi-Site Network for Improving Prostate Segmentation with Heterogeneous MRI Data
by [Quande Liu](https://github.com/liuquande), [Qi Dou](http://www.cse.cuhk.edu.hk/~qdou/), [Lequan Yu](https://yulequan.github.io/), [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/). 

### Introduction

This Tensorflow implementation for our TMI 2020 paper '[Multi-Site Network for Improving Prostate Segmentation with Heterogeneous MRI Data](https://arxiv.org/abs/2002.03366)'. 


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
If MS-Net is useful for your research, please consider citing:

```
@inproceedings{liu2020msnet,
    author = {Quande Liu and Qi Dou and Lequan Yu and Pheng Ann Heng},
    title = {MS-Net: Multi-Site Network for Improving Prostate Segmentation with Heterogeneous MRI Data},
    booktitle = {IEEE Transactions on Medical Imaging},
    year = {2020},
}
```
### Questions

Please contact 'qdliu@cse.cuhk.edu.hk'

