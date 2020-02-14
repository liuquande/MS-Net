## MS-Net: Multi-Site Network for Improving Prostate Segmentation with Heterogeneous MRI Data

### Introduction

Automated prostate segmentation in MRI is highly demanded for computer-assisted diagnosis. Recently, a variety of deep learning methods have achieved remarkable progress in this task, usually relying on large amounts of training data. Due to the nature of scarcity for medical images, it is important to effectively aggregate data from multiple sites for robust model training, to alleviate the insufficiency of single-site samples. However, the prostate MRIs from different sites present heterogeneity due to the differences in scanners and imaging protocols, raising challenges for effective ways of aggregating multi-site data for network training. In this paper, we propose a novel multi-site network (MS-Net) for improving prostate segmentation by learning robust representations, leveraging multiple sources of data. To compensate for the inter-site heterogeneity of different MRI datasets, we develop Domain-Specific Batch Normalization layers in the network backbone, enabling the network to estimate statistics and perform feature normalization for each site separately. Considering the difficulty of capturing the shared knowledge from multiple datasets, a novel learning paradigm, i.e., Multi-site-guided Knowledge Transfer, is proposed to enhance the kernels to extract more generic representations from multi-site data. Extensive experiments on three heterogeneous prostate MRI datasets demonstrate that our MS-Net improves the performance across all datasets consistently, and outperforms state-of-the-art methods for multi-site learning.

### Installation


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
   
### Questions

Please contact 'qdliu@cse.cuhk.edu.hk'

