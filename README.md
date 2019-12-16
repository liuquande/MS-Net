** MS-Net: Multi-Site Network for Improving Prostate Segmentation with Heterogeneous MRI Data **

### Introduction

To appear. 

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

