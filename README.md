# Bi-Encoder, Poly-Encoder, and Cross-Encoder for Response Selection Tasks

- This repository is an unofficial re-implementation of [Poly-encoders: Transformer Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring](https://arxiv.org/abs/1905.01969v4).

- Special thanks to sfzhou5678! Some of the data preprocessing (dataset.py) and training loop code is adapted from his [github repo](https://github.com/sfzhou5678/PolyEncoder). However, the model architecture and data representation in that repository do not follow the paper exactly, thus leading to worse performance. I re-implement the model for Bi-Encoder and Poly-Encoder in encoder.py. In addition, the model and data processing pipeline of cross encoder are also implemented.

- Most of the training code in run.py is adpated from [examples](https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py?fbclid=IwAR3BlKIJYak659a6X12gsYOMs1JJPtnsdFUmn93CovwTJ5VXQZX1TK78yGo#L102) in the [huggingface](https://github.com/huggingface/transformers) repository.

- The most important architectural difference between this implementation and the original paper is that only one bert encoder is used (instead of two separate ones). Please refer to this [issue](https://github.com/chijames/Poly-Encoder/issues/4#issue-728612873) for details. However, this should not affect the performance much.

- This repository does not implement all details as in the original paper, for example, learning rate decay by 0.4 when plateau. Also due to limited computing resources, I cannot use the exact parameter settings such as batch size or context length as in the original paper. In addition, a much smaller bert model is used. Feel free to tune them or use larger models if you have more computing resources.

## Requirements

- Please see requirements.txt.

## Bert Model Setup

1. Download [BERT model](https://storage.googleapis.com/bert_models/2020_02_20/all_bert_models.zip) from Google.

2. Pick the model you like (I am using uncased_L-4_H-512_A-8.zip) and move it into bert_model/ then unzip it.

3. cd bert_model/ then bash run.sh

## Ubuntu Data

1. Download and unzip the [ubuntu data](https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntudata.zip?dl=0).

2. Rename valid.txt to dev.txt for consistency.

## DSTC 7 Data

1. Download the data from the official competition [site](https://ibm.github.io/dstc-noesis/public/datasets.html), specifically, download train (ubuntu_train_subtask_1.json), valid (ubuntu_dev_subtask_1.json), test (ubuntu_responses_subtask_1.tsv, ubuntu_test_subtask_1.json) split of subtask 1 and put them in the dstc7/ folder.

2. cd dstc7/ then bash parse.sh

## DSTC 7 Augmented Data (from ParlAI)

0. This dataset setting does not work for cross encoder. For details, please refer to this [issue](https://github.com/chijames/Poly-Encoder/issues/7).

1. Download the data from ParlAI [website](http://parl.ai/downloads/dstc7/dstc7_v2.tgz) and keep only ubuntu_train_subtask_1_augmented.json.

2. Move ubuntu_train_subtask_1_augmented.json into dstc7_aug/ then python3 parse.py.

3. Copy the dev.txt and test.txt file from dstc7/ into dstc7_aug/ since only training file is augmented.

4. You can refer to the original [post](https://github.com/facebookresearch/ParlAI/issues/2306#issuecomment-571284065) discussing the construction of this augmented data.

## Run Experiments (on dstc7)

1. Train a **Bi-Encoder**:

   ```shell
   python3 run.py --bert_model bert_model/ --output_dir output_dstc7/ --train_dir dstc7/ --use_pretrain --architecture bi
   ```

2. Train a **Poly-Encoder** with 16 codes:

   ```shell
   python3 run.py --bert_model bert_model/ --output_dir output_dstc7/ --train_dir dstc7/ --use_pretrain --architecture poly --poly_m 16
   ```

3. Train a **Cross-Encoder**:

   ```shell
   python3 run.py --bert_model bert_model/ --output_dir output_dstc7/ --train_dir dstc7/ --use_pretrain --architecture cross
   ```

4. Simply change the name of directories to ubuntu and run experiments on the ubuntu dataset.

## Inference

1. Test on **Bi_Encoder**:

   ```shell
   python3 run.py --bert_model bert_model/ --output_dir output_dstc7/ --train_dir dstc7/ --use_pretrain --architecture bi --eval
   ```

2. Test on **Poly_Encoder** with 16 codes:

   ```shell
   python3 run.py --bert_model bert_model/ --output_dir output_dstc7/ --train_dir dstc7/ --use_pretrain --architecture poly --poly_m 16 --eval
   ```

3. Test on **Cross_Encoder**:

   ```shell
   python3 run.py --bert_model bert_model/ --output_dir output_dstc7/ --train_dir dstc7/ --use_pretrain --architecture cross --eval
   ```

## Results

- All the experiments are done on a single GTX 1080 GPU with 8G memory and i7-6700K CPU @ 4.00GHz.

- Default parameters in run.py are used, please refer to run.py for details.

- The results are calculated on sampled portion (1000 instances) of dev set.

- da = data augmentation, we only report one result with poly vectors=64 and bert-base (uncased_L-12_H-768_A-12) with data augmentation (dstc7_aug). This result is really close to numbers reported in the original paper.

Ubuntu:
|       Model       |   **R@1**  |   **R@2**  |  **R@5**   |  **R@10**  |  **MRR**   |
| :---------------: | :--------: | :--------: | :--------: | :--------: | :--------: |
|    Bi-Encoder     |   0.760    |   0.855    |   0.971    |   1.00     |   0.844    |
| Poly-Encoder  16  |   0.766    |   0.868    |   0.974    |   1.00     |   0.851    |
| Poly-Encoder  64  |   0.767    |   0.880    |   0.979    |   1.00     |   0.854    |
| Poly-Encoder  360 |   0.754    |   0.858    |   0.970    |   1.00     |   0.842    |

DSTC 7:
|       Model       |   **R@1**  |   **R@2**  |  **R@5**   |  **R@10**  |  **MRR**   |
| :---------------: | :--------: | :--------: | :--------: | :--------: | :--------: |
|    Bi-Encoder     |   0.437    |   0.524    |   0.644    |   0.753    |   0.538    |
| Poly-Encoder  16  |   0.447    |   0.534    |   0.668    |   0.760    |   0.550    |
| Poly-Encoder  64  |   0.438    |   0.540    |   0.668    |   0.755    |   0.546    |
| Poly-Encoder  360 |   0.453    |   0.553    |   0.665    |   0.751    |   0.545    |
| Cross-Encoder     |   0.502    |   0.595    |   0.712    |   0.790    |   0.599    |
| da + bert base    |   0.561    |   0.659    |   0.765    |   0.858    |   0.659    |



## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=chijames/Poly-Encoder&type=Date)](https://star-history.com/#chijames/Poly-Encoder&Date)

