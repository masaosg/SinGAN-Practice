# SinGAN-Practice

> 注: これは[映像メディア学](https://www.hal.t.u-tokyo.ac.jp/~yamasaki/lecture/index.html)の授業の課題として [SinGAN](https://openaccess.thecvf.com/content_ICCV_2019/html/Shaham_SinGAN_Learning_a_Generative_Model_From_a_Single_Natural_Image_ICCV_2019_paper.html) を再実装したものであり，再利用等は一切の責任を負いません．  
> DISCLAIMER: This repository contains partial implementation of [SinGAN](https://openaccess.thecvf.com/content_ICCV_2019/html/Shaham_SinGAN_Learning_a_Generative_Model_From_a_Single_Natural_Image_ICCV_2019_paper.html), which is an assignment for ["Visual Media"](https://www.hal.t.u-tokyo.ac.jp/~yamasaki/lecture/index.html) class. This implementation is provided "as is" without any kind of gurantee.

対応する記事は[こちら](https://hackmd.io/@masao/BJMnnCEWw)

## Prerequisites

- Python 3.8.5 (via pyenv)
- GPU with CUDA support
- pip requirements

It is recommended to wrap everyting inside a Python virtual environment (venv).

## Train

```shell
# train model
> python3 ./src/train.py ./mountain.jpg
```

Trained model will be saved under `./trained/<filename_without_extension>`

## Generate

```shell
# generate 50 images.
> python3 ./src/generate.py ./mountain.jpg --num=50
```

Generated images will be saved under `./generated/<filename_without_extension>`
