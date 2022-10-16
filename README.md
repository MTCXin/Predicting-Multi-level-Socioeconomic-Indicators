# Predicting-Multi-level-Socioeconomic-Indicators
Repo for "Predicting Multi-level Socioeconomic Indicators from Structural Urban Imagery" CIKM 2022
A framework for Predicting Multi-level Socioeconomic Indicators from Structural Urban Imagery.


## Enviroment setup



## Running code

To pretrain the model on CIFAR-10 with a *single* GPU, try the following command:

```
python run.py --train_mode=pretrain \
  --train_batch_size=512 --train_epochs=1000 \
  --learning_rate=1.0 --weight_decay=1e-4 --temperature=0.5 \
  --dataset=cifar10 --image_size=32 --eval_split=test --resnet_depth=18 \
  --use_blur=False --color_jitter_strength=0.5 \
  --model_dir=/tmp/simclr_test --use_tpu=False
```


## Cite

[SimCLR paper](https://arxiv.org/abs/2002.05709):

```
@article{chen2020simple,
  title={A Simple Framework for Contrastive Learning of Visual Representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  journal={arXiv preprint arXiv:2002.05709},
  year={2020}
}
```