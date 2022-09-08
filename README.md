# Misinformation Detection: A Multimodal Approach along with Sentiment Analysis

This project aims to find a multimodal way to deal with MID (Misinformation Detection) task.


## Directory structure

```
├── README.md
├── log
│   ├── ResNet50
│   ├── ViT
│   ├── bert
│   ├── images
│   │   ├── roberta.png
│   │   ├── roberta_resnet_concat.png
│   │   ├── roberta_resnet_lstm_attn_concat.png
│   │   └── roberta_vit_lstm_attn_concat.png
│   ├── roberta
│   ├── roberta_resnet_add
│   ├── roberta_resnet_avg
│   ├── roberta_resnet_concat
│   ├── roberta_resnet_lstm_attn_concat
│   ├── roberta_resnet_max
│   ├── roberta_vit_concat
│   ├── roberta_vit_lstm_attn_add
│   ├── roberta_vit_lstm_attn_avg
│   ├── roberta_vit_lstm_attn_concat_new
│   ├── roberta_vit_lstm_attn_max
│   ├── roberta_vit_lstm_attn_old
│   ├── vgg
│   └── xlnet
├── model
│   ├── eda.ipynb
│   ├── image_model_demo.ipynb
│   ├── mid_model_demo.ipynb
│   ├── sentiment_analysis.ipynb
│   └── text_model_demo.ipynb
├── test
│   └── text_model.py
└── work_track.md
```

All the model code are in the `model` directory, `log` directory track the results of all the models, `test` contains the python script for training.


## Environment

- python 3.10.4
- pytorch 1.11.0
- torchvision 0.12.0
- transformers 4.18.0

## Dataset

In this project, we choose the [Fakeddit](https://fakeddit.netlify.app/) (which is a multimodal dataset)

## Models

- `eda.ipynb`: this file do some brief exploration and data sampling
- `text_model_demo.ipynb`: textual unimodal model for MID
- `image_model_demo.ipynb`: image-based unimodal model for MID
- `sentiment_analysis.ipynb`: sentiment analysis based on the user comments
- `mid_model_demo.ipynb`: Hybrid multimodal model.

## Train & Evaluation

For the uimodal models, just run the notebook. For the hybrid models, specify the layer hyparameters for the classifier.

- Using Concatenate Fusion method

  -  MLP structure: {Linear(1539, 768), ReLU, Linear(768, 6)}
  - LSTM+ATTN structure: {LSTM(1539, 768, 2), Linear(768, 512), BatchNormalisation, ReLU, Linear(512, 6}

- Using AVG/ADD/MAX method

  - MLP structure: {Linear(771, 512), ReLU, Linear(512, 6)}
  - LSTM+ATTN structure: {LSTM(771, 512, 2), Linear(512, 256), BatchNormalisation, ReLU, Linear(256, 6)}

> NOTE: the results might be different due to the computing device.