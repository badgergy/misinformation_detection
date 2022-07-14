"""
Text classifier using Bert, RoBerta, XlNet
"""
import os
from pip import main
import torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel


# TODO: load, clean and format data
def load_data():
    pass

# TODO: preprocessing text e.g. remove stopwords, lemmatization, stemming ...
def text_preprocessing():

    pass

# TODO: load data with dataloader
class MidDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        return 

    def __len__(self):
        return len()

# TODO: tokenizer use pretrained tokenizer to tokenize the text data
def tokenize():
    # load pretrained tokenizer
    return 

# TODO: model
class TextClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO: load pretrained model from transformer
        self.pre_trained
        self.classifier = nn.Sequential(
            nn.Linear(),
            nn.ReLU(),
            nn.Linear()
        )
    
    def forward():
        return "logits"

# Training loop
def train():
    pass

# Evaluation metrics
def evaluation():
    pass

def main():
    pass

if __name__ == "__main__":
    main()