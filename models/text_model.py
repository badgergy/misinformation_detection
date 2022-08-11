"""
Text classifier using Bert, RoBerta, XlNet
"""
import os
import re

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import matplotlib.pyplot as plt

DATA_DIR = "~/Projects/Datasets/public_news_set"
TRAIN_FILE = "multimodal_train_cleaned.tsv"
TEST_FILE = "multimodal_test_cleaned.tsv"
VALID_FLIE = "multimodal_valid_cleaned.tsv"

MODEL_CKPT = "distilbert-base-uncased"
# MODEL_CKPT = "bert-base-uncased"
# MODEL_CKPT = "roberta-base"
# MODEL_CKPT = "xlnet-base-cased"

# Hyperparamters
MAX_LENGTH = 32
LEARNING_RATE = 2e-5
BATCH_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IDs to labels
LABEL2ID = {0: "True", 1: "Satire/Parody", 2: "Misleading Content", 3: "Imposter", 4: "False connection", 5: "Manipulated Content"}
LABELS = [LABEL2ID[i] for i in range(6)]

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_CKPT)



def same_seed(seed):
    '''
    Fixes random number generator seeds for reproducibility
    '''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data():
    """
    Load data with pandas
    """
    mid_train = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILE), sep='\t')
    mid_test = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE), sep='\t')
    mid_val = pd.read_csv(os.path.join(DATA_DIR, VALID_FLIE), sep='\t')
    return mid_train, mid_val, mid_test

def text_preprocessing(text):
    """
    - Lowercase
    - Remove entity name (e.g. @name)
    @param text (str): a string to be processed
    @return text (str): the processed string
    """
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# filter out useful data
def filter_data(df):
    df = df.filter(['clean_title', '6_way_label']).rename(columns={"clean_title": "text", "6_way_label": "label"})
    df['label_text'] = df['label'].apply(lambda x:LABEL2ID[x])
    return df

def tokenize(text):
    # load pretrained tokenizer
    return TOKENIZER(text, truncation=True, padding="max_length", max_length=MAX_LENGTH)

def preprocess_data(df):
    """
    Format data for model use
    """
    df = filter_data(df)
    df.text = df.text.apply(text_preprocessing)
    df.text = df.text.apply(tokenize)
    df['input_ids'], df['attention_mask'] = df.text.apply(lambda x: x['input_ids']), df.text.apply(lambda x: x['attention_mask'])

    return df

def form_dataloader(df):
    df = preprocess_data(df)
    ds = MidDataset(df)
    loader = DataLoader(ds, batch_size=BATCH_SIZE)
    return loader

class MidDataset(Dataset):
    """
    Load dataset with torch.utils.data.Dataset
    """
    def __init__(self, df):
        df.dtype = np.float64
        self.input_ids = df['input_ids'].values
        self.attention_mask = df['attention_mask'].values
        self.label = df['label'].values

    def __getitem__(self, idx):
        input_ids = torch.Tensor(self.input_ids[idx]).int()
        attention_mask = torch.Tensor(self.attention_mask[idx]).int()
        label = self.label[idx]
        return input_ids, attention_mask, label

    def __len__(self):
        return len(self.label)


class TextClassifier(nn.Module):
    def __init__(self, finetune=True) -> None:
        super().__init__()
        D_in, D_h, D_out = 768, 2 * MAX_LENGTH, len(LABELS)
        self.pretrained_model = AutoModel.from_pretrained(MODEL_CKPT)
        self.classifier = nn.Sequential(
            nn.Linear(D_in, D_h),
            nn.ReLU(),
            nn.Linear(D_h, D_out)
        )

        if not finetune:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state = outputs[0][:, 0, :]
        last_hidden_state = outputs.last_hidden_state[:, 0]
        logits = self.classifier(last_hidden_state)
        return logits


def train(model, train_loader, val_loader, epochs=4, evaluate=True):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(epochs):
        for input_ids, attention_masks, label in tqdm(train_loader, total=len(train_loader)):
            input_ids = input_ids.to(DEVICE)
            attention_masks = attention_masks.to(DEVICE)
            model.train()
            optimizer.zero_grad()
            output = model(input_ids, attention_masks)
            loss = loss_fn(output.cpu(), label)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # TODO: test model on val_set
        if evaluate:
            evaluation()


# TODO: Evaluation metrics
def evaluation(model, valid_loader):
    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask, label in valid_loader:
            output = model(input_ids, attention_mask)
            logits = F()

def test(model, test_loader):
    pass

def compute_metrics(pred, labels):
    preds = pred.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}
    
def plot_confusion_matrix(y_preds, y_true):
    cm = confusion_matrix(y_true, y_preds, normalize='true')
    fig, ax = plt.subplots(figsize=(6,6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format='.2f', ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()


def main():
    mid_train, mid_val, mid_test = load_data()
    train_loader = form_dataloader(mid_train)
    valid_loader = form_dataloader(mid_val)
    test_loader = form_dataloader(mid_test)
    model = TextClassifier().to(DEVICE)

    train(model, train_loader, valid_loader)


if __name__ == "__main__":
    main()