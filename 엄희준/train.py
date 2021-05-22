import pickle as pickle
import os
import pandas as pd
import numpy as np
import random
import wandb
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, EarlyStoppingCallback
from transformers import AdamW, get_linear_schedule_with_warmup
from load_data import *
from importlib import import_module
from sklearn.model_selection import train_test_split
import argparse


# metrics function for evaluation
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

# set fixed random seed
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
def train(args):
    wandb.login()
    seed_everything(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    MODEL_NAME = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    train_dataset_dir = "/opt/ml/code2/data/train_new_tag.tsv"
    train_data = load_dataset(train_dataset_dir)
    train_x = list(train_data.iloc[:,0])
    train_y  = list(train_data.iloc[:,-1])

    valid_dataset_dir = "/opt/ml/code2/data/valid_tag.tsv"
    valid_data = load_dataset(valid_dataset_dir)
    val_x = list(valid_data.iloc[:,0])
    val_y  = list(valid_data.iloc[:,-1])



    
    # tokenize datasets
    tokenized_train = tokenized_dataset(train_x, tokenizer)
    tokenized_val = tokenized_dataset(val_x, tokenizer)

    # make dataset for pytorch
    RE_train_dataset = RE_Dataset(tokenized_train, train_y)
    RE_valid_dataset = RE_Dataset(tokenized_val, val_y)

    # instantiate pretrained language model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=8)
    model.resize_token_embeddings(len(tokenizer)) 
    model.to(device)
    
    # optimizer and scheduler
    # optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=300*args.epochs)
    
    # callbacks
    early_stopping = EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience, early_stopping_threshold=0.00005)

    training_args = TrainingArguments(
        output_dir='./results',          
        logging_dir='./logs',                         
        logging_steps=100,
        save_total_limit=1,
        evaluation_strategy='steps',
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        dataloader_num_workers=args.num_workers,
        fp16=True,

        seed=args.seed,
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        label_smoothing_factor=args.label_smoothing_factor,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
    )

    trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset= RE_valid_dataset,             # evaluation dataset
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,         # define metrics function
    # optimizers=[optimizer, scheduler],
    callbacks=[early_stopping]
    )

    # train model
    trainer.train()
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2021, help='seed (default = 2021)')
    parser.add_argument('--model_name', type=str, default='xlm-roberta-large', help='model_name (default = xlm-roberta-large)')
    parser.add_argument('--run_name', type=str, default='tag', help='wandb run name (default = tag)')
    parser.add_argument('--num_workers', type=int, default=4, help='CPU num_workers (default = 4)')
    parser.add_argument('--epochs', type=int, default=20, help='epochs (default = 15)')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate(default = 1e-5)')
    parser.add_argument('--train_batch_size', type=int, default=64, help='train_batch_size (default = 64)')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='eval_batch_size (default = 64)')
    parser.add_argument('--warmup_steps', type=int, default=300, help='warmup_steps (default = 300)')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay (default = 0.001)')
    parser.add_argument('--label_smoothing_factor', type=str, default=0.3, help='label_smoothing_factor (default = 0.3)')
    parser.add_argument('--early_stopping_patience', type=str, default=3, help='early_stopping_patience (default = 3)')
    
    args = parser.parse_args()
    print(args)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    train(args)
