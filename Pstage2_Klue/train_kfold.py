import pickle as pickle
import os
import pandas as pd
import numpy as np
import random
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, Trainer, TrainingArguments,  AutoModelForSequenceClassification , EarlyStoppingCallback
from load_data import *
from importlib import import_module
from sklearn.model_selection import StratifiedShuffleSplit , StratifiedKFold
import argparse
import wandb
import gc

# 평가를 위한 metrics function.
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
          'accuracy': acc,
    }

# seed 고정 
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
    seed_everything(args.random_seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load model and tokenizer
    MODEL_NAME = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    #load dataset
    if args.entity_token :
        dataset = load_data("/opt/ml/input/data/train/entity_train.tsv" )
    elif args.unk : 
        dataset = load_data("/opt/ml/input/data/train/unk_entity_train.tsv" )
    else :
        dataset = load_data("/opt/ml/input/data/train/new_train.tsv")
    label  = dataset['label'].values
    
    # tokenizing dataset : Single / Multi
    tokenized_module = getattr(import_module("load_data") , args.tok_data + '_tokenized_dataset')

    # make Kfold dataset for pytorch.
    cv = StratifiedKFold(n_splits=5, shuffle = True ,random_state=args.random_seed)
    for idx , (train_idx , val_idx) in enumerate(cv.split(dataset, label)):
        train_dataset = dataset.iloc[train_idx]
        val_dataset = dataset.iloc[val_idx]

        tokenized_train = tokenized_module(train_dataset, tokenizer, args.max_length ,args.entity_token)
        tokenized_val = tokenized_module(val_dataset, tokenizer, args.max_length ,args.entity_token)

        train_y = label[train_idx]
        val_y = label[val_idx]

        # make dataset for pytorch.
        RE_train_dataset = RE_Dataset(tokenized_train, train_y)
        RE_valid_dataset = RE_Dataset(tokenized_val, val_y)
    
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME , num_labels = 42) 
        model.to(device)

        output_dir = './result' + str(idx)              # fold 별 output_dir 다르게 하기
        training_args = TrainingArguments(
            output_dir= output_dir,        
            save_total_limit=1,                             
            save_strategy = 'epoch',
            num_train_epochs = args.epochs,              
            learning_rate = args.lr,               
            per_device_train_batch_size = args.train_batch_size,  
            per_device_eval_batch_size =  args.eval_batch_size,  
            seed = args.random_seed,
            warmup_steps = args.warmup_steps,                
            weight_decay = args.weight_decay,               
            logging_dir='./logs',              
            logging_strategy = "epoch",
            evaluation_strategy="epoch", 
            dataloader_num_workers = 4,
            fp16 = True,
            run_name = args.run_name,
            load_best_model_at_end  = True,
            metric_for_best_model="accuracy",
            greater_is_better = True,
            group_by_length  = False,
            label_smoothing_factor = 0.5
        )

        early_stopping = EarlyStoppingCallback(early_stopping_patience = 3, early_stopping_threshold = 0.00005)
        trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset= RE_valid_dataset,             # evaluation dataset
        tokenizer = tokenizer,
        compute_metrics=compute_metrics,         # define metrics function
        callbacks=[early_stopping],
        #optimizers=[optimizer, scheduler]      # not default
        )

        # train model`
        trainer.train()
        wandb.finish()

        # OOM 방지
        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--random_seed' , type = int , default = 2021 , help = 'random seed (default = 2021)')
    parser.add_argument('--tok_data' , type = str , default = 'single' , help = 'tokenized_data_type : single / multi (default : single)')
    parser.add_argument('--entity_token' , type = bool , default = False , help = 'specail entity token')
    parser.add_argument('--val_ratio' , type = float , default = 0.2 , help = 'val_ratio (default = 0.2)')
    parser.add_argument('--epochs' , type = int , default = 10 , help = 'epochs (default = 4)')
    parser.add_argument('--lr' , type = float , default = 1e-5 , help = 'learning rate(default = 5e-5)')
    parser.add_argument('--max_length' , type = int , default = 100 , help = 'sequence length (default = 300)')
    parser.add_argument('--train_batch_size' , type = int , default = 32 , help = 'train_batch_size (default = 16)')
    parser.add_argument('--eval_batch_size' , type = int , default = 32 , help = 'eval_batch_size (default = 16)')
    parser.add_argument('--warmup_steps' , type = int , default = 300 , help = 'warmup_steps (default = 500)')
    parser.add_argument('--weight_decay' , type = float , default = 0.01 , help = 'weight_decay (default = 0.01)')    
    parser.add_argument('--num_workers' , type = int , default = 4 , help = 'CPU num_workers (default = 4)')

    # import arguments
    parser.add_argument('--model_name' , type = str , default='xlm-roberta-large' , help = 'model_name')
    parser.add_argument('--run_name' , type = str , default = 'base' , help = 'wandb run name (default = base)')
    
    
    args = parser.parse_args()
    print(args)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train(args)