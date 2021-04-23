import pickle as pickle
import os
import pandas as pd
import numpy as np
import random
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, Trainer, TrainingArguments,  AutoModelForSequenceClassification , EarlyStoppingCallback
from transformers import AdamW , get_linear_schedule_with_warmup
from load_data import *
from importlib import import_module
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit , KFold
import argparse
import wandb


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

    # make dataset for pytorch.
    train_x , val_x , train_y , val_y = train_test_split(dataset , label ,  test_size = args.val_ratio, random_state = args.random_seed , shuffle = True, stratify = label)
    
    tokenized_train = tokenized_module(train_x, tokenizer, args.max_length ,args.entity_token)
    tokenized_val = tokenized_module(val_x, tokenizer, args.max_length ,args.entity_token)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_y)
    RE_valid_dataset = RE_Dataset(tokenized_val, val_y)

    # make dataset for pytorch.    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME , num_labels = 42) 
    model.to(device)
    # if args.entity_token :
    #     model.resize_token_embeddings(tokenizer.vocab_size + added_special_token_num)
    

    # embedding_para =[{'params': v, 'lr':1E-5} for k, v in model.roberta.embeddings.named_parameters()] 
    # encoder_para=[{'params': v, 'lr':1E-5}  for k, v in model.roberta.encoder.named_parameters()]
    # classifier_para=[{'params': v, 'lr':5E-5}  for k, v in model.classifier.named_parameters()]
    # all_para = embedding_para + encoder_para + classifier_para

    # optimizer = AdamW(all_para,weight_decay= args.weight_decay, eps =1e-8)
    # scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=3000)
  
    training_args = TrainingArguments(
        output_dir= './result',          
        save_total_limit=1,                             
        save_strategy = 'epoch',
        num_train_epochs = args.epochs,              
        learning_rate = args.lr,               
        per_device_train_batch_size = args.train_batch_size,  
        per_device_eval_batch_size =  args.eval_batch_size,  
        # https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.SchedulerType
        # lr_scheduler_type = args.lr_scheduler_type,
        #warmup_ratio = args.warmup_ratio,
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
        metric_for_best_model="loss",
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
    # optimizers=[optimizer, scheduler]
    )

    # train model`
    trainer.train()
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--random_seed' , type = int , default = 2021 , help = 'random seed (default = 2021)')
    parser.add_argument('--tok_data' , type = str , default = 'single' , help = 'tokenized_data_type : single / multi (default : single)')
    parser.add_argument('--entity_token' , type = bool , default = False , help = 'specail entity token')
    parser.add_argument('--val_ratio' , type = float , default = 0.2 , help = 'val_ratio (default = 0.2)')
    parser.add_argument('--epochs' , type = int , default = 15 , help = 'epochs (default = 4)')
    parser.add_argument('--lr' , type = float , default = 1e-5 , help = 'learning rate(default = 5e-5)')
    parser.add_argument('--max_length' , type = int , default = 100 , help = 'sequence length (default = 300)')
    parser.add_argument('--train_batch_size' , type = int , default = 32 , help = 'train_batch_size (default = 16)')
    parser.add_argument('--eval_batch_size' , type = int , default = 32 , help = 'eval_batch_size (default = 16)')
    #parser.add_argument('--lr_scheduler_type' , type = str , default = 300 , help = 'sequence length (default = 300)')
#     parser.add_argument('--warmup_ratio' , type = float , default = 0 , help = 'warmup_ratio (default = 0.5)')
    parser.add_argument('--warmup_steps' , type = int , default = 300 , help = 'warmup_steps (default = 500)')
    parser.add_argument('--weight_decay' , type = float , default = 0.01 , help = 'weight_decay (default = 0.01)')    
    parser.add_argument('--num_workers' , type = int , default = 4 , help = 'CPU num_workers (default = 4)')
    parser.add_argument('--unk' , type = bool , default = True , help = 'random unk append')

    # import arguments
    parser.add_argument('--model_name' , type = str , default='xlm-roberta-large' , help = 'model_name')
    parser.add_argument('--run_name' , type = str , default = 'base' , help = 'wandb run name (default = base)')
    
    
    args = parser.parse_args()
    print(args)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train(args)