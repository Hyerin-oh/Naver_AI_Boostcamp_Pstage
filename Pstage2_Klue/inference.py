from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertTokenizer , AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse
from importlib import import_module


def inference(model, tokenized_sent, device):
    logits = []
    predictions = []
    dataloader = DataLoader(tokenized_sent, batch_size=1, shuffle=False)
    model.eval()
    output_pred = []
  
    for i, data in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(
              input_ids=data['input_ids'].to(device),
              attention_mask=data['attention_mask'].to(device),
              #token_type_ids=data['token_type_ids'].to(device)
              )
        _logits = outputs[0].detach().cpu().numpy()      
        _predictions = np.argmax(_logits, axis=-1)
        logits.append(_logits)
        predictions.extend(_predictions.ravel())
    return np.concatenate(logits), np.array(predictions)


def load_test_dataset(dataset_dir, tokenizer):
    test_dataset = load_data(dataset_dir)
    test_label = test_dataset['label'].values
    # tokenizing dataset
    tokenized_module = getattr(import_module("load_data") , args.tok_data + '_tokenized_dataset')
    tokenized_test = tokenized_module(test_dataset, tokenizer, 100 ,  args.entity_token)
    return tokenized_test, test_label

def main(args):
    """
    주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
    TOK_NAME = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)
    MODEL_NAME = args.model_dir # model dir.

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME , num_labels = 42) 

    model.parameters
    model.to(device)

     #load test dataset
    if args.entity_token :
        test_dataset_dir = "/opt/ml/input/data/test/entity_test.tsv"
    else :
        test_dataset_dir = "/opt/ml/input/data/test/test.tsv"

    test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    test_dataset = RE_Dataset(test_dataset ,test_label)
    
  # make csv file with predicted answer.
    logits, predictions = inference(model, test_dataset, device)
    output = pd.DataFrame(predictions, columns=['pred'])
    output_name = '/opt/ml/code/prediction/submission' + str(args.idx) + '.csv'
    print(output_name)
    output.to_csv(output_name, index=False)
    np.save(os.path.join('/opt/ml/code/logit', r'logits'+str(args.idx)+'.npy'), logits)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
  
    # model dir
    parser.add_argument('--model_dir', type=str, default="./results/results/checkpoint-2000")
    parser.add_argument('--tok_data', type=str,  default = 'single' , help = 'tokenized_data_type : single / multi (default : single)')
    parser.add_argument('--entity_token' , type = bool , default = False , help = 'specail entity token')
    parser.add_argument('--model_name' , type = str , default='xlm-roberta-large'  , help = 'model_name')
    parser.add_argument('--idx' , type = int , default='0'  , help = 'prediction_num')
    args = parser.parse_args()
    print(args)
    main(args)