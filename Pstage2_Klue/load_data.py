import pickle as pickle
import os
import pandas as pd
import torch

# Dataset 구성.
class RE_Dataset(torch.utils.data.Dataset):
  def __init__(self, tokenized_dataset, labels):
    self.tokenized_dataset = tokenized_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.tokenized_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)


# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고
def preprocessing_dataset(dataset, label_type):
  print(dataset)
  label = []
  for i in dataset[8]:
    if i == 'blind':
      label.append(100)
    else:
      label.append(label_type[i])
  out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2],'entity_02':dataset[5],'label':label,})
  return out_dataset


def load_data(dataset_dir):
  # load label_type, classes
  with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)
  # load dataset
  dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
  # preprecessing dataset
  dataset = preprocessing_dataset(dataset, label_type)
  
  return dataset

# bert input을 위한 tokenizing.
def single_tokenized_dataset(dataset, tokenizer, max_length, entity_token):
    concat_entity = list(dataset['entity_01']  + '</s>' + dataset['entity_02'])
    tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation='only_second',
      max_length=max_length,
      add_special_tokens=True,
      )
    return tokenized_sentences

# bert input을 위한 tokenizing.ver2 
# QA문제와 같이 만들어줌
def multi_tokenized_dataset(dataset, tokenizer,max_length,entity_token):
    concat_entity = []
    concat_entity = list(dataset['entity_01'] + '관계' + dataset['entity_02'])
    tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=max_length,
      add_special_tokens=True,
      )
    return tokenized_sentences