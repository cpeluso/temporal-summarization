#from google.colab import drive, output
#drive.mount('/content/gdrive', force_remount=True)
#% cd "gdrive/MyDrive/MT"

#!pip install import-ipynb -q
#!pip install transformers -q

#import import_ipynb

#output.clear()

import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import pandas as pd
import math

from src.data.datagenerator import get_dataset_path

class TrecDataset(torch.utils.data.Dataset):
  def __init__(self, encodings, labels):
      self.encodings = encodings
      self.labels    = labels

  def __getitem__(self, idx):
      item = {k: torch.tensor(list(map(int, v[idx].strip('][').split(', ')))) for k, v in self.encodings.items()}
      item["labels"] = torch.tensor(
          [list(map(int, self.labels[idx].strip('][').split(', ')))], 
          dtype = torch.float
      )
      return item

  def __len__(self):
      return len(self.labels)

class DataManager():

  def __init__(self):
    pass

  def get_dataset(
    self,
    dataset:       list,
    encoder:       str,
    encoder_type:  bool,
    binary:        bool,
    context:       bool,
    only_relevant: bool
  ) -> pd.DataFrame:

    dataset_path = get_dataset_path(dataset, encoder, encoder_type, binary, context, only_relevant)
    return pd.read_csv(dataset_path)

  def split_datasets(
    self,
    data:            pd.DataFrame, 
    train_size:      float, 
    test_size:       float,
    split_on_topics: bool = False
  ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):

    if split_on_topics:
      train_data, valid_data, test_data = self.__split_datasets_on_topics(data, train_size, test_size)
    else:
      train_data, valid_data, test_data = self.__split_datasets_on_percentages(data, train_size, test_size)

    print("FULL  Dataset: {}".format(data.shape))
    print("TRAIN Dataset: {}".format(train_data.shape))
    print("VALID Dataset: {}".format(valid_data.shape))
    print("TEST  Dataset: {}".format(test_data.shape))

    return train_data, valid_data, test_data

  def __split_datasets_on_topics(
    self,
    data:       pd.DataFrame,
    train_size: float, 
    test_size:  float
  ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):

    topics   = list(data["query"].unique())
    n_topics = len(topics)               

    n_train_topics = int(n_topics * train_size) 

    n_valid_topics = n_topics - n_train_topics 
    n_valid_topics = math.ceil(n_valid_topics * (1 - test_size))

    train_topics   = topics[:n_train_topics]
    test_topics    = topics[n_train_topics+n_valid_topics:]

    train_data = data[data["query"].isin(train_topics)]
    valid_data = data.drop(train_data.index).reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)

    test_data  = valid_data[valid_data["query"].isin(test_topics)]
    valid_data = valid_data.drop(test_data.index).reset_index(drop=True)
    test_data  = test_data.reset_index(drop=True)

    return train_data, valid_data, test_data

  def __split_datasets_on_percentages(
    self,
    data:       pd.DataFrame, 
    train_size: float, 
    test_size:  float
  ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):

    train_data = data.sample(frac=train_size,random_state=200)
    valid_data = data.drop(train_data.index).reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)

    test_data  = valid_data.sample(frac=test_size,random_state=200)
    valid_data = valid_data.drop(test_data.index).reset_index(drop=True)
    test_data  = test_data.reset_index(drop=True)

    return train_data, valid_data, test_data

  def load_torch_datasets(
    self,
    train_data: pd.DataFrame, 
    valid_data: pd.DataFrame,
    test_data:  pd.DataFrame
  ) -> (TrecDataset, TrecDataset, TrecDataset):

    training_set   = TrecDataset(train_data[['input_ids', 'attention_mask']], train_data['mask'])
    validation_set = TrecDataset(valid_data[['input_ids', 'attention_mask']], valid_data['mask'])
    testing_set    = TrecDataset(test_data[['input_ids', 'attention_mask']],  test_data['mask'])

    return training_set, validation_set, testing_set

  def load_torch_dataloaders(
    self,
    training_set:   TrecDataset,
    validation_set: TrecDataset,
    testing_set:    TrecDataset,
    train_params:   dict, 
    valid_params:   dict,
    test_params:    dict
  ) -> (TorchDataLoader, TorchDataLoader, TorchDataLoader):

    training_loader   = TorchDataLoader(training_set,   **train_params)
    validation_loader = TorchDataLoader(validation_set, **valid_params)
    testing_loader    = TorchDataLoader(testing_set,    **test_params)

    return training_loader, validation_loader, testing_loader