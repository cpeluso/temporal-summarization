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

from src.data.s3_connector import download_data
from src.data.connector    import get_dataset_path

class TrecDataset(torch.utils.data.Dataset):
  """
  TrecDataset class. Extends the torch.utils.data.Dataset class.

  The TrecDataset class has the purpose of retrieving processed data from
  the 2013, 2014 and 2015 versions of the TREC Temporal Summarization datasets.

  This class is intended to be used only during the training phase.

  Attributes
  ----------
  encodings : pd.DataFrame
      A DataFrame containing only the columns produced during the encoding phase of the data
      (ie. ["input_ids", "attention_mask"]).
  labels : pd.Series
      A Series containing the lists representing the ground truth.
  """

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

class DataManager:
  """
  DataManager class.

  The DataManager class has the purpose of retrieving processed data from
  the 2013, 2014 and 2015 versions of the TREC Temporal Summarization datasets.

  This class is intended to be used only during the training phase.

  Attributes
  ----------
  datasets : list
      A list containing the names of the dataset that will be returned.
      Supported names are "2013", "2014", "2015".
      Ex: datasets = ["2013", "2014", "2015"]
  only_relevant : bool
      If True, returns a dataset with only the portion of the data containing relevant text.
      Otherwise, returns the whole dataset.
  tokenizer_name : str
      A string representing the name of the tokenizer that was used - offline - to encode the data.
      Supported names are "spanbert", "bert" and "roberta".
  tokenizer_type : str
      A string representing the type of the tokenizer that was used - offline - to encode the data.
      Supported names are "cased" and "uncased".
  binary_masks : bool
      If True, the labels referred to the data will be binary (eg. [0,1,1,1,1...,0,0,1,1,1,0]).
      Otherwise, the labels referred to the data will be not-binary (eg. [0,1,2,2,2,3,0,1,2,2,2,2,2,3,0,0,0,4,0,0]).
  contextual : bool
      If True, the textual data contains also the context.
      Otherwise, only the text data will be retrieved.
  train_size: float
      A float - between 0 and 1 - that identifies
      the percentage of (the whole) dataset that will be used for the training phase.
  test_size: float
      A float - between 0 and 1 - that identifies the percentage in which the validation set is splitted.
      Ex: train_size = 0.8, test_size = 0.5 means that the
        * 80% of the dataset is devoted to the training phase
        * 10% of the dataset is devoted to the validation phase
        * 10% of the dataset is devoted to the testing phase
  split_on_topics: bool = True
      If True, the dataset is split (considering the proportions defined from train_size and test_size) by topics.
      In this case, the validation set (for example) will contain topics that neither the training nor the test set
      will contain.
      Otherwise, the dataset is split only considering the train_size and test_size.
  Methods
  -------
  get_dataset():
      Returns the dataset requested through the parameters given during the initialization of the DataManager class.

  split_dataset(data: pd.DataFrame):
      Returns three DataFrames (train, validation and test),
      depending on the parameters given during the initialization of the DataManager class.

  load_torch_datasets(
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    test_data:  pd.DataFrame
  ):
    Returns three TrecDataset (train, validation and test).

  load_torch_dataloaders(
    training_set:   TrecDataset,
    validation_set: TrecDataset,
    testing_set:    TrecDataset,
    train_params:   dict,
    valid_params:   dict,
    test_params:    dict
  ):
    Returns three torch.utils.data.DataLoader (train, validation and test).
  """

  def __init__(
    self,
    datasets:        list,
    only_relevant:   bool,
    tokenizer_name:  str,
    tokenizer_type:  bool,
    binary_masks:    bool,
    contextual:      bool,
    train_size:      float,
    test_size:       float,
    split_on_topics: bool = True
  ) -> pd.DataFrame:

    self.dataset_path = get_dataset_path(datasets, tokenizer_name, tokenizer_type, binary_masks, contextual, only_relevant)

    self.train_size = train_size
    self.test_size  = test_size

    self.split_on_topics = split_on_topics

    pass


  def get_dataset(self) -> pd.DataFrame:
    """
    Returns the dataset requested 
    through the parameters given during the initialization of the DataManager class.
    """
    download_data(self.dataset_path)
    return pd.read_csv(self.dataset_path)


  def split_datasets(
    self,
    data: pd.DataFrame
  ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Returns three DataFrames (train, validation and test),
    depending on the parameters given during the initialization of the DataManager class.
    """

    if self.split_on_topics:
      train_data, valid_data, test_data = self.__split_datasets_on_topics(data)
    else:
      train_data, valid_data, test_data = self.__split_datasets_on_percentages(data)

    print("FULL  Dataset: {}".format(data.shape))
    print("TRAIN Dataset: {}".format(train_data.shape))
    print("VALID Dataset: {}".format(valid_data.shape))
    print("TEST  Dataset: {}".format(test_data.shape))

    return train_data, valid_data, test_data

  def __split_datasets_on_topics(
    self,
    data: pd.DataFrame,
  ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Splits the DataFrame by query,
    depending on the parameters given during the initialization of the DataManager class.
    """

    topics   = list(data["query"].unique())
    n_topics = len(topics)               

    n_train_topics = int(n_topics * self.train_size)

    n_valid_topics = n_topics - n_train_topics 
    n_valid_topics = math.ceil(n_valid_topics * (1 - self.test_size))

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
    data: pd.DataFrame
  ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Splits the DataFrame by percentage,
    depending on the parameters given during the initialization of the DataManager class.
    """

    train_data = data.sample(frac=self.train_size, random_state=200)
    valid_data = data.drop(train_data.index).reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)

    test_data  = valid_data.sample(frac=self.test_size, random_state=200)
    valid_data = valid_data.drop(test_data.index).reset_index(drop=True)
    test_data  = test_data.reset_index(drop=True)

    return train_data, valid_data, test_data


  def load_torch_datasets(
    self,
    train_data: pd.DataFrame, 
    valid_data: pd.DataFrame,
    test_data:  pd.DataFrame
  ) -> (TrecDataset, TrecDataset, TrecDataset):
    """
    Returns three TrecDataset (train, validation and test).
    """

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
    """
    Returns three torch.utils.data.DataLoader (train, validation and test).
    """

    training_loader   = TorchDataLoader(training_set,   **train_params)
    validation_loader = TorchDataLoader(validation_set, **valid_params)
    testing_loader    = TorchDataLoader(testing_set,    **test_params)

    return training_loader, validation_loader, testing_loader
