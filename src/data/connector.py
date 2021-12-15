"""
This file has the purpose of connecting dataloader.py to datagenerator.py.
In this way, dataloader.py remains inaccessible to the rest of the code.
"""

import pandas as pd

from src.data.dataloader       import DataLoader
from src.tests.data_processing import test_data_processing
from utils.tokenizers          import tokenizers

def get_dataset_path(
    dataset:        list,
    tokenizer:      str,
    tokenizer_type: bool,
    binary:         bool,
    context:        bool,
    only_relevant:  bool
) -> str:

    data = ""

    for name in dataset:
        data = data + name + "_"

    data = data[:-1]

    # RoBERTa tokenization does not differ in terms of cased / uncased
    if tokenizer == "roberta":
        path = f"{data}_{tokenizer}"
    else:
        path = f"{data}_{tokenizer}-{tokenizer_type}"

    binary_str = "_binary_" if binary else "_not-binary_"
    path = path + binary_str

    context_str = "contextual_" if context else "not-contextual_"
    path = path + context_str

    only_relevant_str = "only_relevant" if only_relevant else "full"
    path = path + only_relevant_str

    return path + ".csv"


def load_data(
  datasets:       list,
  only_relevant:  bool,
  tokenizer_name: str,
  tokenizer_type: str,
  max_num_words:  int,
  binary_masks:   bool,
  contextual:     bool,
  test:           bool
) -> pd.DataFrame:

  data_loader = DataLoader(datasets, only_relevant, tokenizer_name, tokenizer_type, max_num_words, binary_masks, contextual)
  data        = data_loader.load_data()
  if test:
    test_data_processing(data.copy(), tokenizers[tokenizer_name][tokenizer_type], max_num_words, tokenizer_type)

  return data
