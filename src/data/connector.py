import pandas as pd

from src.data.dataloader       import DataLoader
from src.tests.data_processing import test_data_processing
from utils.tokenizers          import tokenizers

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
