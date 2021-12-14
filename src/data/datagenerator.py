# from google.colab import drive, output
# drive.mount('/content/gdrive', force_remount=True)
# % cd "gdrive/MyDrive/temporal-summarization"
#
# !pip install import-ipynb -q
# !pip install transformers -q
#
# import import_ipynb
# output.clear()

"""
This file has the purpose of generating the processed datasets,
iterating through several different parameters and using the DataLoader class.

The parameters through which a dataset is generated are:
    * dataset_path:   str,  retrieved through get_dataset_path() function
    * dataset:        list, the names of the datasets
    * tokenizer_name: str,  the name of the tokenizer that will encode the textual data
    * tokenizer_type: str,  the type of the tokenizer
    * binary:         bool, if True, the ground truth labels will be binary
    * context:        bool, if True, the context is added during the embedding phase
    * only_relevant:  bool  if True, only the relevant updates are retrieved
"""

from src.data.connector import load_data, get_dataset_path

MAX_NUM_WORDS = 512

def generate_csv(
    dataset_path:   str,
    dataset:        list,
    tokenizer_name: str,
    tokenizer_type: str,
    binary:         bool,
    context:        bool,
    only_relevant:  bool
):
    """
    Generates a .csv file containing the dataset by means of the input parameters:
        * dataset_path:   str,  retrieved through get_dataset_path() function
        * dataset:        list, the names of the datasets
        * tokenizer_name: str,  the name of the tokenizer that will encode the textual data
        * tokenizer_type: str,  the type of the tokenizer
        * binary:         bool, if True, the ground truth labels will be binary
        * context:        bool, if True, the context is added during the embedding phase
        * only_relevant:  bool  if True, only the relevant updates are retrieved
    """
    data = load_data(dataset, only_relevant, tokenizer_name, tokenizer_type, MAX_NUM_WORDS, binary, context, test = True)
    data.to_csv(dataset_path, index = False)
    return


def generate():
  dataset = ["2013", "2014", "2015"]

  tokenizers_params = {
      "bert-uncased": {
          "tokenizer": "bert",
          "tokenizer_type": "uncased"
      },
      "bert-cased": {
          "tokenizer": "bert",
          "tokenizer_type": "cased"
      },
      "spanbert-cased": {
          "tokenizer": "spanbert",
          "tokenizer_type": "cased"
      },
      "roberta-uncased": {
          "tokenizer": "roberta",
          "tokenizer_type": "uncased"
      },
      "roberta-cased": {
          "tokenizer": "roberta",
          "tokenizer_type": "cased"
      }
  }

  binary             = [True, False]
  context            = [True, False]
  only_relevant_data = [True, False]

  n_datasets = len(tokenizers_params) * len(binary) * len(context) * len(only_relevant_data)
  idx = 1

  for key in tokenizers_params:
      value        = tokenizers_params[key]
      tokenizer      = value["tokenizer"]
      tokenizer_type = value["tokenizer_type"]
      for binary_classes in binary:
          for contextual in context:
              for only_relevant in only_relevant_data:

                  path = get_dataset_path(dataset, tokenizer, tokenizer_type, binary_classes, contextual, only_relevant)

                  print("* * * * * * * * * * * * * * * * * * * * * * * *")
                  print()
                  print(f"({idx}/{n_datasets})")
                  print()
                  print("Generating .csv dataset for params:")
                  print(f"Dataset:        {dataset}")
                  print(f"Encoder:        {tokenizer}-{tokenizer_type}")
                  print(f"Binary classes: {binary_classes}")
                  print(f"Context:        {contextual}")
                  print(f"Only relevant:  {only_relevant}")
                  print()
                  print("- - - - - - - - - - - - - - - - - - - - - - - -")
                  print()
                  print(f"Dataset path:   {path}")
                  print()
                  print("- - - - - - - - - - - - - - - - - - - - - - - -")
                  generate_csv(path, dataset, tokenizer, tokenizer_type, binary_classes, contextual, only_relevant)
                  print()
                  print("Done!")
                  print()
                  print("* * * * * * * * * * * * * * * * * * * * * * * *")
                  print()

                  idx += 1

  return

# generate()