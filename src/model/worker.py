import wandb
from google.colab import output

output.clear()

from src.data.datamanager import *
from src.model.trainer    import *
from utils.cuda_utils     import clean_torch_memory

from src.model.training_params import default_params

import torch
import random
import os
import errno
import numpy as np

from torch        import cuda
from transformers import RobertaForTokenClassification, BertForTokenClassification

model_params = {
    "bert-uncased": {
        "backbone": "BertForTokenClassification",
        "model_name": "bert-large-uncased",
        "tokenizer": "bert",
        "tokenizer_type": "uncased"
    },
    "bert-cased": {
        "backbone": "BertForTokenClassification",
        "model_name": "bert-large-cased",
        "tokenizer": "bert",
        "tokenizer_type": "cased"
    },
    "spanbert-cased": {
        "backbone": "SpanBertForTokenClassification",
        "model_name": "SpanBERT/spanbert-large-cased",
        "tokenizer": "spanbert",
        "tokenizer_type": "cased"
    },
    "roberta-uncased": {
        "backbone": "RobertaForTokenClassification",
        "model_name": "roberta-large",
        "tokenizer": "roberta",
        "tokenizer_type": "uncased"
    },
    "roberta-cased": {
        "backbone": "RobertaForTokenClassification",
        "model_name": "roberta-large",
        "tokenizer": "roberta",
        "tokenizer_type": "cased"
    }
}

datasets = {
    "1": ["2013"],
    "2": ["2014"],
    "3": ["2015"],
    "4": ["2013", "2014"],
    "5": ["2013", "2015"],
    "6": ["2014", "2015"],
    "7": ["2013", "2014", "2015"]
}

def __str_to_bool(key):
  if key not in ["y", "n"]:
    print()
    print(f"Aborting. Selected key {key} is not recognized.")

  return True if key == "y" else False

def __warmup(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True
  output.clear()
  device = 'cuda' if cuda.is_available() else 'cpu'
  return device

def __get_model(args):
  try:
    model = args["model"]
  except:
    print("Which model do you want to train?")
    print("> options:")
    print(list(model_params.keys()))
    print()
    model = str(input("Choose a model: "))

  if model not in list(model_params.keys()):
    print()
    print(f"Aborting. Selected model {model} is not recognized.")

  return model

def __get_dataset(args):
  try:
    key = args["datasets"]
  except:
    print(f"On which datasets do you want to train {model}-{TOKENIZER_TYPE}?")
    print("> options:")
    print(datasets)
    print()
    key = str(input("Choose a dataset: "))

  if key not in list(datasets.keys()):
    print()
    print(f"Aborting. Selected dataset is not recognized.")

  return datasets[key]

def __is_full_dataset(args):
  try:
    key = args["full_dataset"]
  except:
    key = str(input("Should I load the full dataset (with not-relevant texts)? (y/n) "))

  return __str_to_bool(key) 

def __is_binary(args):
  try:
    key = args['binary']
  except:
    key = str(input("Should I use binary masks? (y/n) "))

  return __str_to_bool(key)

def __is_contextual(args):
  try:
    key = args["context"]  
  except:
    key = str(input("Should I add context to the data? (y/n) "))
  
  return __str_to_bool(key)

def __is_splitted_on_topics(args):
  try:
    key = args["split_on_topics"]
  except:
    key = str(input("Should I split the data considering their topics? (y/n) "))

  return __str_to_bool(key)  

def __is_continuous_evaluation(args):
  try:
    key = args["save_model"]
  except:
    key = str(input("Should I save the best model? (y/n) "))

  return __str_to_bool(key) 

def __get_learning_rate(args):
  try:
    learning_rate = args["learning_rate"]
  except:
    learning_rate = default_params["LEARNING_RATE"]

  return learning_rate

def __get_saving_path(model, N_CLASSES, CONTEXTUAL, ONLY_RELEVANT):
  base_path = f"models/{model}/{N_CLASSES}_classes/context_{CONTEXTUAL}/only-relevant_{ONLY_RELEVANT}/"
  path      = base_path + "model.pth"
  try:
    os.makedirs(base_path, mode = 0o666)
  except OSError as e:
    if e.errno == errno.EEXIST:
        print('Directory not created because it already exists.')
    else:
        raise
  
  return path

def __get_dataloaders_params(default_params):
  train_params = {
    'batch_size':  default_params['TRAIN_BATCH_SIZE'],
    'shuffle':     default_params['TRAIN_SHUFFLE'],
    'num_workers': default_params['NUM_WORKERS']
  }

  valid_params = {
    'batch_size':  default_params['VALID_BATCH_SIZE'],
    'shuffle':     default_params['VALID_SHUFFLE'],
    'num_workers': default_params['NUM_WORKERS']
  }

  test_params = {
    'batch_size':  default_params['TEST_BATCH_SIZE'],
    'shuffle':     default_params['TEST_SHUFFLE'],
    'num_workers': default_params['NUM_WORKERS']
  }

  return train_params, valid_params, test_params

def __get_wandb_name(model, BINARY_TOKEN_CLASSIFICATION, CONTEXTUAL, ONLY_RELEVANT):

  name = model

  binary        = "binary"        if BINARY_TOKEN_CLASSIFICATION else "not-binary"
  contextual    = "contextual"    if CONTEXTUAL                  else "not-contextual"
  only_relevant = "only-relevant" if ONLY_RELEVANT               else "full"

  name = name + "_" + binary + "_" + contextual + "_" + only_relevant

  return name

def setup(*args):

    args = args[0]

    device = __warmup(seed=42)

    model = __get_model(args)

    BACKBONE       = model_params[model]["backbone"]
    MODEL_NAME     = model_params[model]["model_name"]
    TOKENIZER_NAME = model_params[model]["tokenizer"]
    TOKENIZER_TYPE = model_params[model]["tokenizer_type"]

    print()
    print("Great.")
    print()

    DATASETS = __get_dataset(args)

    print()
    print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")
    print()
    print(f"Very good! Your model will be trained on {DATASETS} data.")
    print()

    ONLY_RELEVANT = not __is_full_dataset(args)

    print()
    print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")
    print()
    print("Ok, seems legit!")
    print()

    BINARY_TOKEN_CLASSIFICATION = __is_binary(args)
    N_CLASSES                   = 2 if BINARY_TOKEN_CLASSIFICATION else 5

    print()
    print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")
    print()
    print("Ok, that's fine.")
    print()

    CONTEXTUAL = __is_contextual(args)

    print()
    print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")
    print()
    print("Fair enough.")
    print()

    SPLIT_DATASETS_ON_TOPICS = __is_splitted_on_topics(args)

    print()
    print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")
    print()
    print("Ok, as you want.")
    print()

    CONTINUOUS_EVALUATION = __is_continuous_evaluation(args)
    MODEL_SAVE_PATH       = __get_saving_path(model, N_CLASSES, CONTEXTUAL, ONLY_RELEVANT)
    
    print()
    print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")
    print()
    print("Cool! We're done!")
    print()

    learning_rate = __get_learning_rate(args)

    try:
      load_pretrained_model = args['load_pretrained_model']
    except:
      load_pretrained_model = False

    output.clear()

    train_params, valid_params, test_params = __get_dataloaders_params(default_params)

    datamanager = DataManager(DATASETS, 
                              ONLY_RELEVANT, 
                              TOKENIZER_NAME, 
                              TOKENIZER_TYPE, 
                              BINARY_TOKEN_CLASSIFICATION, 
                              CONTEXTUAL, 
                              default_params['TRAIN_SIZE'], 
                              default_params['TEST_SIZE'], 
                              SPLIT_DATASETS_ON_TOPICS)
    
    data = datamanager.get_dataset()
    
    train_data, valid_data, test_data = datamanager.split_datasets(data)
    
    torch_datasets = datamanager.load_torch_datasets(train_data, 
                                                     valid_data, 
                                                     test_data)
    
    training_set, validation_set, testing_set = torch_datasets 
    
    torch_dataloaders = datamanager.load_torch_dataloaders(training_set, 
                                                           validation_set, 
                                                           testing_set, 
                                                           train_params, 
                                                           valid_params, 
                                                           test_params)
    
    training_loader, validation_loader, testing_loader = torch_dataloaders

    if not load_pretrained_model:

        name  = __get_wandb_name(model, 
                                BINARY_TOKEN_CLASSIFICATION, 
                                CONTEXTUAL, 
                                ONLY_RELEVANT)
        group = model
        
        config = dict(
          n_classes        = N_CLASSES,
          datasets         = DATASETS,
          tokenizer_name   = TOKENIZER_NAME,
          tokenizer_type   = TOKENIZER_TYPE,
          contextual       = CONTEXTUAL,
          only_relevant    = ONLY_RELEVANT,
          train_size       = default_params['TRAIN_SIZE'],
          train_batch_size = default_params['TRAIN_BATCH_SIZE'],
          valid_batch_size = default_params['VALID_BATCH_SIZE'],
          learning_rate    = learning_rate,
          train_shuffle    = default_params['TRAIN_SHUFFLE'],
          valid_shuffle    = default_params['VALID_SHUFFLE'],
          epochs           = default_params['EPOCHS']
        )

        wandb.init(group=group, project="temporal-summarization", entity="cpeluso", name=name, config=config)

    if device == "cuda":
        clean_torch_memory()

    if BACKBONE == "BertForTokenClassification" or BACKBONE == "SpanBertForTokenClassification":
        model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=N_CLASSES)

    if BACKBONE == "RobertaForTokenClassification":
        model = RobertaForTokenClassification.from_pretrained(MODEL_NAME, num_labels=N_CLASSES)

    model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

    trainer = BertTrainer(
        load_pretrained_model,
        model,
        device, 
        optimizer,
        default_params['EPOCHS'], 
        default_params['N_LOGGING_STEPS'],
        default_params['MAX_GRAD_NORM'],
        TOKENIZER_NAME, 
        TOKENIZER_TYPE,
        default_params['MAX_LEN'],
        training_loader,
        validation_loader,
        testing_loader,
        default_params['TRAIN_BATCH_SIZE'],
        default_params['VALID_BATCH_SIZE'],
        default_params['TEST_BATCH_SIZE'],
        CONTINUOUS_EVALUATION,
        MODEL_SAVE_PATH
    )

    return trainer