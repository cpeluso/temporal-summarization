import wandb
from google.colab import output

output.clear()

from src.data.datamanager import *
from src.model.trainer    import *
from utils.cuda_utils     import clean_torch_memory

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
        "encoder": "bert",
        "tokenizer_type": "uncased"
    },
    "bert-cased": {
        "backbone": "BertForTokenClassification",
        "model_name": "bert-large-cased",
        "encoder": "bert",
        "tokenizer_type": "cased"
    },
    "spanbert-cased": {
        "backbone": "SpanBertForTokenClassification",
        "model_name": "SpanBERT/spanbert-large-cased",
        "encoder": "spanbert",
        "tokenizer_type": "cased"
    },
    "roberta-uncased": {
        "backbone": "RobertaForTokenClassification",
        "model_name": "roberta-large",
        "encoder": "roberta",
        "tokenizer_type": "uncased"
    },
    "roberta-cased": {
        "backbone": "RobertaForTokenClassification",
        "model_name": "roberta-large",
        "encoder": "roberta",
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

def __is_timestamped(args):
  try:
    key = args["timestamp"]
  except:
    key = str(input("Should I add timestamps to the data? (y/n) "))

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

def __get_saving_path(CONTINUOUS_EVALUATION, model, N_CLASSES, CONTEXTUAL, TIMESTAMPED):
  if CONTINUOUS_EVALUATION:
    base_path = f"models/{model}/{N_CLASSES}_classes/context_{CONTEXTUAL}/timestamp_{TIMESTAMPED}/"
    path      = base_path + "model.pth"
    try:
      os.makedirs(base_path, mode = 0o666)
    except OSError as e:
      if e.errno == errno.EEXIST:
          print('Directory not created because it already exists.')
      else:
          raise
  else:
    path = None
  
  return path

def __get_default_params():

  default_params = dict(
    IS_BACKBONE_TRAINED = True,
    LEARNING_RATE       = 1e-5,
    EPOCHS              = 3,
    MAX_GRAD_NORM       = 10,
    N_LOGGING_STEPS     = 100,
    MAX_LEN             = 512,
    TRAIN_SIZE          = 0.8,
    TEST_SIZE           = 0.5, # of TRAIN_SIZE
    TRAIN_BATCH_SIZE    = 2,
    VALID_BATCH_SIZE    = 1,
    TEST_BATCH_SIZE     = 1,
    TRAIN_SHUFFLE       = True,
    VALID_SHUFFLE       = False,
    TEST_SHUFFLE        = False,
    NUM_WORKERS         = 0
  )

  return default_params

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

def __get_wandb_name(model, BINARY_TOKEN_CLASSIFICATION, CONTEXTUAL, TIMESTAMPED):

  name = model

  binary      = "binary"      if BINARY_TOKEN_CLASSIFICATION else "not-binary"
  contextual  = "contextual"  if CONTEXTUAL                  else "not-contextual"
  timestamped = "timestamped" if TIMESTAMPED                 else "not-timestamped"

  name = name + "_" + binary + "_" + contextual + "_" + timestamped

  return name

def setup(*args):

    args = args[0]

    device = __warmup(seed=42)

    model = __get_model(args)

    BACKBONE       = model_params[model]["backbone"]
    MODEL_NAME     = model_params[model]["model_name"]
    ENCODER        = model_params[model]["encoder"]
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

    TIMESTAMPED = __is_timestamped(args)

    print()
    print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")
    print()
    print("We're almost done! Just few more questions...")
    print()

    SPLIT_DATASETS_ON_TOPICS = __is_splitted_on_topics(args)

    print()
    print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")
    print()
    print("Ok, as you want.")
    print()

    CONTINUOUS_EVALUATION = __is_continuous_evaluation(args)
    MODEL_SAVE_PATH       = __get_saving_path(CONTINUOUS_EVALUATION, model, N_CLASSES, CONTEXTUAL, TIMESTAMPED)
    
    print()
    print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")
    print()
    print("Cool! We're done!")
    print()

    output.clear()

    default_params = __get_default_params()

    train_params, valid_params, test_params = __get_dataloaders_params(default_params)

    datamanager = DataManager()
    data = datamanager.get_dataset(DATASETS, ENCODER, TOKENIZER_TYPE, BINARY_TOKEN_CLASSIFICATION, CONTEXTUAL, TIMESTAMPED)
    train_data, valid_data, test_data = datamanager.split_datasets(data, default_params['TRAIN_SIZE'], default_params['TEST_SIZE'], SPLIT_DATASETS_ON_TOPICS)
    training_set, validation_set, testing_set = datamanager.load_torch_datasets(train_data, valid_data, test_data)
    training_loader, validation_loader, testing_loader = datamanager.load_torch_dataloaders(training_set, validation_set, testing_set, train_params, valid_params, test_params)

    name  = __get_wandb_name(model, BINARY_TOKEN_CLASSIFICATION, CONTEXTUAL, TIMESTAMPED)
    group = model
    
    config = dict(
      n_classes           = N_CLASSES,
      datasets            = DATASETS,
      encoder             = ENCODER,
      max_len             = default_params['MAX_LEN'],
      train_size          = default_params['TRAIN_SIZE'],
      train_batch_size    = default_params['TRAIN_BATCH_SIZE'],
      valid_batch_size    = default_params['VALID_BATCH_SIZE'],
      learning_rate       = default_params['LEARNING_RATE'],
      train_shuffle       = default_params['TRAIN_SHUFFLE'],
      valid_shuffle       = default_params['VALID_SHUFFLE'],
      num_workers         = default_params['NUM_WORKERS'],
      epochs              = default_params['EPOCHS'],
      backbone            = BACKBONE,
      is_backbone_trained = default_params['IS_BACKBONE_TRAINED']
    )

    wandb.init(group=group, project="temporal-summarization", entity="cpeluso", name=name, config=config)

    if device == "cuda":
        clean_torch_memory()

    if BACKBONE == "BertForTokenClassification" or BACKBONE == "SpanBertForTokenClassification":
        model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=N_CLASSES)

    if BACKBONE == "RobertaForTokenClassification":
        model = RobertaForTokenClassification.from_pretrained(MODEL_NAME, num_labels=N_CLASSES)

    model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=default_params['LEARNING_RATE'])

    trainer = BertTrainer(
        model,
        device, 
        optimizer,
        default_params['EPOCHS'], 
        default_params['N_LOGGING_STEPS'],
        default_params['MAX_GRAD_NORM'],
        ENCODER, 
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