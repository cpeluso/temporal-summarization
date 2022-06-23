import pandas as pd
import torch 
import os
from torch.utils.data import DataLoader as TorchDataLoader
from transformers import BertForTokenClassification, AutoTokenizer
from datetime import timezone
import datetime
from os.path import exists

from src.model.qualitative_evaluator import QualitativeEvaluator
from trec.eval import evaluation_script
from src.data.dataloader import filenames
from src.data.s3_connector import download_file
from src.data.datamanager import EvaluationDataset
from src.model.producer import Producer

def make_ids(row: pd.Series):

    update_id = row["update_id"]

    return update_id.split("-")[0] + "-" + update_id.split("-")[1], int(update_id.split("-")[2])

class TrecEvaluator:

    def __init__(
        self,
        model_path   = "models/bert-uncased/2_classes/context_False/only-relevant_True/model.pth",
        dataset_path = "2013_2014_2015_bert-uncased_binary_not-contextual_only_relevant.csv",
        base_run_filename = ".txt"
    ):
        self.nuggets_filename = "data/nuggets.tsv"
        self.updates_filename = "data/updates.tsv"
        self.matches_filename = "data/matches.tsv"

        if not exists("models/bert-uncased/2_classes/context_False/only-relevant_True"):
            os.makedirs("models/bert-uncased/2_classes/context_False/only-relevant_True")

        if not exists("data"):
            os.makedirs("data")

        if not exists("results"):
            os.makedirs("results")

        if not exists(self.nuggets_filename):
            download_file(self.nuggets_filename)

        if not exists(self.updates_filename):
            download_file(self.updates_filename)

        if not exists(self.matches_filename):
            download_file(self.matches_filename)

        if not exists(dataset_path):
          download_file(dataset_path)
        
        if not exists(model_path):
          download_file(model_path)

        self.dataset = pd.read_csv(dataset_path)

        self.dataset[['doc_id', 'sent_id']] = self.dataset.apply(make_ids, axis=1, result_type="expand")
        self.dataset['int_doc_id'] = self.dataset.doc_id.astype('category').cat.codes
        self.doc_id_mapping = self.dataset[['int_doc_id', 'doc_id']].set_index('int_doc_id').to_dict()['doc_id']

        self.query_ids = set(self.dataset.query_id.unique())

        self.model = BertForTokenClassification.from_pretrained("bert-large-uncased", num_labels=2)
        self.model.load_state_dict(torch.load(model_path))  

        self.device = 'cuda'

        self.model.to(self.device)

        self.qualitative_evaluator = QualitativeEvaluator(
            AutoTokenizer.from_pretrained("bert-large-uncased", truncation = True, do_lower_case = True),
            512,
            2
        )

        self.base_run_filename = base_run_filename

        return

    def evaluate_all(
        self
    ):

        runs_filename = []
        for query_id in self.query_ids:

            run_filename = "./results/run-" + str(query_id) +  self.base_run_filename 

            if exists(run_filename):
              continue
                        
            self.make_run(int(query_id))
            print(" - - - - - - - - - - - ")
            
            runs_filename.append(run_filename)

        print()
        print("************************")
        print()
        self.evaluate_run(runs_filename)

        return

    def evaluate(
        self,
        query_id,
    ):
      runs_filename = "./results/run-" + str(query_id) +  self.base_run_filename 

      self.make_run(query_id)
      print()
      print("************************")
      print()
      self.evaluate_run([runs_filename])

      return 

    def make_runs(
        self
    ) -> list:

        runs_filename = []
        for query_id in self.query_ids:

            run_filename = "./results/run-" + str(query_id) +  self.base_run_filename 

            if exists(run_filename):
              open(run_filename, "w").close()
                        
            self.make_run(int(query_id))
            print(" - - - - - - - - - - - ")
            
            runs_filename.append(run_filename)

        return runs_filename

    def make_run(
        self,
        query_id
    ):
      test_dataloader = self.__get_test_dataloader(query_id)
      self.run(test_dataloader, query_id)

      return

    def evaluate_run(
        self,
        runs_filename: list
    ):

      evaluation_script(
          self.nuggets_filename,
          self.updates_filename,
          runs_filename,
          self.matches_filename,
          None,
          args_debug     = False,
          args_binaryrel = True
      )
      return
        
    def run(
        self,
        dataloader, 
        query_id
    ):

        producer = Producer()

        with torch.no_grad():

            self.model.eval()
            
            for _, batch in enumerate(dataloader):
                
                int_doc_id, sent_id, input_ids, attention_mask, labels = self.__unpack_batch(batch)

                logits = self.__forward(input_ids, attention_mask, labels)

                flattened_predictions = self.__flatten(logits)
            
                _, predicted_spans = self.qualitative_evaluator.evaluate_batch(batch, flattened_predictions, verbose = False)
                
                emitted = producer.update_summary(predicted_spans, verbose=False)  

                if emitted:
                    self.__update_run_file(
                        query_id           = query_id,
                        team_id            = 0,
                        run_id             = 0,
                        document_id        = self.doc_id_mapping[int_doc_id],
                        sentence_id        = sent_id,
                        decision_timestamp = int(datetime.datetime.now(timezone.utc).replace(tzinfo=timezone.utc).timestamp()),
                        confidence_value   = 1000
                    )

    def __get_test_dataloader(self, query_id: int) -> TorchDataLoader:

        assert type(query_id) == int
        
        test_params = {
            'batch_size':  1,
            'shuffle':     False,
            'num_workers': 0
        }

        test_data = self.dataset[self.dataset.query_id == query_id].reset_index(drop=True).sort_values(by="timestamp")
        testing_set = EvaluationDataset(test_data[['input_ids', 'attention_mask']], list(test_data['int_doc_id']), list(test_data['sent_id']),  test_data['mask'])

        return TorchDataLoader(testing_set, **test_params)

    def __unpack_batch(self, batch):

        int_doc_id     = batch['int_doc_id'].item()
        sent_id        = batch['sent_id'].item()       
        input_ids      = batch['input_ids']     .to(self.device, dtype = torch.long)
        attention_mask = batch['attention_mask'].to(self.device, dtype = torch.long)
        labels         = batch['labels']        .to(self.device, dtype = torch.long)

        return int_doc_id, sent_id, input_ids, attention_mask, labels

    def __forward(self, input_ids, attention_mask, labels):

        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits

    def __flatten(self, logits):

        active_logits = logits.view(-1, self.model.num_labels) 
        flattened_predictions = torch.argmax(active_logits, axis=1) 
        flattened_predictions = torch.tensor_split(flattened_predictions, 1)

        return flattened_predictions

    # Query_ID Team_ID Run_ID Document_ID Sentence_ID Decision_Timestamp Confidence_Value
    def __update_run_file(
        self,
        query_id,
        team_id,
        run_id,
        document_id,
        sentence_id,
        decision_timestamp,
        confidence_value
    ):
        run_filename = "./results/run-" + str(query_id) + self.base_run_filename 
        
        with open(run_filename, "a") as run_fp:
            print(f"{query_id} {team_id} {run_id} {document_id} {sentence_id} {decision_timestamp} {confidence_value}")
            print()
            run_fp.write(f"{query_id} {team_id} {run_id} {document_id} {sentence_id} {decision_timestamp} {confidence_value}\n")
        
        return