import torch
import wandb
import sys
import math
from sklearn.metrics import accuracy_score

from src.model.evaluator import *
from utils.tokenizers import tokenizers
from src.data.s3_connector import download_file, upload_file
from src.model.producer import Producer

from transformers import AutoTokenizer

class BertTrainer:

    def __init__(
        self,
        load_pretrained_model,
        model,
        device, 
        optimizer,
        epochs, 
        n_logging_steps,
        max_grad_norm,
        
        tokenizer, 
        tokenizer_type,
        max_num_words,
        
        training_loader,
        validation_loader,
        testing_loader,

        train_batch_size,
        valid_batch_size,
        test_batch_size,

        continuous_evaluation: bool = False,
        model_save_path:       str  = None
    ):

      self.model = model

      if load_pretrained_model:
        download_file(model_save_path)
        self.model.load_state_dict(torch.load(model_save_path))     

      self.device          = device
      self.optimizer       = optimizer
      self.epochs          = epochs
      self.n_logging_steps = n_logging_steps
      self.max_grad_norm   = max_grad_norm

      do_lower_case = True if tokenizer_type == "uncased" else False

      self.tokenizer     = AutoTokenizer.from_pretrained(tokenizers[tokenizer][tokenizer_type], truncation = True, do_lower_case = do_lower_case)
      self.max_num_words = max_num_words
      
      self.training_loader   = training_loader
      self.validation_loader = validation_loader
      self.testing_loader    = testing_loader

      self.train_batch_size = train_batch_size
      self.valid_batch_size = valid_batch_size
      self.test_batch_size  = test_batch_size

      self.continuous_evaluation    = continuous_evaluation
      self.model_save_path          = model_save_path
      self.best_validation_loss     = 1_000_000
      self.best_validation_accuracy = 0

      self.evaluator = Evaluator(self.tokenizer, max_num_words, self.model.num_labels)
      pass

    def __unpack_batch(self, batch):
      input_ids      = batch['input_ids']     .to(self.device, dtype = torch.long)
      attention_mask = batch['attention_mask'].to(self.device, dtype = torch.long)
      labels         = batch['labels']        .to(self.device, dtype = torch.long)
      return input_ids, attention_mask, labels

    def __forward(self, input_ids, attention_mask, labels):

      outputs   = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
      loss      = outputs.loss
      tr_logits = outputs.logits
      return loss, tr_logits

    def __compute_accuracy(self, labels, tr_logits, tr_accuracy):
      # compute training accuracy
      flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
      active_logits = tr_logits.view(-1, self.model.num_labels) # shape (batch_size * seq_len, num_labels)
      flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
      
      # only compute accuracy at active labels
      active_accuracy = labels.view(-1) != 0 # shape (batch_size, seq_len)
      #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
      
      labels      = torch.masked_select(flattened_targets, active_accuracy)
      predictions = torch.masked_select(flattened_predictions, active_accuracy)
      
      if math.isnan(tr_accuracy):
        tr_accuracy = 0

      tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())

      if math.isnan(tmp_tr_accuracy):
        tmp_tr_accuracy = 0

      tr_accuracy += tmp_tr_accuracy

      flattened_targets     = torch.tensor_split(flattened_targets,     self.train_batch_size)
      flattened_predictions = torch.tensor_split(flattened_predictions, self.train_batch_size)

      return tr_accuracy, flattened_predictions

    def __gradient_clipping(self):
      torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_grad_norm)

    def __backward(self, loss):
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

    def train(self):

        for epoch in range(self.epochs):

            tr_loss        = 0
            tr_accuracy    = 0
            nb_tr_examples = 0 
            nb_tr_steps    = 0

            self.model.train()
            
            for idx, batch in enumerate(self.training_loader):

                input_ids, attention_mask, labels = self.__unpack_batch(batch)

                loss, tr_logits = self.__forward(input_ids, attention_mask, labels)

                tr_loss += loss.item()

                nb_tr_steps += 1
                nb_tr_examples += labels.size(0)

                tr_accuracy, flattened_predictions = self.__compute_accuracy(labels, tr_logits, tr_accuracy)

                self.__gradient_clipping()

                if idx % self.n_logging_steps == 0:
                    loss_step = tr_loss/nb_tr_steps
                    step_accuracy = tr_accuracy / nb_tr_steps
                    print(f"Training loss per {idx} training steps: {loss_step}")
                    print(f"(logs): tr_accuracy {tr_accuracy}")
                    print(f"(logs): nb_tr_steps {nb_tr_steps}")
                    print(f"(logs): idx         {idx}")
                    print(f"Training accuracy: {step_accuracy}")

                    if "wandb" in sys.modules:
                        wandb.log({"train_loss_step": loss_step})
                        wandb.log({"train_acc_step": step_accuracy})
                
                    self.evaluator.evaluate_batch(batch, flattened_predictions)

                # backward pass
                self.__backward(loss)

            epoch_loss = tr_loss / nb_tr_steps
            tr_accuracy = tr_accuracy / nb_tr_steps
            print(f"Training loss epoch {epoch}: {epoch_loss}")
            print(f"Training accuracy epoch {epoch}: {tr_accuracy}")
            
            if "wandb" in sys.modules:
                wandb.log({"train_loss_epoch": epoch_loss})
                wandb.log({"train_acc_epoch": tr_accuracy})
            
            self.validate(self.validation_loader)

    def validate(self, loader, verbose = True):

        with torch.no_grad():

            tr_loss        = 0
            tr_accuracy    = 0
            nb_tr_examples = 0 
            nb_tr_steps    = 0

            self.model.eval()
            
            for idx, batch in enumerate(loader):
                
                input_ids, attention_mask, labels = self.__unpack_batch(batch)

                loss, tr_logits = self.__forward(input_ids, attention_mask, labels)

                tr_loss += loss.item()

                nb_tr_steps += 1
                nb_tr_examples += labels.size(0)

                tr_accuracy, flattened_predictions = self.__compute_accuracy(labels, tr_logits, tr_accuracy)

                if idx % self.n_logging_steps == 0 and verbose:
                    loss_step = tr_loss/nb_tr_steps
                    step_accuracy = tr_accuracy / nb_tr_steps
                    print(f"Validation loss per {idx} validation steps: {loss_step}")
                    print(f"Validation accuracy: {step_accuracy}")
                    
                    if "wandb" in sys.modules:
                        wandb.log({"valid_loss_step": loss_step})
                        wandb.log({"valid_acc_step": step_accuracy})

                    self.evaluator.evaluate_batch(batch, flattened_predictions)

            epoch_loss = tr_loss / nb_tr_steps
            tr_accuracy = tr_accuracy / nb_tr_steps

            if verbose:
                print(f"Validation loss: {epoch_loss}")
                print(f"Validation accuracy: {tr_accuracy}")
            
            if "wandb" in sys.modules:
                wandb.log({"valid_loss_epoch": epoch_loss})
                wandb.log({"valid_acc_epoch": tr_accuracy})

            if self.continuous_evaluation and tr_accuracy > self.best_validation_accuracy:
                self.best_validation_accuracy = tr_accuracy
                print("Saving the model: just achieved the best performance on validation set!")
                print(f"path: {self.model_save_path}")
                torch.save(self.model.state_dict(), self.model_save_path)
                upload_file(self.model_save_path)
                print(f"Validation accuracy: {tr_accuracy}")
                print(f"Validation loss:     {epoch_loss}")

            return 

    def test(self):

        with torch.no_grad():

            tr_loss        = 0
            tr_accuracy    = 0
            nb_tr_examples = 0 
            nb_tr_steps    = 0

            self.model.eval()
            
            for idx, batch in enumerate(self.testing_loader):
                
                input_ids, attention_mask, labels = self.__unpack_batch(batch)

                loss, tr_logits = self.__forward(input_ids, attention_mask, labels)

                tr_loss += loss.item()

                nb_tr_steps += 1
                nb_tr_examples += labels.size(0)

                tr_accuracy, flattened_predictions = self.__compute_accuracy(labels, tr_logits, tr_accuracy)
            
                if idx % self.n_logging_steps == 0:
                    loss_step = tr_loss/nb_tr_steps
                    step_accuracy = tr_accuracy / nb_tr_steps
                    print(f"Testing loss per {idx} testing steps: {loss_step}")
                    print(f"Testing accuracy: {step_accuracy}")
                    self.evaluator.evaluate_batch(batch, flattened_predictions)

            epoch_loss = tr_loss / nb_tr_steps
            tr_accuracy = tr_accuracy / nb_tr_steps
            print(f"Testing loss: {epoch_loss}")
            print(f"Testing accuracy: {tr_accuracy}")
            
            if "wandb" in sys.modules:
                wandb.log({"valid_loss_epoch": epoch_loss})
                wandb.log({"valid_acc_epoch": tr_accuracy})
    

    def produce_summary(self, dataloader = None):

        if dataloader is None:
            dataloader = self.testing_loader

        producer = Producer()

        with torch.no_grad():

            tr_loss        = 0
            tr_accuracy    = 0
            nb_tr_examples = 0 
            nb_tr_steps    = 0

            self.model.eval()
            
            for _, batch in enumerate(dataloader):
                
                input_ids, attention_mask, labels = self.__unpack_batch(batch)

                loss, tr_logits = self.__forward(input_ids, attention_mask, labels)

                tr_loss += loss.item()

                nb_tr_steps += 1
                nb_tr_examples += labels.size(0)

                tr_accuracy, flattened_predictions = self.__compute_accuracy(labels, tr_logits, tr_accuracy)
            
                real_spans, predicted_spans = self.evaluator.evaluate_batch(batch, flattened_predictions, verbose = False)
                producer.update_summary(real_spans, predicted_spans)

            epoch_loss  = tr_loss / nb_tr_steps
            tr_accuracy = tr_accuracy / nb_tr_steps
            print(f"Loss: {epoch_loss}")
            print(f"Accuracy: {tr_accuracy}")

        producer.debug()