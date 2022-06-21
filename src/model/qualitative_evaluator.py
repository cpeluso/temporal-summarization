class QualitativeEvaluator:
    """
    QualitativeEvaluator class.

    The QualitativeEvaluator class has the purpose of qualitatively evaluating the predictions of the model
    and compare them with the ground truth.

    Attributes
    ----------
    tokenizer : AutoTokenizer
        The tokenizer used during the data generation phase.
    max_num_words : int
        An integer representing the max number of words
        that will be contained in a single row of the dataframe returned.
    num_labels : int
        An integer representing the number of distinct labels appearing in every label / mask.
        Supported values are 2 and 5.
        If 2, the labels referred to the data will be binary (eg. [0,1,1,1,1...,0,0,1,1,1,0]).
        If 5, the labels referred to the data will be not-binary (eg. [0,1,2,2,2,3,0,1,2,2,2,2,2,3,0,0,0,4,0,0]).

    Methods
    -------
    evaluate_batch(batch, predictions):
        Prints the text associated to the ground truth and
        prints the text associated to the predictions.
    """

    def __init__(
        self,
        tokenizer,
        max_num_words,
        num_labels
    ):
        self.tokenizer     = tokenizer
        self.max_num_words = max_num_words
        self.num_labels    = num_labels
        pass

    def evaluate_batch(self, batch, predictions, verbose = True):
        if verbose:
          print("***** EVALUATION *****")
        real_spans, predicted_spans = self.__evaluate(batch['input_ids'], batch['labels'], predictions, verbose)
        
        if verbose:
          print("**********************")
        return real_spans, predicted_spans

    def __evaluate(self, input_ids, labels, predictions, verbose):

        masks = []
        preds = []

        real_spans      = []
        predicted_spans = []
        
        for mask, pred in zip(labels, predictions):
          masks.append(mask[0].tolist())
          preds.append(pred.tolist())

        decoded_texts = []

        for ids in input_ids:
          decoded_text_str = self.__decode(ids.tolist())
          decoded_text     = self.__pad_text(decoded_text_str)
          decoded_texts.append(decoded_text)

        for text, mask, predicted_mask in zip(decoded_texts, masks, preds):
          ground = ""
          prediction = ""
          
          if verbose:
            print()

          for word, visible, predicted in zip(text, mask, predicted_mask):

            if self.num_labels == 5:
              if int(visible) == 1:
                ground = ground + " (s) " + word + " "

              if int(visible) == 2:
                ground = ground + word + " "

              if int(visible) == 3:
                ground = ground + word + " (e) "

              if int(visible) == 4:
                ground = ground + " (S) " + word + " (E) "

              if int(predicted) == 1:
                prediction = prediction + " (s) " + word + " "

              if int(predicted) == 2:
                prediction = prediction + word + " "

              if int(predicted) == 3:
                prediction = prediction + word + " (e) "

              if int(predicted) == 4:
                prediction = prediction + " (S) " + word + " (E) "

            if self.num_labels == 2:

              if int(visible) == 1:
                ground = ground + word + " "

              if int(predicted) == 1:
                prediction = prediction + word + " "

          if verbose:
            print(f"(R) => {ground}")
            print(f"(P) => {prediction}")

          real_spans.append(ground)
          predicted_spans.append(prediction)
        
        if verbose:
          print()

        return real_spans, predicted_spans

    def __decode(self, input_ids):
        decoded_str_text = self.tokenizer.decode(input_ids)

        decoded_str_text = decoded_str_text.replace("[CLS]", "")
        decoded_str_text = decoded_str_text.replace("[SEP]", "")
        decoded_str_text = decoded_str_text.replace("[PAD]", "")
        decoded_str_text = decoded_str_text.replace("<s>", "")
        decoded_str_text = decoded_str_text.replace("</s>", "")
        decoded_str_text = decoded_str_text.replace("<pad>", "")

        return decoded_str_text

    def __pad_text(self, text: str):
        text = text.split()

        pad = [" "] * (self.max_num_words - len(text))
        text.extend(pad)

        return text