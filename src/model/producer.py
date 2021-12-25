import torch
import re
import math
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from sentence_transformers import SentenceTransformer, util

class Producer:

    def __init__(self, base_threshold = 0.5):
        
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.sentence_transfomer = SentenceTransformer('all-MiniLM-L6-v2')

        pegasus_model_name = 'tuner007/pegasus_paraphrase'
        
        self.pegasus_tokenizer = PegasusTokenizer.from_pretrained(pegasus_model_name)
        
        self.pegasus = PegasusForConditionalGeneration.from_pretrained(pegasus_model_name).to(self.torch_device)

        self.num_return_sequences = 10
        self.num_beams            = 10

        self.summary_sentences = []
        self.base_threshold    = base_threshold
        self.threshold_decay   = 0.005
        self.threshold         = base_threshold

        def random_state(seed):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        random_state(42)

        self.counter = 1
        pass


    def __update_threshold(self):
        self.threshold = self.base_threshold * math.exp(-self.threshold_decay * len(self.summary_sentences))


    def __compute_cosine_similarities(self, references, candidate):

        embeddings1 = self.sentence_transfomer.encode(references, convert_to_tensor = True)
        embeddings2 = self.sentence_transfomer.encode(candidate,  convert_to_tensor = True)

        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

        return cosine_scores


    def __get_overall_cosine_similarity(self, candidate):

        return max(self.__compute_cosine_similarities(self.summary_sentences, candidate).tolist())[0]


    def __select_most_representative_paraphrased_sentence(self, paraphrased_texts, original_text):

        cosine_scores = self.__compute_cosine_similarities(paraphrased_texts, original_text)
        f = lambda i: cosine_scores[i]
        idx_sentence = max(range(len(cosine_scores)), key=f)

        return paraphrased_texts[idx_sentence]


    def __paraphrase(self, input_text):

        batch = self.pegasus_tokenizer([input_text],
                          truncation=True,
                          padding='longest',
                          max_length=60, 
                          return_tensors="pt").to(self.torch_device)

        translated = self.pegasus.generate(**batch,
                                           max_length=60,
                                           num_beams=self.num_beams, 
                                           num_return_sequences=self.num_return_sequences)
        
        paraphrased_texts = self.pegasus_tokenizer.batch_decode(translated, 
                                                       skip_special_tokens=True)
        
        return self.__select_most_representative_paraphrased_sentence(paraphrased_texts, input_text)

    
    def __emit(self, paraphrased_candidate):
        print(f"{len(self.summary_sentences)}\t{paraphrased_candidate}")
        self.summary_sentences.append(paraphrased_candidate)
        return

    def update_summary(self, candidate: list):

        self.__update_threshold()

        candidate_str = candidate[0]
        candidate_str = clean_sentence(candidate_str)

        paraphrased_candidate = self.__paraphrase(candidate_str)

        if self.counter == 1:

            self.__emit(paraphrased_candidate)

        if self.__get_overall_cosine_similarity(candidate_str) < self.threshold:

            self.__emit(paraphrased_candidate)

        self.counter += 1
        
        return

def clean_sentence(candidate: str):

    candidate = candidate.replace("(s)", "")
    candidate = candidate.replace("(e)", "")
    candidate = candidate.replace("(S)", "")
    candidate = candidate.replace("(E)", "")
    candidate = re.sub('\s+', ' ', candidate)

    return candidate