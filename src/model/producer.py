from parrot import Parrot
import torch
import warnings
warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer, util

class Producer:

    def __init__(self, threshold = 0.6):
        self.sentence_transfomer = SentenceTransformer('all-MiniLM-L6-v2')
        self.parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=torch.cuda.is_available())

        self.threshold = threshold

        self.real_summary      = []
        self.predicted_summary = []

        def random_state(seed):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        random_state(42)
        pass

    def update_summary(self, real_spans, predicted_spans):
        self.real_summary     .extend(real_spans)
        self.predicted_summary.extend(predicted_spans)
        pass

    def __get_overall_cosine_similarity(self, summary_sentences, candidate):

        embeddings1 = self.sentence_transfomer.encode(summary_sentences, convert_to_tensor=True)
        embeddings2 = self.sentence_transfomer.encode(candidate,         convert_to_tensor=True)

        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

        return max(cosine_scores.tolist())[0]

    def show_summary(self):

        ### For debugging purposes only
        with open('debug.txt', 'w') as file:
            for item in self.predicted_summary:
                print(item)
                file.write('%s\n' % item)
        
        summary_sentences = []

        self.predicted_summary.sort(key=lambda s: len(s))

        for idx, candidate in enumerate(self.predicted_summary):

            if not idx:
                summary_sentences.append(candidate[0])
            else:
                overall_cos_sim = self.__get_overall_cosine_similarity(summary_sentences, candidate[0])

                if overall_cos_sim < self.threshold:
                    summary_sentences.append(candidate[0])

        summary = ""

        for sentence in summary_sentences:
            summary = summary + sentence + ". "

        para_phrases = self.parrot.augment(input_phrase=summary,
                                    use_gpu=False,
                                    diversity_ranker="levenshtein",
                                    do_diverse=False, 
                                    max_return_phrases = 10, 
                                    max_length=128, 
                                    adequacy_threshold = 0.99, 
                                    fluency_threshold = 0.90)

        for phrase in para_phrases:
            print(phrase)

