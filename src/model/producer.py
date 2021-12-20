from src.data.s3_connector import upload_file
from parrot import Parrot
import torch
import warnings
warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer, util
import re

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

        self.counter = 1
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
                file.write('%s\n' % item)

        upload_file('debug.txt')
        
        # Initialize empty list of sentences.
        # This list will contain the sentences that in future will be
        # paraphrased to produce an update summary.
        summary_sentences = []

        # Initialize empty string.
        # This string is the stringified version of the list summary_sentences.
        summary_sentences_str = ""

        # For each predicted text
        for idx, candidate in enumerate(self.predicted_summary):

            # Clean text
            candidate = clean_sentence(candidate[0])

            # If is first text considered, 
            # just append it to summary_sentences and summary_sentences_str.
            if not idx:
                summary_sentences_str = candidate + ". "
                summary_sentences.append(candidate)
            # Otherwise,
            # compute the overall cosine similarity of the text
            # with the candidate sentences.
            else:
                overall_cosine_similarity = self.__get_overall_cosine_similarity(summary_sentences, candidate)

                # If the overall cosine similarity of the text
                # is lower than a predefined threshold, 
                # add the text to the summary_sentences list 
                # and concatenate it to summary_sentences_str. 
                if overall_cosine_similarity < self.threshold:

                    # If appending candidate to summary_sentences
                    # will produce a summary_sentences_str longer than 400 words,
                    # a partial update summary is produced.
                    #
                    # Then, summary_sentences and summary_sentences_str
                    # are re-initialized with the just produced summary.
                    if len(summary_sentences_str.split()) + len(candidate.split()) >= 400:
                        produced_summary = self.produce(summary_sentences_str)
                        
                        summary_sentences_str = produced_summary + ". "
                        summary_sentences = [produced_summary]
                    # Otherwise,
                    # keep collecting the candidates
                    # appending the candidate to summary_sentences
                    # and concatenating it to summary_sentences_str.
                    else:
                        summary_sentences_str = summary_sentences_str + candidate + ". "
                        summary_sentences.append(candidate)


    def produce(self, candidates: str):

        para_phrases = self.parrot.augment(input_phrase=candidates,
                                    diversity_ranker="levenshtein",
                                    do_diverse=False, 
                                    max_return_phrases = 10, 
                                    max_length=512, 
                                    adequacy_threshold = 0.99, 
                                    fluency_threshold = 0.90)

        print(f"Summary, part {self.counter}:\n")

        for phrase in para_phrases:
            print(phrase)

        print()

        self.counter += 1

        return para_phrases[-1][0]


def clean_sentence(candidate: str):

    candidate = candidate.replace("(s)", "")
    candidate = candidate.replace("(e)", "")
    candidate = candidate.replace("(S)", "")
    candidate = candidate.replace("(E)", "")
    candidate = re.sub('\s+', ' ', candidate)

    return candidate