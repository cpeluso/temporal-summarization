# from google.colab import drive, output
# drive.mount('/content/gdrive', force_remount=True)
# !pip install import-ipynb -q
# !pip install transformers -q
# output.clear()
# import import_ipynb
# % cd "gdrive/MyDrive/temporal-summarization"

from src.features.lifter   import *
from src.features.pipeline import *

from utils.tokenizers import tokenizers, separators

from transformers import AutoTokenizer

import pandas as pd
import csv

pd.options.mode.chained_assignment = None

T13_topics_filename  = './data/raw/T13/trec2013-ts-topics-test.xml'
T13_matches_filename = './data/raw/T13/matches.tsv'
T13_nuggets_filename = './data/raw/T13/nuggets.tsv'
T13_updates_filename = './data/raw/T13/updates.tsv'

T14_topics_filename  = './data/raw/T14/trec2014-ts-topics-test.xml'
T14_matches_filename = './data/raw/T14/matches.tsv'
T14_nuggets_filename = './data/raw/T14/nuggets.tsv'
T14_updates_filename = './data/raw/T14/updates_sampled.tsv'

T15_topics_filename  = './data/raw/T15/trec2015-ts-topics-test.xml'
T15_matches_filename = './data/raw/T15/matches.tsv'
T15_nuggets_filename = './data/raw/T15/nuggets.tsv'
T15_updates_filename = './data/raw/T15/updates_sampled.tsv'

TOPICS  = "topics"
MATCHES = "matches"
UPDATES = "updates"
NUGGETS = "nuggets"

filenames = {
    "2013": {
        TOPICS:  T13_topics_filename,
        MATCHES: T13_matches_filename,
        NUGGETS: T13_nuggets_filename,
        UPDATES: T13_updates_filename
    },
    "2014": {
        TOPICS:  T14_topics_filename,
        MATCHES: T14_matches_filename,
        NUGGETS: T14_nuggets_filename,
        UPDATES: T14_updates_filename
    },
    "2015": {
        TOPICS:  T15_topics_filename,
        MATCHES: T15_matches_filename,
        NUGGETS: T15_nuggets_filename,
        UPDATES: T15_updates_filename
    }
}


class DataLoader:
    """
    DataLoader class.

    The DataLoader class has the purpose of retrieving and processing data from
    the 2013, 2014 and 2015 versions of the TREC Temporal Summarization datasets.

    This class is intended to be used only during the data generation phase, in which the processed datasets are stored.

    Attributes
    ----------
    datasets : list
        A list containing the names of the dataset that will be returned.
        Supported names are "2013", "2014", "2015".
        Ex: datasets = ["2013", "2014", "2015"]
    only_relevant : bool
        If True, returns only the portion of the data containing relevant text.
        Otherwise, returns the whole data, containing relevant and not-relevant text.
    tokenizer_name : str
        A string representing the name of the tokenizer that will be used to encode the data.
        Supported names are "spanbert", "bert" and "roberta".
    tokenizer_type : str
        A string representing the type of the tokenizer that will be used to encode the data.
        Supported names are "cased" and "uncased".
    max_num_words : int
        An integer representing the max number of words
        that will be contained in a single row of the dataframe returned.
    binary_masks : bool
        If True, the labels referred to the data will be binary (eg. [0,1,1,1,1...,0,0,1,1,1,0]).
        Otherwise, the labels referred to the data will be not-binary (eg. [0,1,2,2,2,3,0,1,2,2,2,2,2,3,0,0,0,4,0,0]).
    contextual : bool
        If True, during the encoding phase of the data, the context will be added to the text.
        If False, only the text data will be encoded.

    Methods
    -------
    load_data()
        Load the data by means of the attributes defined during the initialization of the DataLoader instance.
    """

    def __init__(
            self,
            datasets:       list,
            only_relevant:  bool = True,
            tokenizer_name: str  = "bert",
            tokenizer_type: str  = "uncased",
            max_num_words:  int  = 512,
            binary_masks:   bool = True,
            contextual:     bool = False
    ):
        self.datasets       = datasets
        self.only_relevant  = only_relevant
        self.tokenizer_name = tokenizer_name
        self.tokenizer_type = tokenizer_type
        self.max_num_words  = max_num_words
        self.binary_masks   = binary_masks
        self.contextual     = contextual

        self.lower_text_data = True if self.tokenizer_type == "uncased" else False

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizers[tokenizer_name][tokenizer_type], truncation=True, do_lower_case=self.lower_text_data)
        pass


    def load_data(self) -> pd.DataFrame:
        """
        Loads, prepares, encodes and adjusts the data.
        """
        df = self.__load_datasets()
        df = self.__preprocess_data(df)
        df = self.__encode_data(df)
        df = self.__postprocess_data(df)
        return df


    def __load_datasets(self) -> pd.DataFrame:
        """
        Loads and concatenates the datasets requested during the initialization
        of the DataLoader class.
        """

        for idx, dataset in enumerate(self.datasets):
            df = self.__load(dataset)

            if idx == 0:
                merged_df = df
            else:
                merged_df = pd.concat([merged_df, df])

        return merged_df


    def __load(
            self,
            dataset: str
    ) -> pd.DataFrame:
        """
        Loads the requested dataset.
        If the DataLoader was initialized with param only_relevant to False,
        returns the complete DataFrame
        (containing both relevant and not-relevant texts).
        --------
        Returns a pandas DataFrame consisting of
        ['update_text', 'match_start', 'match_end', 'query', 'relevant'] columns.
        """

        topics, matches, nuggets, updates = self.__read_data(dataset)
        relevant_df = self.__load_relevant_updates(matches, updates, nuggets)

        if self.only_relevant:
            df = relevant_df
        else:
            not_relevant_df = self.__load_not_relevant_updates(matches, updates)
            df = pd.concat([relevant_df, not_relevant_df])

        return df


    @staticmethod
    def __read_data(dataset: str):
        """
        Reads the raw data from .csv files.
        The path(s) in which the data can be found is defined inside the filenames dictionary.
        """

        topics_filename  = filenames[dataset][TOPICS]
        matches_filename = filenames[dataset][MATCHES]
        nuggets_filename = filenames[dataset][NUGGETS]
        updates_filename = filenames[dataset][UPDATES]

        topics  = read_xml(topics_filename)
        matches = pd.read_table(matches_filename, quoting=csv.QUOTE_NONE)
        nuggets = pd.read_table(nuggets_filename, quoting=csv.QUOTE_NONE)
        updates = pd.read_table(updates_filename, quoting=csv.QUOTE_NONE)

        topics, matches, nuggets, updates = process_data(topics, matches, nuggets, updates)

        return topics, matches, nuggets, updates


    def __load_relevant_updates(
            self,
            matches_df: pd.DataFrame,
            updates_df: pd.DataFrame,
            nuggets_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Joins the matches DataFrame with the updates DataFrame.
        In this way, only the relevant updates are retrieved.
        The resulting DataFrame is then joined with the nuggets DataFrame.
        If no columns from the nuggets DataFrame are selected, this join is useless.
        Some columns that may be useful are "timestamp" and "nugget_text".

        --------

        Returns a pandas DataFrame consisting of
        ['update_text', 'match_start', 'match_end', 'query', 'relevant'] columns.
        """

        df = matches_df \
            .merge(updates_df, left_on='update_id', right_on='update_id') \
            .merge(nuggets_df, left_on='nugget_id', right_on='nugget_id')[["update_text", "match_start", "match_end", "query"]]

        df['relevant'] = True

        return df


    def __load_not_relevant_updates(
            self,
            matches_df: pd.DataFrame,
            updates_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Returns a pandas DataFrame containing only the updates not relevant.
        An update is not relevant if its update_id is not inside the matches DataFrame.

        --------

        Returns a pandas DataFrame consisting of
        ['update_text', 'match_start', 'match_end', 'query', 'relevant'] columns.
        """

        update_ids           = set(updates_df.update_id.unique())
        relevant_updates     = set(matches_df.update_id.unique())
        not_relevant_updates = update_ids.difference(relevant_updates)

        df = updates_df[updates_df.update_id.isin(not_relevant_updates)]

        df = df[['update_text', 'query']]
        df[['match_start', 'match_end']] = -1, -1
        df['relevant'] = False

        return df


    def __preprocess_data(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Processes the data, in several steps:
            * Every row in which "update_text" or "query" are not strings, are deleted.
            * If using an "uncased" tokenizer, the text is converted to lowercase.
            * Several REs are applied to the data, in order to fit the tokenizer.
            * The spans (i.e., the interval of the text considered relevant) are aligned, avoiding
            referencing internal characters of a word by match_start or match_end.
            * The masks (the ground truth) are produced.
            * The data is collapsed: in fact, the same update_text can have several relevant portions.
            During this phase, the update_text is maintained unique, and the masks are merged together through
            a boolean OR.
        """

        df = delete_not_string_texts(df)

        if self.lower_text_data:
            df = lower_text_data(df)

        df = preprocess_text(df)
        df = align_spans(df)
        df = mask_spans(df, self.max_num_words, self.binary_masks)
        df = collapse_data(df)

        print(f"DataFrame length: {len(df)}")

        return df


    def __postprocess_data(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Post-processes the data in two (one optional) steps:
            * Deletes the first character of a sentence, if the first character is a space.
            * If during the initialization phase of the DataLoader, the attribute "binary" was set to False,
            the masks will be fixed in this way:
            if in the mask there is a situation like [0,1,2,2,2,2,2,3,2,2,2,2,3,0],
            the first 3 is replaced with a 2.
            This because 3 represents the end of a span: however, here, 2 masks are overlapping.
            So, it is necessary to fix them.
        """

        df[['text', 'mask']] = df.apply(fix_first_character_if_space, axis=1, result_type="expand")

        if not self.binary_masks:
            df['mask'] = df['mask'].apply(fix_consecutive_masks)

        return df


    def __encode_data(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Encodes the data by means of some parameters defined during the initialization of the DataLoader class.

        In particular, the encoding phase is driven by the parameters
            * "tokenizer_name" identifies the name of the Tokenizer that will be used for the encoding
            * "tokenizer_type" identifies the typology of the Tokenizer that will be used for the encoding
            * if "contextual" is True, the context of the update_text is also encoded.
            (e.g., context = "costa concordia", update_text = "500 deaths in costa concordia disaster",
            text encoded = "500 deaths in costa concordia disaster" + Separator + "costa concordia")
            * "max_num_words" identifies the maximum number of words that the tokenizer will take into consideration
            during the encoding phase.
        """

        print(f"> Contextual: {self.contextual}")
        print(f"> Lower case: {self.lower_text_data}")

        def encode_plus(row):
            text = row["text"]
            query = row["query"]

            if self.contextual:
                text = text + separators[self.tokenizer_name] + query

            inputs = self.tokenizer.encode_plus(
                text,
                max_length=self.max_num_words,
                padding="max_length",
                return_token_type_ids=False
            )

            return inputs["input_ids"], inputs["attention_mask"]

        df[["input_ids", "attention_mask"]] = df.apply(lambda row: encode_plus(row), axis=1, result_type='expand')

        df = drop_rows_exceeding_max_len(df, self.max_num_words)

        return df
