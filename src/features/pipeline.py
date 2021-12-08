import pandas as pd

from src.features.aligner      import *
from src.features.masker       import *
from src.features.collapser    import *
from src.features.preprocessor import *

def lower_text_data(df: pd.DataFrame) -> pd.DataFrame:
    print("> Making text lowercase...")
    df['update_text'] = df['update_text'].apply(str.lower)
    df['query'] = df['query'].apply(str.lower)
    return df


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    print("> Preprocessing the data...")
    df[['update_text', 'match_start', 'match_end']] = df.apply(pd_apply_regex, axis=1, result_type="expand")
    return df


def align_spans(df: pd.DataFrame) -> pd.DataFrame:
    print("> Aligning spans...")
    df[["update_text", "match_start", "match_end", "changed"]] = df.apply(pd_align_span_indices, axis=1, result_type="expand")
    df = df[df["match_start"] != -1]
    return df


def mask_spans(df: pd.DataFrame, padding: int, binary=True) -> pd.DataFrame:
    masks_type = "binary" if binary else "not-binary"
    print(f"> Creating {masks_type} masks...")

    if binary:
        df[["text", "mask", "match_start", "match_end"]] = df.apply(pd_word_masking, axis=1, result_type="expand")
    else:
        df[["text", "mask", "match_start", "match_end"]] = df.apply(pd_word_multi_masking, axis=1, result_type="expand")

    df[["mask", "delete_it"]] = df.apply(lambda row: pad_mask(row, padding), axis=1, result_type="expand")
    print(f"About to remove {len(df[df.delete_it == True])} row(s).")
    df = df[df.delete_it == False]
    return df


def fix_consecutive_masks(mask: list) -> pd.DataFrame:
    cleaned_mask = []

    last_element = -1

    for element in mask:

        if element == 2 and last_element == 3:
            cleaned_mask[-1] = 2

        cleaned_mask.append(element)
        last_element = element

    return cleaned_mask


def collapse_data(df: pd.DataFrame) -> pd.DataFrame:
    print("> Collapsing the data...")
    dict_df = df.to_dict('records')
    collapsed_data = collapse_records(dict_df, "mask")
    df = pd.DataFrame.from_records([data.to_dict() for data in collapsed_data])
    return df


def drop_rows_exceeding_max_len(
        df: pd.DataFrame,
        max_len: int
) -> pd.DataFrame:
    print("> Removing large rows...")
    df['n_words'] = df.input_ids.apply(lambda input_ids: len(input_ids))

    print(f"About to remove {len(df[df.n_words > max_len])} row(s) that contain text larger than {max_len} words.")
    df = df[df.n_words <= max_len]

    df.drop(columns=['n_words'], inplace=True)

    return df