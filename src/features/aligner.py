import re
import pandas as pd
import string

SEP = "<_>"

def validate_index(
        text:  str,
        index: int
) -> bool:
    return 0 <= index <= len(text)


def previous_character_is_valid(
        text:  str,
        index: int
) -> bool:
    return text[index - 1] == " " or text[index - 1] in string.punctuation and text[index].islower()


def following_character_is_valid(
        text:  str,
        index: int
) -> bool:
    return len(text) - 1 == index or text[index] == " " or text[index] in string.punctuation


def align_span_start_index(
        text:  str,
        index: int
) -> int:

    if text[index] == " " and text[index + 1].islower():
      return index + 1

    if previous_character_is_valid(text, index):
        return index

    index = find_closest_valid_character(text, index, "P")

    return index


def align_span_end_index(
        text:  str,
        index: int
) -> int:
    max_len = len(text)

    if index == max_len or following_character_is_valid(text, index):
        return index

    index = find_closest_valid_character(text, index, "F")

    return index


def find_closest_valid_character(
        text:    str,
        index:   int,
        default: str
) -> int:
    fwd_index = index
    pvs_index = index

    max_len = len(text)

    while not previous_character_is_valid(text, pvs_index) and pvs_index > 0:
        pvs_index -= 1

    while not following_character_is_valid(text, fwd_index) and fwd_index < max_len:
        fwd_index += 1

    fwd_error = abs(index - fwd_index)
    pvs_error = abs(index - pvs_index)

    if pvs_error < fwd_error:
        return pvs_index

    if fwd_error < pvs_error:
        return fwd_index

    if default == "P":
        return pvs_index

    if default == "F":
        return fwd_index

    raise Exception()


def align_span_indices(
        text:       str,
        span_start: int,
        span_end:   int
) -> (str, int, int, bool):
    """
      Receives in input a string and two integer indices (span_start, span_end).
      Align the indices to the beginning of the word selected by span_start
      and to the end of the word selected by span_end.
      Returns two integers (span_start_aligned and span_end_aligned).
    """
    if not validate_index(text, span_start) or not validate_index(text, span_end):
        return text, -1, -1, True

    aligned_span_start = align_span_start_index(text, span_start)
    aligned_span_end   = align_span_end_index(text, span_end)

    if re.search('[a-zA-Z]', text[aligned_span_start:aligned_span_end]) is None:
        return text, -1, -1, True

    span = SEP + text[aligned_span_start:aligned_span_end] + SEP
    text = text[:aligned_span_start] + span + text[aligned_span_end:]

    return text, \
           aligned_span_start + len(SEP), \
           aligned_span_end + len(SEP), \
           aligned_span_start != span_start or aligned_span_end != span_end


def pd_align_span_indices(
        row: pd.Series
) -> (str, int, int, bool):
    """
      Pandas wrapper of the align_span_indices function.
    """

    text       = row["update_text"]
    span_start = row["match_start"]
    span_end   = row["match_end"]

    return align_span_indices(text, span_start, span_end)