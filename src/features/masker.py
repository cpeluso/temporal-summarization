import re
import pandas as pd

SEP = "<_>"

def pad_mask(row: pd.Series, padding: int):
  mask = row["mask"]

  pad = [0] * (padding - len(mask))
  mask.extend(pad)

  return mask, len(mask) != padding


def find_sub_list(
    sub_list:  list,
    main_list: list
) -> (int, int):

    sub_list_length = len(sub_list)

    for index in (i for i, e in enumerate(main_list) if e == sub_list[0]):
        if main_list[index : index + sub_list_length] == sub_list:
            return index, index + sub_list_length - 1

    raise Exception("Sub list not found!")

def word_masking(
    str_text:   str,
    span_start: int,
    span_end:   int
) -> (str, list, int, int):

  to_be_splitted_text = str_text
  to_be_splitted_span = str_text[span_start:span_end]

  text = list(filter(None, re.split(f'{SEP}| ', to_be_splitted_text)))
  span = list(filter(None, re.split(f' ', to_be_splitted_span)))

  mask_start, mask_end = find_sub_list(span, text)

  mask = [0] * len(text)
  mask[mask_start:mask_end] = [1] * (mask_end - mask_start + 1)

  return to_be_splitted_text.replace(SEP, ""), mask, span_start-len(SEP), span_end-len(SEP)

def pd_word_masking(
    row: pd.Series
) -> (str, list, int, int):

  text        = row["update_text"]
  match_start = row["match_start"] 
  match_end   = row["match_end"]
  relevant    = row["relevant"]

  if not relevant:
    return text, [0] * len(text.split()), match_start, match_end

  if len(text) - 1 == match_end and match_start == 0:
    return text.replace(SEP, ""), [1] * len(text.split()), match_start-len(SEP), match_end-len(SEP)

  return word_masking(text, match_start, match_end)

def word_multi_masking(
    str_text:   str,
    span_start: int,
    span_end:   int
) -> (str, list, int, int):

  to_be_splitted_text = str_text
  to_be_splitted_span = str_text[span_start:span_end]

  text = list(filter(None, re.split(f'{SEP}| ', to_be_splitted_text)))
  span = list(filter(None, re.split(f' ', to_be_splitted_span)))

  mask_start, mask_end = find_sub_list(span, text)

  mask = [0] * len(text)

  if mask_start == mask_end:
    mask[mask_start] = 4

  if mask_end - mask_start >= 2:
    mask[mask_start] = 1
    mask[mask_start + 1: mask_end - 1] = [2] * (mask_end - mask_start - 1)
    mask[mask_end] = 3

  return to_be_splitted_text.replace(SEP, ""), mask, span_start-len(SEP), span_end-len(SEP)

def pd_word_multi_masking(
    row: pd.Series
) -> (str, list, int, int):

  text        = row["update_text"]
  match_start = row["match_start"]
  match_end   = row["match_end"]
  relevant    = row["relevant"]

  if not relevant:
    return text, [0] * len(text.split()), match_start, match_end

  if len(text) - 1 == match_end and match_start == 0:
    mask = [1]
    mask.extend([2] * (len(text.split()) - 2))
    mask.extend([3])
    return text.replace(SEP, ""), mask, match_start-len(SEP), match_end-len(SEP)

  return word_multi_masking(text, match_start, match_end)