"""
This file has the purpose of masking each text contained in the given dataset,
in order to distinguish between relevant and not-relevant spans.

The exposed functions, pd_word_multi_masking and pd_word_masking, 
receives in input a pd.Series containing a string (the update_text column),
two integer indices (span_start, span_end) and a boolean (relevant).

It aligns the indices to the beginning of the word selected by span_start
and to the end of the word selected by span_end.

In particular:

  pd_word_masking creates masks like [0,1,1,1,1,1,0], 
  where the i-th element of the mask could be:
    * 0 if the i-th word of the text is not a word belonging to the span
    * 1 if the i-th word of the text is a word belonging to the span 
  
  It returns a string, a list and two integers (span_start and span_end).

  ------

  pd_word_multi_masking creates masks like [0,1,2,2,2,3,0], 
  
  where the i-th element of the mask could be:
    * 0 if the i-th word of the text is not a word belonging to the span
    * 1 if the i-th word of the text is the first word belonging to the span 
    * 2 if the i-th word of the text is neither the first nor the last word belonging to the span
    * 3 if the i-th word of the text is the last word belonging to the span
    * 4 if the i-th word of the text is the only word belonging to the span
  
  It returns a string, a list and two integers (span_start and span_end).
"""

import re
import pandas as pd

SEP = "<_>"

def pad_mask(row: pd.Series, padding: int):
  mask = row["mask"]

  pad = [0] * (padding - len(mask))
  mask.extend(pad)

  return mask, len(mask) != padding


def __find_sub_list(
    sub_list:  list,
    main_list: list
) -> (int, int):

    sub_list_length = len(sub_list)

    for index in (i for i, e in enumerate(main_list) if e == sub_list[0]):
        if main_list[index : index + sub_list_length] == sub_list:
            return index, index + sub_list_length - 1

    raise Exception("Sub list not found!")

def __word_masking(
    str_text:   str,
    span_start: int,
    span_end:   int
) -> (str, list, int, int):
  """
    Receives in input a string and two integer indices (span_start, span_end).
    Creates a mask [0,1,1,1,1,1,0], where the i-th element of the mask could be:
      * 0 if the i-th word of the text is not a word belonging to the span
      * 1 if the i-th word of the text is a word belonging to the span 
    Returns a string, a list and two integers (span_start and span_end).
  """

  to_be_splitted_text = str_text
  to_be_splitted_span = str_text[span_start:span_end]

  text = list(filter(None, re.split(f'{SEP}| ', to_be_splitted_text)))
  span = list(filter(None, re.split(f' ', to_be_splitted_span)))

  mask_start, mask_end = __find_sub_list(span, text)

  mask = [0] * len(text)
  mask[mask_start:mask_end] = [1] * (mask_end - mask_start + 1)

  return to_be_splitted_text.replace(SEP, ""), mask, span_start-len(SEP), span_end-len(SEP)

def __word_multi_masking(
    str_text:   str,
    span_start: int,
    span_end:   int
) -> (str, list, int, int):
  """
    Receives in input a string and two integer indices (span_start, span_end).
    Creates a mask [0,1,2,2,2,3,0], where the i-th element of the mask could be:
      * 0 if the i-th word of the text is not a word belonging to the span
      * 1 if the i-th word of the text is the first word belonging to the span 
      * 2 if the i-th word of the text is neither the first nor the last word belonging to the span
      * 3 if the i-th word of the text is the last word belonging to the span
      * 4 if the i-th word of the text is the only word belonging to the span
    Returns a string, a list and two integers (span_start and span_end).
  """

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


def pd_word_masking(
    row: pd.Series
) -> (str, list, int, int):
  """
    Pandas wrapper of the word_masking function.
  """

  text        = row["update_text"]
  match_start = row["match_start"] 
  match_end   = row["match_end"]
  relevant    = row["relevant"]

  if not relevant:
    return text, [0] * len(text.split()), match_start, match_end

  if len(text) - 1 == match_end and match_start == 0:
    return text.replace(SEP, ""), [1] * len(text.split()), match_start-len(SEP), match_end-len(SEP)

  return __word_masking(text, match_start, match_end)


def pd_word_multi_masking(
    row: pd.Series
) -> (str, list, int, int):
  """
    Pandas wrapper of the word_mulit_masking function.
  """

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

  return __word_multi_masking(text, match_start, match_end)