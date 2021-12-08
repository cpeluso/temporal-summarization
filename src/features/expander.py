import pandas as pd

class ExpandedRecord:
  
  def __init__(self, text: list, mask: list):
    
    self.text = text
    self.mask = mask
    pass

  def update_mask(self, span_start: int, span_end: int, value: int = 1):
    self.mask[span_start: span_end] = [value] * (span_end - span_start)
    pass

  def is_empty(self):
    return len(self.text) == 0

  def has_enough_space(self, index: int, starts: list, span_lengths: list) -> bool:
    if not entering_relevant_span(index, starts):
      return True

    span_index = get_span_index(index, starts)
    relevant_text_length = span_lengths[span_index]

    max_len = len(self.mask)

    return max_len - len(self.text) > relevant_text_length
    pass

  def is_full(self) -> bool:
    return len(self.mask) - 1 == len(self.text)
    pass

def entering_relevant_span(index: int, starts: list) -> bool:
  return index in starts

def get_span_index(index: int, starts: list) -> int:
  return starts.index(index) if index in starts else None

def append(expanded_record: ExpandedRecord, records: list, MAX_LEN: int) -> ExpandedRecord:
  # Find index of last space character inside expanded_record array
  last_space_in_expanded_record_index = "".join(expanded_record.text).rfind(" ")

  # Append trimmed expanded_record list to records
  records.append(
    ExpandedRecord(
      text = "".join(expanded_record.text[:last_space_in_expanded_record_index]), 
      # interval    = expanded_record.interval,
      # nugget_text = expanded_record.nugget_text,
      # timestamp   = expanded_record.timestamp,
      # query       = expanded_record.query,
      mask        = expanded_record.mask
    ).__dict__
  )

  # Return expanded_record leftover portion  
  return ExpandedRecord(
      text = expanded_record.text[last_space_in_expanded_record_index:],
      # interval    = default_interval,
      # nugget_text = expanded_record.nugget_text,
      # timestamp   = expanded_record.timestamp,
      # query       = expanded_record.query,
      mask        = [0] * MAX_LEN
  )
  
def expand_records(data: list, padding: int):

  MAX_LEN          = padding
  expanded_records = []

  for record in data:

    records = []

    text         = record.text
    starts       = [span.match_start for span in record.spans]
    span_lengths = [span.length      for span in record.spans]

    expanded_record = ExpandedRecord(
        text = [],
        mask = [0] * MAX_LEN
    )

    for index, character in enumerate(list(text)):

      if expanded_record.is_full() or not expanded_record.has_enough_space(index, starts, span_lengths):
        if not expanded_record.is_empty():
          expanded_record = append(expanded_record, records, MAX_LEN)
      
      if entering_relevant_span(index, starts) and expanded_record.has_enough_space(index, starts, span_lengths):

        relevant_span_index = get_span_index(index, starts)
        span_length         = span_lengths[relevant_span_index]

        start_index = len(expanded_record.text)
        end_index   = len(expanded_record.text) + span_length

        expanded_record.update_mask(start_index, end_index, 1)

      expanded_record.text.append(character)

    expanded_record.text = "".join(expanded_record.text)

    if not expanded_record.is_empty():
      records.append(expanded_record.__dict__)

    expanded_records.append(records)

  return expanded_records

def fix_first_character_if_space(row: pd.Series):
  text = row["text"]
  mask = row["mask"]

  if text[0] == " ":
    text = text[1:]
  
  if text[-1] == " ":
    text = text[:-1]

  return text, mask