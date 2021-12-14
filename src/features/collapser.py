"""
This file has the purpose of collapsing each row contained in the given dataset,
while maintaining their masks.

The exposed function, collapse_records, receives in input a list of dictionaries,
where each dictionary represent a row inside the DataFrame being processed.

There are dictionaries whose keys "text" may result to be equal.
Also in these cases, a single CollapsedRecord is yield, containing attributes
    * text  : the document
    * query : the context
    * mask  : the masks applied previously to each text (based on match_start and match_end attributes).
    Each mask is applied on top of every other mask already applied, through a boolean OR operation.

It returns a list of CollapsedRecord, whose length is equal to the number of distinct update_texts.
"""

import pandas as pd

from utils.decorators import deprecated

class Span:

    def __init__(
            self,
            match_start: int,
            match_end:   int
    ):
        self.match_start = match_start
        self.match_end   = match_end
        self.length      = match_end - match_start


class CollapsedRecord:

    def __init__(
            self,
            text:        str,
            match_start: int,
            match_end:   int,
            query:       str,
            mask:        list = None
    ):
        self.text  = text
        self.spans = [Span(match_start, match_end)]
        self.query = query
        self.mask  = mask
        pass

    def to_dict(self):
        return {
            'text':  self.text,
            'mask':  self.mask,
            'query': self.query
        }

    def update_span(self, match_start: int, match_end: int):
        self.spans.append(Span(match_start, match_end))
        pass

    def update_mask(self, mask: list):
        self.mask = [a or b for a, b in zip(self.mask, mask)]
        pass


@deprecated
def __collapse_span_records(data: list) -> list:
    collapsed_data = []

    distinct_updates = set([record["update_text"] for record in data])

    for record in data:

        update_text = record["update_text"]
        match_start = record["match_start"]
        match_end   = record["match_end"]
        query       = record["query"]

        collapsed_record = list(filter(lambda collapsed_record: collapsed_record.text == update_text, collapsed_data))
        collapsed_record = collapsed_record[0] if collapsed_record != [] else None

        if collapsed_record:
            collapsed_record.update_span(match_start, match_end)
        else:
            collapsed_data.append(CollapsedRecord(update_text, match_start, match_end, query))

    assert len(distinct_updates) == len(collapsed_data)

    return collapsed_data


def __collapse_mask_records(data: list) -> list:
    collapsed_data = []

    distinct_updates = set([record["text"] for record in data])

    for record in data:

        update_text = record["text"]
        match_start = record["match_start"]
        match_end   = record["match_end"]
        mask        = record["mask"]
        query       = record["query"]

        collapsed_record = list(filter(lambda record: record.text == update_text, collapsed_data))
        collapsed_record = collapsed_record[0] if collapsed_record != [] else None

        if collapsed_record:
            collapsed_record.update_span(match_start, match_end)
            collapsed_record.update_mask(mask)
        else:
            collapsed_data.append(CollapsedRecord(update_text, match_start, match_end, query, mask))

    assert len(distinct_updates) == len(collapsed_data)

    return collapsed_data


def collapse_records(data: list, records_type: str = "mask") -> list:
    """
    Receives in input a list of dictionaries,
    where each dictionary represent a row inside the DataFrame being processed.

    There are dictionaries whose keys "text" may result to be equal.
    In these cases, a single CollapsedRecord is yield, containing attributes
        * text  : the document
        * query : the context
        * mask  : the masks applied previously to each text (based on match_start and match_end attributes).
        Each mask is applied on top of every other mask already applied through a boolean OR operation.

    Returns a list of CollapsedRecord, whose length is equal to the number of distinct update_texts.
    """

    if records_type == "span":
        return __collapse_span_records(data)

    if records_type == "mask":
        return __collapse_mask_records(data)

    raise Exception(f"Wrong records_type parameter: {records_type}")
