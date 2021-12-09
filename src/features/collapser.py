import pandas as pd

class Span:

    def __init__(
            self,
            match_start: int,
            match_end: int
    ):
        self.match_start = match_start
        self.match_end = match_end
        self.length = match_end - match_start


class CollapsedRecord:

    def __init__(
            self,
            text: str,
            match_start: int,
            match_end: int,
            query: str,
            mask: list = None
    ):
        self.text = text
        self.spans = [Span(match_start, match_end)]
        self.query = query
        self.mask = mask
        pass

    def to_dict(self):
        return {
            'text': self.text,
            'mask': self.mask,
            'query': self.query
        }

    def update_span(self, match_start: int, match_end: int):
        self.spans.append(Span(match_start, match_end))
        pass

    def update_mask(self, mask: list):
        self.mask = [a or b for a, b in zip(self.mask, mask)]
        pass


def collapse_span_records(data: list) -> list:
    collapsed_data = []

    distinct_updates = set([record["update_text"] for record in data])

    for record in data:

        update_text = record["update_text"]
        match_start = record["match_start"]
        match_end = record["match_end"]
        query = record["query"]

        collapsed_record = list(filter(lambda collapsed_record: collapsed_record.text == update_text, collapsed_data))
        collapsed_record = collapsed_record[0] if collapsed_record != [] else None

        if collapsed_record:
            collapsed_record.update_span(match_start, match_end)
        else:
            collapsed_data.append(CollapsedRecord(update_text, match_start, match_end, query))

    assert len(distinct_updates) == len(collapsed_data)

    return collapsed_data


def collapse_mask_records(data: list) -> list:
    collapsed_data = []

    distinct_updates = set([record["text"] for record in data])

    for record in data:

        update_text = record["text"]
        match_start = record["match_start"]
        match_end = record["match_end"]
        mask = record["mask"]
        query = record["query"]

        collapsed_record = list(filter(lambda collapsed_record: collapsed_record.text == update_text, collapsed_data))
        collapsed_record = collapsed_record[0] if collapsed_record != [] else None

        if collapsed_record:
            collapsed_record.update_span(match_start, match_end)
            collapsed_record.update_mask(mask)
        else:
            collapsed_data.append(CollapsedRecord(update_text, match_start, match_end, query, mask))

    assert len(distinct_updates) == len(collapsed_data)

    return collapsed_data


def collapse_records(data: list, records_type="span") -> list:
    if records_type == "span":
        return collapse_span_records(data)

    if records_type == "mask":
        return collapse_mask_records(data)

    raise Exception(f"Wrong records_type parameter: {records_type}")