tokenizers = {
    "spanbert":
        {
            "cased":   "SpanBERT/spanbert-large-cased"
        },
    "bert":
        {
            "uncased": "bert-large-uncased",
            "cased":   "bert-large-cased"
        },
    "roberta":
        {
            "uncased": "roberta-large",
            "cased":   "roberta-large"
        }
}

separators = {
    "spanbert": " [SEP] ",
    "bert":     " [SEP] ",
    "roberta":  " <s> ",
}