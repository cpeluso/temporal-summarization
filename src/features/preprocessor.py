"""
This file has the purpose of applying regular expressions to each text contained in the dataset.

Such regular expressions are applied in order to maintain the one-to-one mapping between
original text and encode-decoded text (by means of a BERT-like Tokenizer).

The exposed function pd_apply_regex, 
receives in input a pd.Series containing a string (the update_text column),
two integer indices (span_start, span_end) and a boolean (relevant).

It applies every regex defined to each text and keeps the span_start and span_end aligned.

It returns a string and two integers (aligned_span_start and aligned_span_end).
"""

import re
import pandas as pd


def __realign_indices(shift, match_start, match_end, regex_start, regex_end):
    """
    Realign indices after applying a regex.
    """

    if match_start <= regex_start:
        new_match_start = match_start

    if match_start > regex_start:
        new_match_start = match_start - shift

    if match_end <= regex_end:
        new_match_end = match_end

    if match_end > regex_end:
        new_match_end = match_end - shift

    return new_match_start, new_match_end


def apply_regex(text: str, match_start: int, match_end: int, relevant: bool):
    """
    Receives in input a pd.Series containing a string (the update_text column),
    two integer indices (span_start, span_end) and a boolean (relevant).
    Applies every regex defined to each text and keeps the span_start and span_end aligned.
    Returns a string and two integers (aligned_span_start and aligned_span_end).
    """
    all_regex = [
        (r'[0-9],[0-9]', lambda match: match.group()[0] + match.group()[-1]),
        (r'[`“]', "_ "),
        (r' [^\w] ?', lambda match: match.group()[1:]),
        (r'-', ' '),
        (r'[^\s]’[^\s]', ' ’ '),
        (r'lap\.\.\.s', 'lap... s'),
        (r'\s\s+', ' '),
        (r'[A-Za-z]\.[^\s\.]', lambda match: match.group()[:-1] + " " + match.group()[-1]),
        (r'([^ ])\$([^ ])', lambda match: match.group()[0] + " " + match.group()[1] + " " + match.group()[2]),
        (r'^"[^\s]', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r'[^\s]"\s', lambda match: match.group()[0] + ' \" '),
        (r'[^\s]:\s', lambda match: match.group()[0] + ' : '),
        (r'[^\s]\([^\s]', lambda match: match.group()[0] + " " + match.group()[1] + " " + match.group()[2]),
        (r'[^\s]\)', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'[^\s]_\s', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'[^\s]”\s', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'^\s', ''),
        (r'[^\s][#\:][^\s]', lambda match: match.group()[0] + " " + match.group()[1] + " " + match.group()[2]),
        (r'[^\s]\/\s', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'[^\s]\/[^\s]', lambda match: match.group()[0] + " " + match.group()[1] + " " + match.group()[2]),
        (r'[^\s]\$\s', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'\s\'\'', lambda match: match.group()[1:]),
        (r'could n\'t', 'couldn\'t'),
        (r'are n\'t', 'aren\'t'),
        (r'do not', 'don\'t'),
        (r'do n\'t', 'don\'t'),
        (r'is n\'t', 'isn\'t'),
        (r'does n\'t', 'doesn\'t'),
        (r'wo n\'t', 'won\'t'),
        (r'have n\'t', 'haven\'t'),
        (r'has n\'t', 'hasn\'t'),
        (r'did n\'t', 'didn\'t'),
        (r'was n\'t', 'wasn\'t'),
        (r'were n\'t', 'weren\'t'),
        (r'ca n\'t', 'can\'t'),
        (r'should n\'t', 'shouldn\'t'),
        (r'\'\s', '\''),
        (r'\'\'[^\s\']', lambda match: match.group()[0] + match.group()[1] + " " + match.group()[2]),
        (r'\'\'\s[\.,]', lambda match: match.group()[:2] + match.group()[-1]),
        (r'[^\s]\|\s', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'\([^\s]', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'[^\s];', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'[^\s]\]', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'\.\s\'', '.\''),
        (r'anything\.\'', "anything."),
        (r'\.\s,', lambda match: match.group()[0] + match.group()[2]),
        (r'\.[\s\n]$', lambda match: match.group()[0]),
        (r'[^\s]"[^\s]', lambda match: match.group()[0] + " " + match.group()[1] + " " + match.group()[2]),
        (r'[^\s]»', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'[^\s]\&', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'\&[^\s]', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'\][^\s]', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'\&amp', '& amp'),
        (r';raquo', '; raquo'),
        (r'&gt', '& gt'),
        (r'[^\s]_[^\s]', ' _ '),
        (r'\*[^\s]', '* '),
        (r'[^\s]— ', lambda match: match.group()[0] + ' — '),
        (r'[^\s]>', lambda match: match.group()[0] + ' >'),
        (r'[^\s]\%\s', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'[0-9]\%', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'[^\s]:', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'[0-9]\.[0-9]', lambda match: match.group()[:2] + " " + match.group()[2]),
        (r'\s\.{3}\s', lambda match: match.group()[1:]),
        (r'[a-zA-Z]=\'', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'href=', 'href ='),
        (r'[^\s]=[^\s]', ' = '),
        (r'<a', '< a'),
        (r'html#', 'html #'),
        (r'[a-z]=\s', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'<[a-zA-Z]', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'_[a-zA-Z0-9]', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'\.html', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'[a-zA-Z]\(', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'[^\s]’\s', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'[^\s]\',', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'[a-zA-Z]‘[a-zA-Z]', lambda match: match.group()[0] + " " + match.group()[1] + " " + match.group()[2]),
        (r'[a-zA-Z]\+', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'[a-zA-Z]\@', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'[a-zA-Z]‘', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'[a-zA-Z]…', lambda match: match.group()[0] + "..."),
        (r'…[a-zA-Z]', lambda match: "..." + " " + match.group()[-1]),
        (r'\'$', ''),
        (r'\s…$', ''),
        (r'[a-zA-Z]\s\.', lambda match: match.group()[0] + match.group()[-1]),
        (r'\'\s\'', '\'\''),
        (r'^[#>][a-zA-Z0-9]', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r';nbsp', '; nbsp'),
        (r'â', 'a'),
        (r'\|@', ' | @ '),
        (r'@[a-zA-Z]', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r'[a-zA-Z]«', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r'[a-zA-Z]\[', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r'[a-zA-Z]\|', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r'[a-zA-Z]\*', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r': \.\.\.', ':...'),
        (r'[a-zA-Z]\?[a-zA-Z]', lambda match: match.group()[0] + match.group()[1] + " " + match.group()[2]),
        (r'[a-zA-Z]\.\.\.[a-zA-Z]', lambda match: match.group()[:-1] + " " + match.group()[-1]),
        (r'\.\s!', '.!'),
        (r'[0-9]\+', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r'[^\x00-\x7F]', ''),
        (r': \'', ':\''),
        (r'[0-9]\.[a-zA-Z]', lambda match: match.group()[:-1] + " " + match.group()[-1]),
        (r'\[[0-9]', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r'\'\s;', lambda match: match.group()[0] + match.group()[-1]),
        (r'[a-zA-Z]<', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r'>[a-zA-Z]', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r'\|\|$', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r'[a-zA-Z]\'\.\.\.', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'[a-zA-Z]\'\.', lambda match: match.group()[0] + " " + match.group()[:-1]),
        (r'\"\s\.$', lambda match: match.group()[0] + match.group()[-1]),
        (r'[a-zA-Z]\'\s', lambda match: match.group()[:-1]),
        (r'\[[a-zA-Z]', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r'\'\'\.[a-zA-Z]', lambda match: match.group()[:-1] + " " + match.group()[-1]),
        (r'\"[a-zA-Z]', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r'rrb ,', 'rrb,'),
        (r'[0-9]\(', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r'\![0-9]', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r'\?\(', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r'[a-zA-Z]\'\?', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'><', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'\)\(', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'[0-9][\*/]', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'\+[a-zA-Z]', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'rrb\s\'s', 'rrb\'s'),
        (r',@', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'[a-zA-Z],[a-zA-Z]', lambda match: match.group()[0] + match.group()[1] + " " + match.group()[2]),

        # 2015
        (r'\\[a-zA-Z]', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r'\?\?+', '?'),
        (r'\?[\s]\?[\s]', '?? '),
        (r'[\"\.a-zA-Z0-9]\\[\s]n', lambda match: match.group()[0]),
        (r'#[a-zA-Z]', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r',\.\.\.', ', '),
        (r'\+[0-9]', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r':[\s]\?', lambda match: match.group()[0] + match.group()[-1]),
        (r'\.\.\.[a-zA-Z]',
         lambda match: match.group()[0] + match.group()[1] + match.group()[2] + " " + match.group()[3]),
        (r'/[\s]\.\.\.', lambda match: match.group()[0] + match.group()[2] + match.group()[3] + match.group()[4] + " "),
        (r'\'\'[\s]\?', lambda match: match.group()[0] + match.group()[1] + match.group()[2]),
        (r'[0-9]#', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r'\?[\s]/[\s]\?\.\.\.', ''),
        (r'\/$', ''),
        (r'[a-z]\.[a-z]', lambda match: match.group()[0] + match.group()[1] + " " + match.group()[2]),
        (r'@[0-9]', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'[0-9]\\\\', lambda match: match.group()[0] + " " + match.group()[1] + " "),
        (r'_$', ''),
        (r'\'\'\'\s\[\[', ' ['),
        (r'\'\'\'\s', '\'\'\''),
        (r'\\\\\s[a-zA-Z]', ''),
        (r'< img .*?>', ''),
        (r'< a .*? >', ''),
        (r'\'\'\'\'\/', '\'\'\'\' \/'),
        (r'alt\s=\'\'\'\'\s\\\/', ''),
        (r'^\/ref', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'[\'\?]$', ''),
        (r'[a-zA-Z0-9]http', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'[0-9]=', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'[a-zA-Z]%[0-9]', lambda match: match.group()[0] + " " + match.group()[1] + " " + match.group()[2]),
        (r'\s%[0-9]      ', lambda match: match.group()[0] + match.group()[1] + " " + match.group()[2]),
        (r'_\s\s', lambda match: match.group()[0] + match.group()[1]),
        (r'_\s\?', lambda match: match.group()[0] + match.group()[2]),
        (r'[a-zA-Z]\s\?', lambda match: match.group()[0] + match.group()[2]),
        (r'\|\\\sn$', ''),
        (r'\\\sn$', ''),
        (r'shi\s\sites', 'shi ites'),
        (r'^=', ''),
        (r'\s\ss', ' s'),
        (r',\(', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'\/\s\?\sp$', ''),
        (r'[a-z]![a-z]', lambda match: match.group()[0] + match.group()[1] + " " + match.group()[2]),
        (r'\/[0-9]', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'\'\'\'\'[a-zA-Z]', lambda match: match.group()[:-1] + " " + match.group()[-1]),
        (r'\s%[0-9]', lambda match: match.group()[0] + match.group()[1] + " " + match.group()[2]),
        (r'[a-z],[0-9]', lambda match: match.group()[0] + match.group()[1] + " " + match.group()[2]),
        (r'=http', '= http'),
        (r'~[a-z0-9]', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'&\s\s=\s\'\'', '& =\'\''),
        (r'^\/[a-z]\s', ''),
        (r'&\s\s=', '& ='),
        (r'&\s\'\'', '&\'\''),
        (r'\/p$', ''),
        (r'\/\s\'', '\/\''),
        (r'\\\/\'[a-z]', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'\\\/\\\.[a-z]\s\/', ''),
        (r'\/\s\.[a-z]', lambda match: '\/\.' + match.group()[-1]),
        (r'^\/[a-z]\.', ''),
        (r'\/\s\?', lambda match: match.group()[0] + match.group()[2] + " "),
        (r'#\s\.', lambda match: match.group()[0] + match.group()[2] + " "),
        (r'\?[a-z]', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'[0-9]\s\.\.\.', lambda match: match.group()[0] + match.group()[2:] + " "),
        (r'[a-zA-Z]~', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'#[0-9]', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'\+\%[0-9]', lambda match: match.group()[0] + " " + match.group()[1] + " " + match.group()[2]),
        (r'  =  =  =  =  = == ', ' = '),
        (r'# !', '#!'),
        (r'\\\/ \/ p p', ''),
        (r'^\/tr ', ''),
        (r'!\.\.\.[^\s]', lambda match: match.group()[:-1]),
        (r'\\\/\\\.a?', ''),
        (r'\?\s\s', lambda match: match.group()[:-1]),
        (r'_\s\'\'', lambda match: match.group().split()[0] + match.group().split()[1]),
        (r'  =  =  =  =  = \?.*=', ''),
        (r'^\/[a-z]', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r'[0-9]\.\.\.[0-9]', lambda match: match.group()[:-1] + " " + match.group()[-1]),
        (r'~~', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'[0-9],[a-z]', lambda match: match.group()[:-1] + " " + match.group()[-1]),
        (r',\s\s', lambda match: match.group()[:-1]),
        (r'\?\.\.\.[0-9]', lambda match: match.group()[:-1] + " " + match.group()[-1]),
        (r'img.*=\s_\s_\speople', lambda match: match.group().split()[-1]),
        (r'\]\s,', lambda match: match.group()[0] + match.group()[-1]),
        (r'\s=[a-z]', lambda match: match.group()[:-1] + " " + match.group()[-1]),
        (r'^\/\sstrong.*massive', 'massive'),
        (r'would n\'t', 'wouldn\'t'),
        (r'u\.\ss\.\.\.\.\*', 'u. s.... *'),
        (r'[a-z]\.\.[a-z]', lambda match: match.group()[:-1] + " " + match.group()[-1]),

        # 2013
        (r'[a-z]\"\s\/', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'>\+', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r'>{{', lambda match: match.group()[0] + " " + match.group()[1] + " " + match.group()[2] + " "),
        (r'[a-z]\|[a-z]', lambda match: match.group()[0] + " " + match.group()[1] + " " + match.group()[2]),
        (r'\|[a-z]', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r'[0-9]}}\+',
         lambda match: match.group()[0] + " " + match.group()[1] + " " + match.group()[2] + " " + match.group()[3]),
        (r'}}}', lambda match: match.group()[0] + " " + match.group()[1] + " " + match.group()[2]),
        (r'\[\[', lambda match: match.group()[0] + " " + match.group()[1]),
        (r',\[\[', lambda match: match.group()[0] + " " + match.group()[1] + " " + match.group()[2]),
        (r'\s\s\|', lambda match: match.group()[1:]),
        (r'[0-9]\|', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'[0-9]{{', lambda match: match.group()[0] + " " + match.group()[1] + " " + match.group()[2]),
        (r'\+{{[a-z]',
         lambda match: match.group()[0] + " " + match.group()[1] + " " + match.group()[2] + " " + match.group()[3]),
        (r'\],\[', lambda match: match.group()[0] + match.group()[1] + " " + match.group()[-1]),
        (r'\]\s\.$', lambda match: match.group()[0] + match.group()[-1]),
        (r'[0-9]}}', lambda match: match.group()[0] + " " + match.group()[1] + " " + match.group()[2]),
        (r'[a-z]\"\s', lambda match: match.group()[0] + " " + match.group()[1:]),
        (r'[0-9]\[', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'\.\^', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'[a-z]\^', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'\)[a-z]', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'_\(', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'[a-z]\s\s[a-z]', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r'\.\[', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'\.\s\?', lambda match: match.group()[0] + match.group()[2]),
        (r'{{[a-z]', lambda match: match.group()[0] + " " + match.group()[1] + " " + match.group()[2]),
        (r',#', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'\)\|', lambda match: match.group()[0] + " " + match.group()[1]),
        (r';[a-z]', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'[#a-z]{[0-9]}',
         lambda match: match.group()[0] + " " + match.group()[1] + " " + match.group()[2] + " " + match.group()[3]),
        (r':\s\s', lambda match: match.group()[:-1]),
        (r'\"\s\.\.\.', lambda match: match.group()[0] + match.group()[2:]),
        (r'\.\s\.\.\.', lambda match: match.group()[0] + match.group()[2:]),
        (r'\.\.\.[\*\+]', lambda match: match.group()[:-1] + " " + match.group()[-1]),
        (r':\s\.', lambda match: match.group()[0] + match.group()[-1]),
        (r'\$[0-9]', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r'\"\[', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r'\[\^', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'\.\(', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'\)\s\s[a-z]', lambda match: match.group()[0] + " " + match.group()[-1]),
        (r'\]\s!!', lambda match: match.group()[0] + match.group()[2:]),
        (r';}', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'[a-z]\s\.\.\.', lambda match: match.group()[0] + match.group()[2:]),
        (r'\.\.[0-9]', lambda match: match.group()[:2] + " " + match.group()[2]),
        (r'\"\s,', lambda match: match.group()[0] + match.group()[-1]),
        (r'[a-z]#', lambda match: match.group()[0] + " " + match.group()[1]),
        (r':\.@', lambda match: match.group()[:2] + " " + match.group()[2]),
        (r';{', lambda match: match.group()[0] + " " + match.group()[1]),
        (r',\[', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'\)#', lambda match: match.group()[0] + " " + match.group()[1]),
        (r'=\s\.', lambda match: match.group()[0] + match.group()[2]),
        (r'\)\.[a-z]', lambda match: match.group()[:2] + " " + match.group()[2]),
        (r'                  ,, ,', ','),
    ]

    for regex_tuple in all_regex:

        regex, fn = regex_tuple[0], regex_tuple[1]

        while re.search(regex, text) is not None:

            # Get interval where regex will be applied
            regex_result = re.search(regex, text)
            regex_start, regex_end = regex_result.span()

            old_text_len = len(text)

            # Modify text enlighted by regex
            text = re.sub(regex, fn, text, 1)

            new_text_len = len(text)

            shift = old_text_len - new_text_len

            # Update indices
            if relevant:
                match_start, match_end = __realign_indices(shift, match_start, match_end, regex_start, regex_end)

    return text, match_start, match_end, relevant


def pd_apply_regex(row: pd.Series):
    """
    Pandas wrapper of the apply_regex function.
    """
    return apply_regex(row["update_text"], row["match_start"], row["match_end"], row["relevant"])