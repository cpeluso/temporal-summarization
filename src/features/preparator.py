import xml.etree.ElementTree as et
import pandas as pd

def read_xml(path):

    xtree = et.parse(path)
    xroot = xtree.getroot()

    df_cols = [
        "query_id",
        "query",
        # "title",
        # "description",
        # "start",
        # "end",
        # "type",
        # "locations",
        # "deaths",
        # "injuries"
    ]

    rows = []

    for node in xroot:
        s_id = node.find("id").text if node is not None else None
        s_query = node.find("query").text if node is not None else None
        # s_title       = node.find("title").text if node is not None else None
        # s_description = node.find("description").text if node is not None else None
        # s_start       = node.find("start").text if node is not None else None
        # s_end         = node.find("end").text if node is not None else None
        # s_type        = node.find("type").text if node is not None else None
        # s_locations   = node.find("locations").text if node is not None else None
        # s_deaths      = node.find("deaths").text if node is not None else None
        # s_injuries    = node.find("injuries").text if node is not None else None

        rows.append(
            {
                "query_id": s_id,
                "query": s_query,
                # "title"       : s_title,
                # "description" : s_description,
                # "start"       : s_start,
                # "end"         : s_end,
                # "type"        : s_type,
                # "locations"   : s_locations,
                # "deaths"      : s_deaths,
                # "injuries"    : s_injuries
            }
        )

    return pd.DataFrame(rows, columns=df_cols)


def query_id_to_topic_id(row):
    _, topic_id = row.split(".")

    return topic_id


def process_data(topics_df, matches_df, nuggets_df, updates_df):

    if "int" not in str(type(matches_df['query_id'][0])):
      matches_df['query_id'] = matches_df['query_id'].apply(str.strip)
      matches_df['topic_id'] = matches_df['query_id'].apply(query_id_to_topic_id)
      matches_df = matches_df.join(topics_df.set_index('query_id'), on='topic_id', lsuffix='_caller').drop('query_id',axis=1)
    else:
      topics_df['query_id'] = topics_df['query_id'].apply(lambda query: int(query))
      matches_df = matches_df.join(topics_df.set_index('query_id'), on='query_id', lsuffix='_caller').drop('query_id',axis=1)

    if "int" not in str(type(updates_df['query_id'][0])):
      updates_df['query_id'] = updates_df['query_id'].apply(str.strip)
      updates_df['topic_id'] = updates_df['query_id'].apply(query_id_to_topic_id)
      updates_df = updates_df.join(topics_df.set_index('query_id'), on='topic_id', lsuffix='_caller').drop('query_id',axis=1)
    else:
      updates_df = updates_df.join(topics_df.set_index('query_id'), on='query_id', lsuffix='_caller').drop('query_id',axis=1)

    if "int" not in str(type(nuggets_df['query_id'][0])):
      nuggets_df['query_id'] = nuggets_df['query_id'].apply(str.strip)
      nuggets_df['topic_id'] = nuggets_df['query_id'].apply(query_id_to_topic_id)
      nuggets_df = nuggets_df.join(topics_df.set_index('query_id'), on='topic_id', lsuffix='_caller').drop('query_id',axis=1)
    else:
      nuggets_df = nuggets_df.join(topics_df.set_index('query_id'), on='query_id', lsuffix='_caller').drop('query_id',axis=1)

    return topics_df, matches_df, nuggets_df, updates_df