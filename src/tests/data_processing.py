from transformers import AutoTokenizer
from src.features.pipeline import fix_first_character_if_space

def __encode(tokenizer, text, max_len):

  return tokenizer.encode_plus(
      text,
      max_length            = max_len,
      padding               = "max_length",
      return_token_type_ids = False
  )['input_ids']

def __decode(tokenizer, input_ids):
  decoded_str_text = tokenizer.decode(input_ids)

  decoded_str_text = decoded_str_text.replace("[CLS]", "")
  decoded_str_text = decoded_str_text.replace("[SEP]", "")
  decoded_str_text = decoded_str_text.replace("[PAD]", "")
  decoded_str_text = decoded_str_text.replace("<s>", "")
  decoded_str_text = decoded_str_text.replace("</s>", "")
  decoded_str_text = decoded_str_text.replace("<pad>", "")

  return decoded_str_text

def test_data_processing(df, tokenizer_name, max_len, tokenizer_type):
  do_lower_case = True if tokenizer_type == "uncased" else False
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, truncation=True, do_lower_case=do_lower_case)

  df[['text', 'mask']] = df.apply(fix_first_character_if_space, axis = 1, result_type = "expand")
  df['input_ids']    = df['text'].apply(lambda text: __encode(tokenizer, text, max_len))
  df['decoded_text'] = df['input_ids'].apply(lambda input_ids: __decode(tokenizer, input_ids))
  df['match']        = df.apply(lambda row: row['decoded_text'].split() == row['text'].split(), axis = 1)

  if len(df[df.match == False]) > 0:
    print(f"{len(df[df.match == False])} sentences wrongly encoded-decoded.")
    print(f"{len(df[df.match == False]) * 100 / len(df)} % of sentences are wrongly encoded-decoded.")
    print()
    for i in range(len(df[df.match == False])):
      og  = df[df.match == False].iloc[i]["text"].split()
      new = df[df.match == False].iloc[i]["decoded_text"].split()
      print("*************")
      print()
      print("\033[1m" + "Original text:" + "\033[0m")
      print(f"length: {len(og)}")
      print(og)
      print()
      print("\033[1m" + "Decoded text:" + "\033[0m")
      print(f"length: {len(new)}")
      print(new)
      print()
      print("*************")
      print()
      og  = df[df.match == False].iloc[i]["text"]
      new = df[df.match == False].iloc[i]["decoded_text"]
      print("*************")
      print()
      print("\033[1m" + "Original text:" + "\033[0m")
      print(f"length: {len(og)}")
      print(og)
      print()
      print("\033[1m" + "Decoded text:" + "\033[0m")
      print(f"length: {len(new)}")
      print(new)
      print()
      print("*************")
      print()
      print("*************")
    raise Exception()

  if len(df[df.match == False]) == 0:
    print("100.00 % of sentences are correctly encoded-decoded.")
    return True
