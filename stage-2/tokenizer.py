import os
import csv
from typing import List
import pandas as pd
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger

INPUT_CSV_DIR: str = "./stage-2/cleaned_data"
OUTPUT_CSV_DIR: str = "./stage-2/tokenized_data"
# INPUT_CSV_DIR: str = "./stage-2/cleaned_data_sample"
# OUTPUT_CSV_DIR: str = "./stage-2/tokenized_data_sample"

BATCH_SIZE = 20000
DRIVER_BATCH_SIZE = 5000
# ref: https://github.com/ckiplab/ckip-transformers/blob/master/docs/main/tag/pos.csv
STOPWORDS_POS = {
    "Caa", # 對等連接詞
    "Cab", # 連接詞，如：等等
    "Cba", # 連接詞，如：的話
    "Cbb", # 關聯連接詞
    "D", # 副詞
    "I", # 感嘆詞
    "Neu", #數詞定詞
    "Nf", #量詞
    "P", # 介詞
    "T", # 語助詞
    "COLONCATEGORY",  # 冒號
    "COMMACATEGORY",  # 逗號
    "DASHCATEGORY",  # 破折號
    "DOTCATEGORY",  # 點號
    "ETCCATEGORY",  # 刪節號
    "EXCLAMATIONCATEGORY",  # 驚嘆號
    "PARENTHESISCATEGORY",  # 括號
    "PAUSECATEGORY",  # 頓號
    "PERIODCATEGORY",  # 句號
    "QUESTIONCATEGORY",  # 問號
    "SEMICOLONCATEGORY",  # 分號
    "SPCHANGECATEGORY",  # 雙直線
    "WHITESPACE",  # 空白
    }
STOPWORDS_WS = {" ", "\\", "/", "／", "？", "「", "」", "～", "～", "！", "　", "："}

ws_driver  = CkipWordSegmenter(model="bert-base", device=-1)
pos_driver = CkipPosTagger(model="bert-base", device=0)

os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

def tokenize_and_remove_stopwords(input_batch) -> List[List[str]]:
    ws_batch  = ws_driver(input_batch, batch_size=DRIVER_BATCH_SIZE)
    pos_batch = pos_driver(ws_batch)

    res: List[List[str]] = [
        [
            ws.strip(" 　")  # remove full width and half width spaces
                for ws, pos in zip(wss, poss)
                if ws and pos not in STOPWORDS_POS and ws not in STOPWORDS_WS
        ] for wss, poss in zip(ws_batch, pos_batch)
    ]

    return res

def process_csv_file(filename: str, input_filepath: str, output_filepath: str) -> None:
    with open(input_filepath, "r", encoding="utf-8") as in_file, \
         open(output_filepath, "w", newline='', encoding="utf-8") as out_file:
        reader = csv.DictReader(in_file)
        fieldnames: List[str] = reader.fieldnames
        if fieldnames is None:
            print(f"No fieldnames in CSV file: {input_filepath}")
            return
        
        writer = csv.writer(out_file) # write data with no field name
        for chunk in pd.read_csv(input_filepath, chunksize=BATCH_SIZE):
            chunk_texts = chunk["Title"].fillna("")
            chunk_boards = chunk["Board"].fillna("")

            tokenized = tokenize_and_remove_stopwords(chunk_texts)

            for board, tokens in zip(chunk_boards, tokenized):
                output_data = [board, *tokens]
                writer.writerow(output_data)

def start_tokenize() -> None:
    for filename in os.listdir(INPUT_CSV_DIR):
        if filename.endswith(".csv"):
            input_filepath = os.path.join(INPUT_CSV_DIR, filename)
            output_filepath = os.path.join(OUTPUT_CSV_DIR, filename.replace("cleaned", "tokenized"))
            print(f"Processing {input_filepath} -> {output_filepath}")
            process_csv_file(filename.removesuffix(".csv"), input_filepath, output_filepath)

    print("Data tokenization complete.")

if __name__ == "__main__":
    start_tokenize()