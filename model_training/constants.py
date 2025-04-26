from typing import List

CRAWLER_DATA_PATH: str = "./model_training/crawler_data"
DATA_SOURCE_URL: str = "https://www.ptt.cc"
D2V_MODEL_PATH = "./model_training/doc2vec_model/custom_d2v_model.bin"
EMBEDDED_DATA_PATH = "./model_training/embedded_data/embedded_data.csv"
CLASSIFY_MODEL_PATH = "./model_training/classify_models/classify_model.pth"

DATA_SOURCE_COOKIES: dict[str, str] = {"over18": "1"}

BOARDS: List[str] = [
    "Baseball",
    "Boy-Girl",
    "C_Chat",
    "HatePolitics",
    "LifeIsMoney",
    "Military",
    "PC_Shopping",
    "Stock",
    "Tech_Job",
]

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