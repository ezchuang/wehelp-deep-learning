import os
import csv
from typing import List
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import custom_logger
from custom_gensim_epoch_logger import EpochLogger
from tqdm import tqdm

SAMPLE_MODE = False

if SAMPLE_MODE:
    INPUT_CSV_DIR: str = "./model_training/tokenized_data_sample"
    OUTPUT_CSV_DIR: str = "./model_training/doc2vec_model_sample"
    VECTOR_SIZE = 50
    TRAINING_WINDOW = 2
    TRAINING_EPOCHS = 20
    TRAINING_MIN_COUNT = 2
    TRAINING_WORKERS = 4
else:
    INPUT_CSV_DIR: str = "./model_training/tokenized_data"
    OUTPUT_CSV_DIR: str = "./model_training/doc2vec_model"
    VECTOR_SIZE = 50
    TRAINING_WINDOW = 3
    TRAINING_EPOCHS = 100
    TRAINING_MIN_COUNT = 3
    TRAINING_WORKERS = 10

os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

LOG_NAME = f"doc2vec_model_{TRAINING_EPOCHS}"
LOG_PATH = OUTPUT_CSV_DIR
logger = custom_logger.default_logger(name=LOG_NAME, log_dir=LOG_PATH)
logger.info(f"VECTOR_SIZE: {VECTOR_SIZE}")
logger.info(f"TRAINING_WINDOW: {TRAINING_WINDOW}")
logger.info(f"TRAINING_EPOCHS: {TRAINING_EPOCHS}")
logger.info(f"TRAINING_MIN_COUNT: {TRAINING_MIN_COUNT}")
logger.info(f"TRAINING_WORKERS: {TRAINING_WORKERS}")

def load_tokenized_data(input_filepath: str) -> List[TaggedDocument]:
    logger.info(f"Loading Tokenized Titles From {input_filepath}")
    documents = []
    basename = os.path.basename(input_filepath)

    with open(input_filepath, "r", encoding="utf-8") as in_file:
        reader = csv.reader(in_file)
        for i, row in enumerate(reader):
            tokens = row[1:]
            tag = f"{basename}_{i}"
            documents.append(TaggedDocument(words=tokens, tags=[tag]))

    return documents

def data_self_extend(docs: List[TaggedDocument], target_size: int) -> List[TaggedDocument]:
    res = []
    while target_size > 0:
        append_len = min(target_size, len(docs))
        res.extend(docs[:append_len])
        target_size -= append_len
    return res

def create_model(
    vector_size: int = VECTOR_SIZE,
    window: int = TRAINING_WINDOW,
    min_count: int = TRAINING_MIN_COUNT,
    workers: int = TRAINING_WORKERS,
    epochs: int = TRAINING_EPOCHS
) -> Doc2Vec:
    logger.info(f"Creating Model")
    return Doc2Vec(
        vector_size=vector_size, 
        window=window,
        min_count=min_count, 
        workers=workers, 
        epochs=epochs,
        dm =0
    )
 
def train_doc2vec(documents: List[TaggedDocument], model: Doc2Vec) -> Doc2Vec:
    logger.info(f"Start Training")
    tqdm_bar = tqdm(total=TRAINING_EPOCHS, desc="Training Epochs")
    epoch_logger = EpochLogger(tqdm_bar)

    model.build_vocab(documents)
    model.train(documents, 
                total_examples=len(documents), 
                epochs=TRAINING_EPOCHS,
                callbacks=[epoch_logger])
    
    tqdm_bar.close()
    return model

def save_trained_model(model: Doc2Vec, output_filepath: str):
    logger.info(f"Saving Model")
    model.save(output_filepath)

def eval_model(documents: List[TaggedDocument], model: Doc2Vec):
    logger.info(f"Evaluating")
    count_of_self = 0
    count_of_second = 0
    total = len(documents)

    embeddings = []
    for i, doc in enumerate(documents):
        if not SAMPLE_MODE and (i % 20000 == 0 or i == total - 1):
            logger.info(f"{i}/{total}")

        inferred_vector = model.infer_vector(documents[i].words)
        embeddings.append([inferred_vector])

    for i, vector in enumerate(embeddings):
        top_sims = model.dv.most_similar(vector, topn=2)

        if top_sims[0][0] == documents[i].tags[0]:
            count_of_self += 1
            count_of_second += 1
        elif top_sims[1][0] == documents[i].tags[0]:
            count_of_second += 1

    self_similarity = count_of_self / total * 100
    second_self_similarity = count_of_second / total * 100

    logger.info(f"Self Similarity {self_similarity:.3f} %")
    logger.info(f"Second Self Similarity {second_self_similarity:.3f} %")

def start_custom_d2v():
    model = create_model()

    logger.info("============== Start Counting ==============")
    max_count = 0
    for filename in os.listdir(INPUT_CSV_DIR):
        if filename.endswith(".csv"):
            input_filepath = os.path.join(INPUT_CSV_DIR, filename)
            max_count = max(max_count, len(load_tokenized_data(input_filepath)))

    # training
    logger.info("============== Start Training ==============")
    documents = []
    for filename in os.listdir(INPUT_CSV_DIR):
        if filename.endswith(".csv"):
            input_filepath = os.path.join(INPUT_CSV_DIR, filename)
            extended_docs = data_self_extend(
                load_tokenized_data(input_filepath), 
                max_count
            )
            documents.extend(extended_docs)

    model = train_doc2vec(documents, model)
    output_filepath = os.path.join(OUTPUT_CSV_DIR, "custom_d2v_model.bin")
    save_trained_model(model, output_filepath)

    # evaluating 
    logger.info("============== Start Evaluating ==============")
    for filename in os.listdir(INPUT_CSV_DIR):
        if filename.endswith(".csv"):
            logger.info(f"Evaluating {filename}")
            # input_filepath = os.path.join(INPUT_CSV_DIR, filename)
            # documents = load_tokenized_data(input_filepath)
    eval_model(documents, model)

if __name__ == "__main__":
    start_custom_d2v()