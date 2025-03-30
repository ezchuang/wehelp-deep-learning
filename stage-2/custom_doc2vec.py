import os
import csv
from typing import List
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

SAMPLE_MODE = True

if SAMPLE_MODE:
    INPUT_CSV_DIR: str = "./stage-2/tokenized_data_sample"
    OUTPUT_CSV_DIR: str = "./stage-2/doc2vec_model_sample"
    VECTOR_SIZE = 100
    TRAINING_WINDOW = 5
    TRAINING_EPOCHS = 200
    TRAINING_MIN_COUNT = 2
    TRAINING_WORKERS = 10
else:
    INPUT_CSV_DIR: str = "./stage-2/tokenized_data"
    OUTPUT_CSV_DIR: str = "./stage-2/doc2vec_model"
    VECTOR_SIZE = 100
    TRAINING_WINDOW = 5
    TRAINING_EPOCHS = 200
    TRAINING_MIN_COUNT = 3
    TRAINING_WORKERS = 10

os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

def load_tokenized_data(input_filepath: str) -> List[TaggedDocument]:
    print(f"Loading Tokenized Titles")
    documents = []

    with open(input_filepath, "r", encoding="utf-8") as in_file:
        reader = csv.reader(in_file)
        for i, row in enumerate(reader):
            tokens = row[1:]
            documents.append(TaggedDocument(words=tokens, tags=[i]))

    print(f"Tokenized Titles Ready")
    return documents

def create_model(
    vector_size: int = VECTOR_SIZE,
    window: int = TRAINING_WINDOW,
    min_count: int = TRAINING_MIN_COUNT,
    workers: int = TRAINING_WORKERS,
    epochs: int = TRAINING_EPOCHS
) -> Doc2Vec:
    print(f"Creating Model")
    return Doc2Vec(
        vector_size=vector_size, 
        window=window,
        min_count=min_count, 
        workers=workers, 
        epochs=epochs
    )

def train_doc2vec(documents: List[TaggedDocument], model: Doc2Vec) -> Doc2Vec:
    print(f"Start Training")
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    return model

def save_trained_model(model: Doc2Vec, output_filepath: str):
    print(f"Saving Model")
    model.save(output_filepath)

def eval_model(documents: List[TaggedDocument], model: Doc2Vec):
    print(f"Evaluating")
    count_of_self = 0
    count_of_second = 0
    total = len(documents)

    for i in range(len(documents)):
        if SAMPLE_MODE and i % 100 == 0:
            print(i)
        if not SAMPLE_MODE and i % 10000 == 0:
            print(i)

        inferred_vector = model.infer_vector(documents[i].words)
        top_sims = model.dv.most_similar([inferred_vector], topn=2)

        if top_sims[0][0] == i:
            count_of_self += 1
            count_of_second += 1
        elif top_sims[1][0] == i:
            count_of_second += 1

    self_similarity = count_of_self / total * 100
    second_self_similarity = count_of_second / total * 100

    print(f"Self Similarity {self_similarity:.3f} %")
    print(f"Second Self Similarity {second_self_similarity:.3f} %")

def start_custom_d2v():
    for filename in os.listdir(INPUT_CSV_DIR):
        if filename.endswith(".csv"):
            print("============== Start Training ==============")
            input_filepath = os.path.join(INPUT_CSV_DIR, filename)
            output_filepath = os.path.join(OUTPUT_CSV_DIR, filename.replace("tokenized", "d2v").replace(".csv", ".bin"))

            print(f"Processing {input_filepath} -> {output_filepath}")
            documents = load_tokenized_data(input_filepath)

            model = create_model()
            model = train_doc2vec(documents, model)
            save_trained_model(model, output_filepath)

            eval_model(documents, model)

if __name__ == "__main__":
    start_custom_d2v()