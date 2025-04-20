from typing import List
import torch
from gensim.models.doc2vec import Doc2Vec
from model_training.constants import CLASSIFY_MODEL_PATH
import model_training.tokenizer as tokenizer
from model_training.classifier import Classifier, embed_data
from model_training.constants import D2V_MODEL_PATH, BOARDS

def get_embedding_model() -> Doc2Vec:
    model = Doc2Vec.load(D2V_MODEL_PATH)
    return model

def get_classify_model(embedding_model: Doc2Vec, sorted_boards: List[str]) -> Classifier:
    model = Classifier(embedding_model.vector_size, len(sorted_boards))
    state_dict = torch.load(CLASSIFY_MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def get_sorted_boards() -> List[str]:
    return sorted(BOARDS, key=lambda x: x.lower())

def clean_title(title: str) -> str:
    cleaned = title.strip()
    cleaned_lower = cleaned.lower()
    if cleaned_lower.startswith("re:") or cleaned_lower.startswith("fw:"):
        return cleaned_lower[3:]
    return cleaned_lower

def tokenize(title: str) -> List[List[str]]:
    return tokenizer.tokenize_and_remove_stopwords([title])

def embed(d2v_model: Doc2Vec, tokenized_title: List[List[str]]) -> List[List[int]]:
    return embed_data(d2v_model, tokenized_title)

def predict(classify_model: Classifier, sorted_boards: List[str], embedded_title: List[List[int]]) -> str:
    X = torch.tensor(embedded_title, dtype=torch.float32)
    predict_board = "Unknown"
    with torch.no_grad():
        outputs = classify_model(X)
        _, predicted_index = torch.max(outputs, 1)
        print(predicted_index)
        print(sorted_boards)

        predict_board = sorted_boards[predicted_index]
    return predict_board

print(get_sorted_boards())