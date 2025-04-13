import csv
import os
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from gensim.models.doc2vec import Doc2Vec
import custom_logger

D2V_MODEL_PATH = "./stage-2/doc2vec_model/custom_d2v_model.bin"
INPUT_CSV_DIR: str = "./stage-2/tokenized_data"
EMBEDDED_DATA_PATH = "./stage-2/embedded_data/embedded_data.csv"
CLASSIFY_MODEL_SAVE_PATH = "./stage-2/classify_models/classify_model.pth"

TRAINING_EPOCHS = 200
LEARNING_RATE = 0.001
NN_HIDDEN_DIM_1 = 200
NN_HIDDEN_DIM_2 = 200
NN_HIDDEN_DIM_3 = 200
NN_HIDDEN_DIM_4 = 200
BATCH_SIZE = 100000

LOG_NAME = f"classify_model_{TRAINING_EPOCHS}"
LOG_PATH = "./stage-2/classify_models"
logger = custom_logger.default_logger(name=LOG_NAME, log_dir=LOG_PATH)
logger.info(f"TRAINING_EPOCH: {TRAINING_EPOCHS}")
logger.info(f"LEARNING_RATE: {LEARNING_RATE}")
logger.info(f"NN_HIDDEN_DIM_1: {NN_HIDDEN_DIM_1}")
logger.info(f"NN_HIDDEN_DIM_2: {NN_HIDDEN_DIM_2}")
logger.info(f"NN_HIDDEN_DIM_3: {NN_HIDDEN_DIM_3}")
logger.info(f"BATCH_SIZE: {BATCH_SIZE}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def load_all_data() -> Tuple[List[str], List[List[str]]]:
    all_labels = []
    all_tokens = []

    for filename in os.listdir(INPUT_CSV_DIR):
        if filename.endswith(".csv"):
            logger.info(f"Loading {filename}")
            input_filepath = os.path.join(INPUT_CSV_DIR, filename)
            with open(input_filepath, "r", encoding="utf-8") as in_file:
                reader = csv.reader(in_file)
                for _, row in enumerate(reader):
                    all_labels.append(row[0])
                    all_tokens.append(row[1:])
    return all_labels, all_tokens

def load_embedded_result(csv_filepath: str) -> List[List[float]]:
    result = []
    logger.info(f"Loading Embedded Data")
    with open(csv_filepath, "r", encoding="utf-8") as in_file:
        reader = csv.reader(in_file)
        result.extend([list(map(float, row)) for row in reader if row])
    logger.info(f"Embedded data Loaded")
    return result

def save_embedded_result(csv_filepath: str, embeddings: List[List[float]]) -> None:
    os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
    with open(csv_filepath, "w", newline='', encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        for row in embeddings:
            writer.writerow(row)
    logger.info(f"Embedded Data Saved")

def encode_labels(labels: List[str]) -> Tuple[List[int], dict]:
    unique_labels = sorted(set(labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = [label2id[label] for label in labels]
    logger.info(f"Label Mapping: {label2id}")
    return encoded_labels, label2id

def embed_data(d2v_model: Doc2Vec, all_tokens: List[List[str]]) -> List[List[int]]:
    embeddings = []
    logger.info(f"Start Embedding Data")
    for i, tokens in enumerate(all_tokens):
        if i % 20000 == 0 or i == len(all_tokens) - 1:
            logger.info(f"{i}/{len(all_tokens)}")
        vec = d2v_model.infer_vector(tokens)
        embeddings.append(vec)
    return embeddings

class Classifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(Classifier, self).__init__()
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(input_dim, NN_HIDDEN_DIM_1)
        self.fc2 = nn.Linear(NN_HIDDEN_DIM_1, NN_HIDDEN_DIM_2)
        self.fc3 = nn.Linear(NN_HIDDEN_DIM_2, NN_HIDDEN_DIM_3)
        self.fc4 = nn.Linear(NN_HIDDEN_DIM_3, NN_HIDDEN_DIM_4)
        self.fc5 = nn.Linear(NN_HIDDEN_DIM_4, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        return out

def start_classify():
    d2v_model = Doc2Vec.load(D2V_MODEL_PATH)

    logger.info("============== Start Classifying ==============")
    all_labels, all_tokens = load_all_data()

    encoded_labels, label2id = encode_labels(all_labels)
    num_classes = len(label2id)

    embeddings = []
    if os.path.exists(EMBEDDED_DATA_PATH):
        embeddings = load_embedded_result(EMBEDDED_DATA_PATH)
    else:
        embeddings.extend(embed_data(d2v_model, all_tokens))
        save_embedded_result(EMBEDDED_DATA_PATH, embeddings)

    X = torch.tensor(embeddings, dtype=torch.float32).to(device)
    y = torch.tensor(encoded_labels, dtype=torch.long).to(device)

    dataset = TensorDataset(X, y)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = d2v_model.vector_size
    hidden_dim = NN_HIDDEN_DIM_1
    output_dim = num_classes
    model = Classifier(input_dim, hidden_dim, output_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    num_epochs = TRAINING_EPOCHS
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        total_train = 0
        correct_train = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)
            total_train += y_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == y_batch).sum().item()

        avg_loss = epoch_loss / total_train
        train_accuracy = correct_train / total_train
        
        # Evaluating
        model.eval()
        total_test = 0
        correct_test = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total_test += y_batch.size(0)
                correct_test += (predicted == y_batch).sum().item()
        test_accuracy = correct_test / total_test

        logger.info(f"Epoch [{epoch+1}/{TRAINING_EPOCHS}]")
        logger.info(f"Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy*100:.2f}%, Test Accuracy: {test_accuracy*100:.2f}%")

    os.makedirs(os.path.dirname(CLASSIFY_MODEL_SAVE_PATH), exist_ok=True)
    # torch.save(model.state_dict(), CLASSIFY_MODEL_SAVE_PATH)
    logger.info(f"Save Classify Model To: {CLASSIFY_MODEL_SAVE_PATH}")

if __name__ == "__main__":
    start_classify()