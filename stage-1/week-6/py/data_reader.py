import csv
import math
import numpy as np

class ReaderHelper():
    @staticmethod
    def get_mean(data: list[np.ndarray]):
        return np.mean(data, axis=0)

    @staticmethod
    def get_std(data: list[np.ndarray]):
        return np.std(data, axis=0)
    
    @staticmethod
    def get_z_score(data: list[np.ndarray], data_mean: list[np.ndarray], data_std: list[np.ndarray]):
        return (data - data_mean) / data_std

class GenderHeightWeight():
    def __init__(self, file_path: str = "./week-6/resource/gender-height-weight.csv"):
        self.raw: list[np.ndarray] = []
        self.data: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.data_size: int = 0
        self.mean: float = 0.0
        self.std:float = 0.0
        
        self._load_data(file_path)
        self._standardize_data()
        
    def _load_data(self, file_path: str) -> None:
        try:
            with open(file_path, "r", newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    gender = 1 if row['Gender'] == 'Female' else 0
                    height = float(row['Height'])
                    weight = float(row['Weight'])
                    
                    # Store as 2D arrays
                    self.raw.append(np.array([[gender, height, weight]]))
            
            self.data_size = len(self.raw)
            print(f"Loaded {self.data_size} samples from {file_path}")
            
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            self.raw_data = []
            self.data, self.labels = [], []
            self.data_size = 0

    def _standardize_data(self) -> None:
        raw_data = np.array(self.raw)
        np.random.shuffle(raw_data)

        # Compute mean and std and z-score
        data_mean = ReaderHelper.get_mean(raw_data)
        self.mean = data_mean[0][2]
        data_std = ReaderHelper.get_std(raw_data)
        self.std = data_std[0][2]
        data_z_score = ReaderHelper.get_z_score(raw_data, data_mean, data_std)

        for i in range(self.data_size):
            gender = data_z_score[i][0][0]
            height = data_z_score[i][0][1]
            weight = data_z_score[i][0][2]
            self.data.append(np.array([[gender, height]]))
            self.labels.append(np.array([[weight]]))

    def get_training_data(self, x: float):
        proportion = math.ceil(self.data_size * x)
        return self.data[:proportion]

    def get_training_labels(self, x: float):
        proportion = math.ceil(self.data_size * x)
        return self.labels[:proportion]
    
    def get_testing_data(self, x: float):
        proportion = math.ceil(self.data_size * x)
        return self.data[proportion:]

    def get_testing_labels(self, x: float):
        proportion = math.ceil(self.data_size * x)
        return self.labels[proportion:]
    
    def get_training_size_boundary(self, x: float):
        return math.ceil(self.data_size * x)
    

class Titanic():
    def __init__(self, file_path: str = "./week-6/resource/titanic.csv"):
        self.raw: list[np.ndarray] = []
        self.data: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.data_size: int = 0
        
        self._load_data(file_path)
        self._standardize_data()
        
    def _load_data(self, file_path: str) -> None:
        try:
            with open(file_path, "r", newline='') as file:
                reader = csv.DictReader(file)
                ages = [] # contain non-empty ages

                for row in reader:
                    survived = int(row['Survived'])

                    pclass = [0, 0, 0]
                    if row['Pclass'] == "1":
                        pclass = [1, 0, 0]
                    elif row['Pclass'] == "2":
                        pclass = [0, 1, 0]
                    elif row['Pclass'] == "3":
                        pclass = [0, 0, 1]
                    else:
                        pclass = [0.33, 0.33, 0.33] 

                    # pclass = int(row['Pclass'])
                    

                    sex = 1 if row['Sex'] == 'female' else 0

                    try:
                        age = float(row['Age']) if row['Age'] else np.nan
                        if not np.isnan(age):
                            ages.append(age)
                    except ValueError:
                        age = np.nan

                    # sibSp = 1 if int(row['SibSp']) >= 1 else 0
                    # parch = 1 if int(row['Parch']) >= 1 else 0
                    sibSp = int(row['SibSp'])
                    parch = int(row['Parch'])
                    fare = float(row['Fare'])
                    cabin = 1 if row['Cabin'] != '' else 0

                    embarked = [0, 0, 0]
                    if row['Embarked'] == "Q":
                        embarked = [1, 0, 0]
                    elif row['Embarked'] == "C":
                        embarked = [0, 1, 0]
                    elif row['Embarked'] == "S":
                        embarked = [0, 0, 1]
                    else:
                        embarked = [0, 0, 1] 

                    self.raw.append(np.array([[survived, *pclass, sex, age, sibSp, parch, fare, cabin, *embarked]]))
                    # self.labels.append(np.array([[survived]]))
            
            self.data_size = len(self.raw)
            print(f"Loaded {self.data_size} samples from {file_path}")

            if ages:
                mean_age = np.nanmean(ages)
                for i in range(len(self.raw)):
                    if np.isnan(self.raw[i][0, 5]):
                        self.raw[i][0, 5] = mean_age
            
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            self.raw_data = []
            self.data, self.labels = [], []
            self.data_size = 0

    def _standardize_data(self) -> None:
        raw_data = np.array(self.raw)
        np.random.shuffle(raw_data)

        data_need_trans = np.array([row for row in raw_data])
        data_mean = ReaderHelper.get_mean(data_need_trans)
        data_std = ReaderHelper.get_std(data_need_trans)
        data_z_score = ReaderHelper.get_z_score(raw_data, data_mean, data_std)

        self.data = [np.array([row[0][1:]])for row in data_z_score]
        # for i in range(len(raw_data)):
        #     self.data[i][0, 4] = data_z_score[i][0, 5]
            
        # self.data = data_z_score
        self.labels = [np.array([[row[0][0]]]) for row in raw_data]

    def get_training_data(self, x: float):
        proportion = math.ceil(self.data_size * x)
        return self.data[:proportion]

    def get_training_labels(self, x: float):
        proportion = math.ceil(self.data_size * x)
        return self.labels[:proportion]
    
    def get_testing_data(self, x: float):
        proportion = math.ceil(self.data_size * x)
        return self.data[proportion:]

    def get_testing_labels(self, x: float):
        proportion = math.ceil(self.data_size * x)
        return self.labels[proportion:]
    
    def get_training_size_boundary(self, x: float):
        return math.ceil(self.data_size * x)