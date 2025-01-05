from math import nan
from pathlib import Path
import re
from typing import Literal
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.utils.data.dataset import ConcatDataset

class TitanicDataset(Dataset):

    def __init__(self, path: Path | str, scope: Literal["TRAIN", "VAL", "ALL"] = "TRAIN"):
        super().__init__()
        self.scope: Literal["TRAIN", "VAL", "ALL"] = scope
        if isinstance(path, str):
            path = Path(path)
        self.data = pd.read_csv(path / "train.csv").loc[:, ["Survived", "Name", "Age", "Pclass", "Sex", "SibSp", "Parch", "Embarked", "Ticket", "Fare", "Cabin"]]
        Title_Dictionary = {
            "Capt": 0,
            "Col": 0,
            "Major": 0,
            "Jonkheer": 1,
            "Don": 1,
            "Sir" : 1,
            "Dr": 0,
            "Rev": 0,
            "the Countess":1,
            "Mme": 2,
            "Mlle": 3,
            "Ms": 2,
            "Mr" : 4,
            "Mrs" : 2,
            "Miss" : 3,
            "Master" : 5,
            "Lady" : 1,
            "Dona": 1,
        }

        self.data['Title'] = self.data['Name'].map(lambda x:(re.compile(r",(.+?)\.").search(x).group(1)).strip()) # type: ignore
        self.data['Title'] = self.data['Title'].map(Title_Dictionary)
        grouped = self.data.groupby(['Sex','Pclass', 'Title'], group_keys=False)  
        self.data["Age"] = grouped["Age"].apply(lambda x: x.fillna(x.median()))
        self.data['Embarked'] = self.data['Embarked'].fillna(self.data['Embarked'].mode().iloc[0])
        self.data['Fare'] = self.data['Fare'].fillna(self.data['Fare'].median())
        self.data['Cabin'] = self.data['Cabin'].fillna('U')
        self.data = self.data.drop(["Name", "Ticket"], axis=1)
        self.data["Cabin"] = self.data["Cabin"].apply(lambda x: x[0])
        self.data["Cabin"] = self.data["Cabin"].map({"U": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8})
        self.data["Embarked"] = self.data["Embarked"].map({"S": 0, "C": 1, "Q": 2})
        self.data["Sex"] = self.data["Sex"].map({"male": 0, "female": 1})
        if not (path / "indices.npy").exists():
            np.save(path / "indices.npy", np.random.permutation(self.data.shape[0]))

        indices = np.load(path / "indices.npy")
        self.np_data = self.data.iloc[:, 1:].to_numpy(dtype=np.float32)
        self.np_target = self.data.iloc[:, 0].to_numpy(dtype=np.int_)
        self.train_indices = indices[:int(len(indices)*0.8)]
        self.val_indices = indices[int(len(indices)*0.8):]
        self.train_data_np = self.np_data[self.train_indices]
        self.train_result_np = self.np_target[self.train_indices]
        self.val_data_np = self.np_data[self.val_indices]
        self.val_result_np = self.np_target[self.val_indices]
            
    def _fill_missing_ages(self, data: pd.DataFrame) -> None:
        ages = data["Age"].loc[data["Age"].notna()]
        age_mean = ages.mean()
        data.loc[data["Age"].isna(), "Age"] = age_mean

    def _change_sex(self, data: pd.DataFrame) -> None:
        data.loc[data["Sex"] == "male", "Sex"] = 1
        data.loc[data["Sex"] == "female", "Sex"] = 0

    def _change_type(self, data: pd.DataFrame) -> None:
        data["Age"] = data["Age"].astype(float)
        data["Pclass"] = data["Pclass"].astype(int)
        data["SibSp"] = data["SibSp"].astype(int)
        data["Parch"] = data["Parch"].astype(int)
        data["Survived"] = data["Survived"].astype(int)
        data["Sex"] = data["Sex"].astype(int)

    def __len__(self) -> int:
        if self.scope == "TRAIN":
            return self.train_data_np.shape[0] - 1
        elif self.scope == "VAL":
            return self.val_data_np.shape[0] - 1
        else:
            return self.np_data.shape[0] - 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.scope == "TRAIN":
            result = torch.from_numpy(self.train_data_np[idx]).to(torch.float32), torch.tensor(self.train_result_np[idx]).to(torch.int)
        elif self.scope == "VAL":
            result = torch.from_numpy(self.val_data_np[idx]).to(torch.float32), torch.tensor(self.val_result_np[idx]).to(torch.int)
        else:
            result = torch.from_numpy(self.np_data[idx]).to(torch.float32), torch.tensor(self.np_target[idx]).to(torch.int)
        return result

dt = TitanicDataset(Path("./data"))

MU: np.ndarray
STD: np.ndarray

def _titanic_mu_std():
    global MU, STD
    np.random.seed(42)
    dataset = TitanicDataset(Path("./data"))
    MU = dataset.train_data_np.mean(0)
    STD = dataset.train_data_np.std(0)
    
_titanic_mu_std()