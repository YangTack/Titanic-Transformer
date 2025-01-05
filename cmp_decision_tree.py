import numpy as np
from dataset import TitanicDataset
from sklearn.tree import DecisionTreeClassifier
import pickle

model = DecisionTreeClassifier()
dataset_train = TitanicDataset("./data", scope="TRAIN")
dataset_val = TitanicDataset("./data", scope="VAL")

x_train, y_train = map(np.stack, zip(*map(lambda x: (x[0].cpu().numpy(), x[1].cpu().numpy()), iter(dataset_train))))
x_val, y_val = map(np.stack, zip(*map(lambda x: (x[0].cpu().numpy(), x[1].cpu().numpy()), iter(dataset_val))))
model.fit(x_train, y_train)
print(model.score(x_val, y_val))

pkl_filename = "cmp_decision_tree_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)
