import torch
from torch.utils.data import Dataset
from typing import Tuple
import random
import numpy as np
import requests
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#### LOADING THE MODEL
from torchvision.models import resnet18

model = resnet18(weights=None)
model.fc = torch.nn.Linear(512, 44)

ckpt = torch.load("./01_MIA_67.pt", map_location="cpu")

model.load_state_dict(ckpt)

#### DATASETS
class TaskDataset(Dataset):
    def __init__(self, transform=None):

        self.ids = []
        self.imgs = []
        self.labels = []

        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if not self.transform is None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)

class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int, int]:
        id_, img, label = super().__getitem__(index)
        membership = self.membership[index]
        if membership is None:
            membership = -1
        return id_, img, label, membership

model.eval()

data: MembershipDataset = torch.load("./priv_out.pt")
# Set model to evaluation mode

data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)

# Move model to appropriate device
device = torch.device("cuda")
model.to(device)

# Initialize dictionary to store predictions and IDs
priv_predictions = {}

# Run data through the model
for ids, images, labels, membership in data_loader:

    images = images.to(device)
    model_outs = model(images)
    
    # Apply softmax to model outputs
    softmax_outputs = torch.nn.functional.softmax(model_outs, dim=1)

    # Find the maximum probability for each image
    max_probabilities, _ = torch.max(softmax_outputs, dim=1) 

    memberships = torch.where(max_probabilities > 0.70, max_probabilities, torch.tensor(0))
    data.membership[data.ids.index(ids.item())] = memberships.item() 
    
df = pd.DataFrame(
    {
        "ids": data.ids,
        "score": data.membership,
    }
)
df.to_csv("test.csv", index=None)
response = requests.post("http://34.71.138.79:9090/mia", files={"file": open("test.csv", "rb")}, headers={"token": "93989180"})
print(response.json())
