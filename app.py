import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import faiss
import pickle
from PIL import Image

# Load Assets 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model architecture
class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(-1, 128 * 4 * 4)
        embedding = F.relu(self.fc1(x))
        out = self.fc2(embedding)
        return out, embedding

# Load model and data
model = CIFAR10_CNN().to(device)
model.load_state_dict(torch.load('cnn_model.pth', map_location=device))
model.eval()

embeddings = np.load('embeddings.npy')
indices = np.load('indices.npy')
with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

# Dataset (for retrieving matching images)
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# FAISS
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Streamlit UI
st.title("Visual Search Engine")

uploaded_file = st.file_uploader("Upload an image (32x32 or larger)", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB").resize((32, 32))
    st.image(img, caption="Query Image", width=150)

    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output, emb = model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()
        st.success(f"Predicted Class: {classes[pred_class]}")

        emb_np = emb.cpu().numpy()
        D, I = index.search(emb_np, 5)
        matched_ids = [indices[i] for i in I[0]]

        st.markdown("### Top 5 Visually Similar Images:")
        cols = st.columns(5)
        for i, idx in enumerate(matched_ids):
            sim_img, _ = train_dataset[idx]
            sim_img = transforms.ToPILImage()(sim_img)
            cols[i].image(sim_img, use_container_width=True)