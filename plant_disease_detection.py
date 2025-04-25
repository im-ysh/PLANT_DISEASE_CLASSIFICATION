import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Define class names (same order as during training!)
class_names = ['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust',
               'Apple___Healthy','Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
               'Corn_(maize)___Northern_Leaf_Blight','Grape___Black_rot']

# CNN Model (same as your training code)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.flattened_size = self._get_flattened_size()
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, len(class_names))

    def _get_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, 150, 150)
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            return x.view(-1).size(0)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, self.flattened_size)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load trained model
model = CNN()
model.load_state_dict(torch.load("PLANT_DISEASE_DETECTION_model.pth", map_location=torch.device('cpu')))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Streamlit UI
st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to predict its disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    # Preprocess image
    img_tensor = transform(image).unsqueeze(0)  # Add batch dim

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        prediction = class_names[predicted.item()]

    st.success(f"🩺 Predicted Disease: **{prediction}**")
