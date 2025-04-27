import streamlit as st
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Define the CNN model architecture
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
        self.fc2 = nn.Linear(128, 7)  # 7 classes

    def _get_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, 150, 150)
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.pool(x)
            return x.view(-1).size(0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, self.flattened_size)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define model path
model_path = "streamlit-app/PLANT_DISEASE_DETECTION_model.pth" 

# Streamlit UI and image processing
st.title("🍀 Plant Disease Detection App")
st.markdown("Upload a **plant leaf image** below, and I'll predict its disease class! 🌟")

uploaded_file = st.file_uploader("📤 Upload Image Here...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Check if the model file exists before attempting to load it
        if not os.path.exists(model_path):
            st.error("❌ Model file not found. Please upload the model file first!")
            st.stop()

        # Load the model
        model = CNN()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        # Define class names
        class_names = [
            'Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___Healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Grape___Black_rot'
        ]

        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Open and process image
        image = Image.open(uploaded_file)
        image = image.convert('RGB')

        st.image(image, caption='Uploaded Leaf Image', use_column_width=True)

        # Preprocess
        img = transform(image).unsqueeze(0)  # Add batch dimension

        # Predict
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            prediction = class_names[predicted.item()]

        st.success(f"✅ Predicted Disease: **{prediction}**")

    except FileNotFoundError as fnf_error:
        st.error(f"❌ Error: {fnf_error}")
    except Exception as e:
        st.error(f"❌ Oops! Could not process the image.")
        st.error(f"Error details: {e}")
else:
    st.info("Please upload a leaf image to start prediction.")
