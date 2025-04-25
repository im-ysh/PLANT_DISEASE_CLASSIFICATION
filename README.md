# PLANT_DISEASE_DETECTION


---

# 🌿 Plant Disease Detection using Custom CNN

This project presents a **Convolutional Neural Network (CNN)**-based approach to classify plant diseases from leaf images. It’s built using **PyTorch** and trained on a dataset consisting of healthy and diseased leaves of Apple, Corn, and Grape.

---

## 📁 Dataset

The dataset is divided into:
- **Train Folder**: Contains training images, categorized into subfolders based on class names.
- **Test Folder**: Contains unseen test images for evaluating model performance.

### 🏷️ Class Labels
```python
['Apple___Apple_scab', 
 'Apple___Black_rot', 
 'Apple___Cedar_apple_rust', 
 'Apple___Healthy', 
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
 'Corn_(maize)___Northern_Leaf_Blight', 
 'Grape___Black_rot']
```

---

## 🧠 Model Architecture

A custom CNN model was designed with:
- **2 Convolutional Layers**: Each followed by ReLU activation and MaxPooling.
- **Dropout Layer**: Applied after the first fully connected layer to prevent overfitting.
- **2 Fully Connected Layers**: Final output mapped to the number of disease categories.

---

## 🔧 Hyperparameters

| Parameter         | Value         |
|------------------|---------------|
| Image Size       | 150x150       |
| Batch Size       | 32            |
| Learning Rate    | 0.001         |
| Epochs           | 5             |
| Optimizer        | Adam          |
| Loss Function    | CrossEntropy  |

---

## 🔄 Workflow

### 📦 Preprocessing
- Images resized to 150x150
- Normalized using ImageNet means and std
- Converted to tensors using torchvision transforms

### 🚂 Training
- Model trained for 5 epochs
- Tracked training loss every 100 batches

### 🧪 Evaluation
- Model tested on unseen images from the test set
- Final accuracy reported as:
  ```
  Accuracy of the network on the test images: 75%
  ```

---

## 📊 Visualization

Utility functions are included to:
- Display random images with predicted labels
- Plot dataset distribution

Example:
```python
display_random_image(train_loader, class_names)
```

---
---

## 📦 Libraries Used

This project utilizes the following Python libraries:

- **PyTorch** – For building and training the custom CNN  
- **Torchvision** – For loading datasets and applying image transformations  
- **NumPy** – For numerical operations  
- **Matplotlib** – For data visualization  
- **PIL (Pillow)** – For image loading and processing  
- **os** – For file path handling  
- **random** – For random selection of images  
- **Google Colab** (optional) – For GPU-accelerated training in the cloud  

---



## ▶️ How to Run

1. Clone or download the repository
2. Upload dataset to the mentioned path (Google Drive or local directory)
3. Open `PLANT_DISEASE_CLASSIFICATION 
.ipynb` in Google Colab or Jupyter Notebook
4. Run cells sequentially

---

## 🏆 Results

- **Test Accuracy Achieved**: ~75%
- **Dataset Size**: 106 training, 106 testing images
- Balanced training observed with no severe overfitting

---

## 🙏 Acknowledgements

- PyTorch & Torchvision
- PlantVillage dataset (via Kaggle)
- Google Colab for training support

---
