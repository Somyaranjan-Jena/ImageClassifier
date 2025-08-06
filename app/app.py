import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os

def load_model(model_path, num_classes):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_image(image, model, classes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    device = next(model.parameters()).device
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]

# Streamlit UI
st.title("🐶 Cat vs. Dog Classifier")
st.write("Upload an image to classify it as a cat or dog.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        model_path = "models/best_model.pth"  # adjusted path if app.py is at root
        classes = ['cat', 'dog']
        model = load_model(model_path, len(classes))
        prediction = predict_image(image, model, classes)
        st.success(f"**Prediction**: {prediction.capitalize()}")
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
