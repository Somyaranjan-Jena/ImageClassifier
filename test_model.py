import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os
import time  # for measuring inference time

def load_model(model_path, num_classes):
    # ⬇ Force CPU for Heroku or testing (optional)
    device = torch.device("cuda" if torch.cuda.is_available() and not st._is_running_with_streamlit else "cpu")

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
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

    # ⏱ Start timing
    start_time = time.time()

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # ⏱ End timing
    inference_time = time.time() - start_time

    return classes[predicted.item()], inference_time

# Streamlit UI
st.title("🐶 Cat vs. Dog Classifier")
st.write("Upload an image to classify it as a cat or dog.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        model_path = "models/best_model.pth"
        classes = ['cat', 'dog']
        model = load_model(model_path, len(classes))
        prediction, inf_time = predict_image(image, model, classes)

        st.success(f"**Prediction**: {prediction.capitalize()} (Inference time: {inf_time:.3f} seconds)")
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
