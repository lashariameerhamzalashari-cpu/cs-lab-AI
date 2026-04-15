import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import json
import urllib.request

# 1. Page settings (Hamesha top par honi chahiye)
st.set_page_config(page_title="AI Image Classifier", layout="centered")

# 2. Model load karne ka function
@st.cache_resource
def load_model():
    # Pretrained MobileNet V2 model (Free and local)
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    return model

# 3. Model aur Labels ko tayyar karna
model = load_model()

@st.cache_resource
def get_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_class_index.json"
    response = urllib.request.urlopen(url)
    return json.load(response)

labels = get_labels()

# 4. User Interface (UI)
st.title("🤖 Smart AI Vision")
st.write("Apni image upload karein aur AI usay pehchan lega.")

uploaded_file = st.file_uploader("Image select karein...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Aap ki image', use_container_width=True)
    
    # Prediction logic
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_t = preprocess(image)
    batch_t = torch.unsqueeze(img_t, 0)
    
    with st.spinner('AI soch raha hai...'):
        with torch.no_grad():
            output = model(batch_t)
        
        _, index = torch.max(output, 1)
        percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
        result = labels[str(index.item())][1]
        
        st.success(f"Mujhe lagta hai ye **{result.replace('_', ' ')}** hai!")
        st.write(f"Confidence: {percentage[index[0]].item():.2f}%")
