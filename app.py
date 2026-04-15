

  import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import json
import urllib.request

# --- Page Configuration ---
st.set_page_config(page_title="AI Vision Pro", layout="centered")

# --- CSS for User Friendly UI ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Title and Header ---
st.title("📸 Smart Vision AI")
st.write("Upload an image and let the AI tell you what it is! (No API, No Cost)")
st.divider()

# --- Load Pretrained Model (MobileNet V2) ---
@st.cache_resource
def load_model():
    # Model ko local download karke load karta hai
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    return model

model = load_model()

# --- Load Labels (ImageNet classes) ---
@st.cache_resource
def get_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_class_index.json"
    response = urllib.request.urlopen(url)
    return json.load(response)

labels = get_labels()

# --- Image Preprocessing ---
def predict(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_t = preprocess(image)
    batch_t = torch.unsqueeze(img_t, 0)
    
    with torch.no_grad():
        output = model(batch_t)
    
    # Get the index of the highest score
    _, index = torch.max(output, 1)
    percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
    
    label = labels[str(index.item())][1]
    confidence = percentage[index[0]].item()
    
    return label.replace('_', ' ').title(), confidence

# --- UI Sidebar ---
st.sidebar.header("About App")
st.sidebar.info("Ye app local Machine Learning model use karti hai. Is mein koi external API use nahi hui, is liye data private rehta hai.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Two columns for better UI
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    with col2:
        st.write("### AI Analysis")
        if st.button('Identify Object'):
            with st.spinner('Thinking...'):
                result, score = predict(image)
                st.success(f"**Result:** {result}")
                st.metric(label="Confidence Score", value=f"{score:.2f}%")
                
                if score > 70:
                    st.balloons()
                else:
                    st.warning("Confidence is a bit low, but that's my best guess!")
 

 
