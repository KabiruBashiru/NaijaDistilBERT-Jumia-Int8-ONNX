import streamlit as st
import torch
import numpy as np
import time 
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

# --- Page Configuration ---
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Sentiment Analysis on Naija Pidgin")
st.markdown("Enter product review below to analyze sentiment using our state-of-the-art model")

# --- Model Configuration ---
# MODEL_ID = "BinBashir/Q_ONNX_distil_NaijaBERT_on_jumia_dataset"
# MODEL_ID = "BinBashir/Q_ONNX_TinyNaijaBERT_on_jumia_dataset"
# MODEL_ID = "BinBashir/Q_ONNX_Mini_NaijaBERT_on_jumia_dataset"

MODEL_ID = "BinBashir/NaijaDistilBERT-Jumia-Int8-ONNX"



@st.cache_resource
def get_model():
    """
    Loads the Tokenizer and the ONNX Model.
    """
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,
                                            #    force_download=True
                                               )

    # Load ONNX Model
    provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
    
    model = ORTModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        provider=provider
        # force_download=True
    )
    
    return tokenizer, model

# --- Load Model ---
try:
    with st.spinner(f"Loading {MODEL_ID} Model..."):
        tokenizer, model = get_model()
except Exception as e:
    st.error(f"Error loading model from Hugging Face. Please check the Model ID.\nDetails: {e}")
    st.stop()

# --- User Input ---
user_input = st.text_area('Enter Text to Analyze')

button = st.button("Analyze Sentiment", type="primary", use_container_width=True)

# Label Mapping
d = {
    0: 'NEUTRAL 😐',
    1: 'POSITIVE 😊',
    2: 'NEGATIVE 😞'
}

# --- Inference Logic ---
if user_input and button:
    # 1. Tokenize Input
    inputs = tokenizer(
        user_input, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors='pt'
    )

    # 2. Inference & Timing
    # Start the timer
    start_time = time.time()

    with torch.no_grad():
        output = model(**inputs)

    # Stop the timer
    end_time = time.time()
    inference_time = end_time - start_time

    # 3. Process Results
    logits = output.logits
    
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
        
    st.write("Logits:", logits)
    
    # Get prediction
    y_pred = int(np.argmax(logits, axis=1)[0])
    
    # Display Result
    st.success(f"Prediction: {d[y_pred]}")
    
    # Display Inference Speed
    st.info(f"⚡ Inference Speed: {inference_time:.4f} seconds")