import streamlit as st
import torch
import time
import numpy as np
from transformers import AutoTokenizer  # Changed from BertTokenizer to AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

# --- Page Configuration ---
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Sentiment Analysis on Naija Pidgin")
st.markdown("Enter product review below to analyze sentiment using our state-of-the-art ONNX model.")

# --- Model Configuration ---
MODEL_ID = "BinBashir/Q4_Naija-BERT_on_jumia_dataset"

@st.cache_resource
def get_model():
    # Switch to AutoTokenizer - it is safer and automatically detects the architecture
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Load the ONNX model
    # ORTModel handles CPU/GPU providers internally
    model = ORTModelForSequenceClassification.from_pretrained(MODEL_ID)
    
    return tokenizer, model

# Load model (cached)
try:
    with st.spinner("Loading ONNX Model..."):
        tokenizer, model = get_model()
except Exception as e:
    st.error(f"Error loading model. Details: {e}")
    st.stop()

# --- User Input ---
user_input = st.text_area('Enter Text to Analyze', placeholder="E.g., This phone dey work well well!")

button = st.button("Analyze Sentiment", type="primary", use_container_width=True)

# Define labels (Ensure these match your specific model's training labels)
labels_map = {
    0: 'NEUTRAL 😐',
    1: 'POSITIVE 😊',
    2: 'NEGATIVE 😞'
}

# --- Main Logic ---
if user_input and button:
    # 1. Tokenize
    inputs = tokenizer(user_input, padding=True, truncation=True, max_length=512, return_tensors='pt')
    
    # 2. Inference & Timing
    st.markdown("### Results")
    
    # Start Timer
    start_time = time.perf_counter()
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Stop Timer
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    
    # 3. Process Results
    logits = outputs.logits
    
    # Apply Softmax to get probabilities (Confidence Score)
    probabilities = torch.softmax(logits, dim=1)
    
    # Get the predicted class index (0, 1, or 2)
    predicted_class_id = torch.argmax(probabilities, dim=1).item()
    
    # Get the confidence score for that specific class
    confidence_score = probabilities[0][predicted_class_id].item()
    
    # Map ID to Label
    prediction_label = labels_map.get(predicted_class_id, "UNKNOWN")

    # 4. Display Metrics in Columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Prediction", prediction_label)
        
    with col2:
        st.metric("Confidence", f"{confidence_score:.2%}")
        
    with col3:
        st.metric("Inference Time", f"{inference_time:.4f} sec")

    # Optional: Debug view
    with st.expander("View Raw Logits"):
        st.write("Logits:", logits.numpy())
        st.write("Probabilities:", probabilities.numpy())