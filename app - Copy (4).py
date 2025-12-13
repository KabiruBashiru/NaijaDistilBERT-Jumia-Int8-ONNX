import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch



# Set page configuration 
st.set_page_config( 
    page_title="Sentiment Analysis", 
    page_icon="💬", 
    layout="centered" 
)

st.title("💬 Sentiment Analysis on Naija Pidgin") 
st.markdown("Enter product review below to analyze sentiment using our state-of-the-art model")




@st.cache_resource 
def get_model():
  tokenizer = BertTokenizer.from_pretrained('BinBashir/TinyBERT_on_jumia_dataset')
  model = BertForSequenceClassification.from_pretrained("BinBashir/TinyBERT_on_jumia_dataset")
  
  model.eval()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  return tokenizer, model


tokenizer, model = get_model()





user_input = st.text_area('Enter Text to Analyze')


button = st.button("Analyze Sentiment" , type="primary",
                   use_container_width=True)





d = {
  0: 'NEUTRAL😐',
  1: 'POSITIVE 😊',
  2: 'NEGATIVE 😞'
}

if user_input and button:
  test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512, return_tensors='pt')
  # move tensors to the same device as the model
  device = next(model.parameters()).device
  test_sample = {k: v.to(device) for k, v in test_sample.items()}
  with torch.no_grad():
    output = model(**test_sample)
  logits = output.logits.cpu()
  st.write("Logits:", logits.numpy())
  y_pred = int(torch.argmax(logits, dim=1).item())
  st.write("Prediction:", d[y_pred])
