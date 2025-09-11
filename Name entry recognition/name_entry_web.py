import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

with open('token2idx.pkl', 'rb') as f:
    token2idx = pickle.load(f)

with open('tag2idx.pkl', 'rb') as f:
    tag2idx = pickle.load(f)

with open('maxlen.pkl', 'rb') as f:
    maxlen = pickle.load(f)

idx2tag = {v: k for k, v in tag2idx.items()}

model = load_model('ner_model.h5')

st.title("Named Entity Recognition Demo")
st.write("Enter text to analyze for named entities:")

text = st.text_input("Input Text:", "John Smith works at Google in California")

if text:
    tokens = text.split()
    token_indices = [token2idx.get(token.lower(), len(token2idx)) for token in tokens]
    
    padded_tokens = pad_sequences([token_indices], maxlen=maxlen, 
                                padding='post', value=len(token2idx))
    
    predictions = model.predict(padded_tokens)
    predicted_tags = np.argmax(predictions, axis=-1)[0]
    
    st.write("### Results:")
    result_html = "<div style='line-height: 2.0;'>"
    for token, tag_idx in zip(tokens, predicted_tags[:len(tokens)]):
        tag = idx2tag.get(tag_idx, 'O')
        if tag != 'O':
            result_html += f"<mark style='background-color: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;'>{token}<span style='font-size: 0.8em; font-weight: bold; margin-left: 0.5rem'>{tag}</span></mark> "
        else:
            result_html += f"{token} "
    result_html += "</div>"
    
    st.markdown(result_html, unsafe_allow_html=True)
    
    with st.expander("Show raw tags"):
        for token, tag_idx in zip(tokens, predicted_tags[:len(tokens)]):
            st.write(f"{token}: {idx2tag.get(tag_idx, 'O')}")