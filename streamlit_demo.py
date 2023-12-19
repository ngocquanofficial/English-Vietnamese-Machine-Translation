import streamlit as st
import requests
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
checkpoint = "ngocquanofficial/machine_translation_VinAI"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Set the title of your Streamlit app
st.title("ENGLISH-VIETNAMESE MACHINE TRANSLATION")
text = st.text_input("Enter text for Vietnamese Translation: ")
if text == "" :
    output = "Hãy nhập văn bản Tiếng Anh"
else :
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Generate outputs on the same device as the input
    outputs = model.generate(input_ids, max_length=128, do_sample=True, top_k=15, top_p=0.95)

    # Decode the generated output
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # output = output[0]["generated_text"]


st.write(output)
