import streamlit as st
from time import sleep
import transformers
import os
import transformers
import torch
from PyPDF2 import PdfReader
import LocalEmbedding
from langchain_core.vectorstores import VectorStoreRetriever
from transformers import AutoTokenizer, AutoModelForCausalLM



def handle_userinput(user_question, chatbot_response_slot):
    # TODO: Implement the chatbot response
    chatbot_response_slot.text("Chatbot: " + user_question)


def main():
    st.title("Chat Application")
    st.subheader("Welcome to the Chat Application")
    user_input = st.text_input("User Input", "")
    chatbot_response_slot = st.empty()  # Create an empty slot for chatbot response


    if user_input:
        handle_userinput(user_input, chatbot_response_slot)


    with st.sidebar:
        st.subheader("Upload PDF")
        pdf = st.file_uploader("Upload your PDF here")

        if pdf:
            with st.spinner("Processing"):
                pdf_reader = PdfReader(pdf)
                pdf_text = ""
                
                for page_num in (pdf_reader.pages):
                    pdf_text += page_num.extract_text()

                vector_store = LocalEmbedding.store_embeddings(LocalEmbedding.splitPDF(pdf_text))

                # TODO: Store the vector_store in a vector database

if __name__ == "__main__":
    main()