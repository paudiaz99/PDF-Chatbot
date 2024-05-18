import streamlit as st
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import LocalEmbedding
from langchain_core.vectorstores import VectorStoreRetriever
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import io

def handle_userinput(user_question, chatbot_response_slot):
    # TODO: Implement the chatbot response
    chatbot_response_slot.text("Chatbot: " + user_question)

def pdf_preview(file_buffer):
    pdf_bytes = io.BytesIO(file_buffer.read())
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    return pdf_document

def main():
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False 

    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0

    if 'processing' not in st.session_state:
        st.session_state.processing = False

    st.title("PDF Chatbot")
    st.subheader("Add a PDF file to start the conversation")

    if st.session_state.pdf_processed:
        user_input = st.text_input("User Input", "")
        chatbot_response_slot = st.empty() 

        if user_input:
            handle_userinput(user_input, chatbot_response_slot)

    with st.sidebar:
        st.subheader("Upload PDF")
        pdf = st.file_uploader("Upload your PDF here")

        if pdf:
            if st.button("Process PDF"):
                st.session_state.processing = True
                with st.spinner("Processing"):
                    pdf_reader = PdfReader(pdf)
                    pdf_text = ""
                    
                    for page in pdf_reader.pages:
                        pdf_text += page.extract_text()

                    vector_store = LocalEmbedding.store_embeddings(LocalEmbedding.splitPDF(pdf_text))

                    # TODO: Store the vector_store in a vector database

                    st.success("PDF uploaded successfully")
                    st.session_state.pdf_processed = True 
                    st.session_state.processing = False

            pdf_document = pdf_preview(pdf)

            st.subheader("PDF Preview")

            if 0 <= st.session_state.current_page < len(pdf_document):
                page = pdf_document.load_page(st.session_state.current_page)
                image_bytes = page.get_pixmap().tobytes()
                st.image(image_bytes, use_column_width=True)
            else:
                st.write("Page not found")
            
            prev_col, next_col = st.columns(2)
            if prev_col.button("Previous Page") and st.session_state.processing == False:
                st.session_state.current_page = max(0, st.session_state.current_page - 1)

            if next_col.button("Next Page") and st.session_state.processing == False:
                st.session_state.current_page = min(len(pdf_document) - 1, st.session_state.current_page + 1)

if __name__ == "__main__":
    main()