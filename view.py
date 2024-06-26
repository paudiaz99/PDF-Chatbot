import streamlit as st
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import LocalEmbedding
import io
from vector_database import VectorDatabase
from llama3LLM import get_answer_from_llama3, get_pdf_summary_from_llama3

if 'vector_db' not in st.session_state:
    dimension = 768
    st.session_state.vector_db = VectorDatabase(dimension)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def reset_chat_state():
    st.session_state.chat_history = []
    st.session_state.pdf_processed = False
    st.session_state.current_page = 0
    st.session_state.processing = False

def handle_userinput(user_question, chat_box):
    try:
        question_embedding = LocalEmbedding.get_embeddings(user_question)
        
        # Buscar los textos relevantes en la base de datos vectorial
        relevant_texts = st.session_state.vector_db.search(question_embedding, top_k=3)
        
        if relevant_texts:
            context = " ".join([text[1] for text in relevant_texts])
            
            # Obtener la respuesta del LLM usando el contexto
            try:
                answer = get_answer_from_llama3(user_question, context)
                st.session_state.chat_history.append({"user": "User", "text": user_question})
                st.session_state.chat_history.append({"user": "Chatbot", "text": answer})
                update_chat_box(chat_box)
            except Exception as e:
                st.session_state.chat_history.append({"user": "Chatbot", "text": f"An error occurred while fetching the answer: {e}"})
                update_chat_box(chat_box)
        else:
            st.session_state.chat_history.append({"user": "Chatbot", "text": "No relevant information found in the PDF."})
            update_chat_box(chat_box)
    except Exception as e:
        st.session_state.chat_history.append({"user": "Chatbot", "text": f"Error: {e}"})
        update_chat_box(chat_box)
        
def pdf_summary():
    summary_question = "What is the document about?"
    try:
        question_embedding = LocalEmbedding.get_embeddings(summary_question)
        
        relevant_texts = st.session_state.vector_db.search(question_embedding, top_k=3)
        
        if relevant_texts:
            context = " ".join([text[1] for text in relevant_texts])
            
            try:
                summary = get_pdf_summary_from_llama3(summary_question, context)
                summary_text = "Welcome to LSDatasheet! Here's a summary of the PDF: " + summary + " I will be happy to answer any questions related to this document."
                return summary_text
            except Exception as e:
                return f"An error occurred while fetching the answer: {e}"
        else:
            return "No relevant information found in the PDF."
    except Exception as e:
        return f"Error: {e}"
            
def pdf_preview(file_buffer):
    try:
        pdf_bytes = io.BytesIO(file_buffer.read())
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        return pdf_document
    except Exception as e:
        st.error(f"Error opening PDF: {e}")
        return None

def update_chat_box(chat_box):
    chat_history = ""
    for msg in st.session_state.chat_history:
        if isinstance(msg, dict) and "user" in msg and "text" in msg:
            if msg["user"] == "User":
                chat_history += f"<div style='text-align: right;'><span style='background-color: blue; padding: 5px 10px; border-radius: 10px; display: inline-block; color: white;'>{msg['text']}</span></div><br>"
            else:
                chat_history += f"<div style='text-align: left;'><span style='background-color: black; padding: 5px 10px; border-radius: 10px; display: inline-block; color: white;'>{msg['text']}</span></div><br>"
    chat_box.markdown(f"<div style='height: 400px; overflow-y: auto;'>{chat_history}</div>", unsafe_allow_html=True)

def main():
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False 

    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0

    if 'processing' not in st.session_state:
        st.session_state.processing = False

    st.title("PDF Chatbot")
    st.subheader("Add a PDF file to start the conversation")

    with st.sidebar:
        st.subheader("Upload PDF")
        pdf = st.file_uploader("Upload your PDF here")

        if pdf:
            if st.button("Process PDF"):
                reset_chat_state()
                st.session_state.processing = True
                with st.spinner("Processing"):
                    try:
                        pdf_reader = PdfReader(pdf)
                        pdf_text = "".join([page.extract_text() for page in pdf_reader.pages])
                        
                        chunks = LocalEmbedding.splitPDF(pdf_text)
                        for chunk in chunks:
                            embedding = LocalEmbedding.get_embeddings(chunk)
                            st.session_state.vector_db.add_document(embedding, chunk)

                        st.success("PDF uploaded successfully")
                        st.session_state.pdf_processed = True 
                        st.session_state.processing = False

                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")
                        st.session_state.processing = False

            if st.session_state.pdf_processed:
                pdf_document = pdf_preview(pdf)

                st.subheader("PDF Preview")

                if pdf_document and 0 <= st.session_state.current_page < len(pdf_document):
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

    if st.session_state.pdf_processed:
        chat_box = st.empty()
        update_chat_box(chat_box)
        if not st.session_state.chat_history:
            summary_text = pdf_summary()
            st.session_state.chat_history.append({"user": "Chatbot", "text": summary_text})
            update_chat_box(chat_box)

        user_input = st.text_input("Your question")
        if st.button("Send"):
            if user_input:
                handle_userinput(user_input, chat_box)
            else:
                st.write("Please enter your question.")

if __name__ == "__main__":
    main()
