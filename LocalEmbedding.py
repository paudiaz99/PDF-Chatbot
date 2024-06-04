from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings

def store_embeddings(chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_store

def splitPDF(text):
    # Split the text into sentences
    text_split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)
    chunks = text_split.split_text(text=text)
    return chunks

def get_embeddings(text):
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    return embeddings.embed_query(text)