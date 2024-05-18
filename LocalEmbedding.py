from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings



def store_embeddings(chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    vector_store = FAISS.from_texts(texts = chunks, embedding = embeddings)
    return vector_store


def splitPDF(text):
    # Split the text into sentences
    text_split = RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap= 100, length_function = len)
    chunks = text_split.split_text(text=text)
    return chunks