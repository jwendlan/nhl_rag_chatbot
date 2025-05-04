import os
import sys

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceEmbeddings

os.environ['PINECONE_API_KEY'] = ""

index_name = ""
# embeddings = OpenAIEmbeddings()
embeddings = HuggingFaceEmbeddings()

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

txt_dir = "../data/texts_carl_limit500000"

def check_metadata_size(metadatas):
    for meta in metadatas:
        meta_size = sys.getsizeof(str(meta))  # Get the size of metadata
        if meta_size > 40960:
            print(f"Metadata exceeds size limit: {meta_size} bytes")
            return False  # Metadata size exceeds limit
    return True

def prune_metadata(metadata):
    return {"source": metadata["source"]}  # Keep only the essential metadata

for filename in os.listdir(txt_dir):
    if filename.endswith('.txt'):
        file_path = os.path.join(txt_dir, filename)
        print(f"Processing: {filename}")
        
        loader = TextLoader(file_path)
        documents = loader.load()
        
        text_splitter = CharacterTextSplitter(chunk_size=7000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        
        vectorstore.add_documents(docs)
        print(f"Added documents from {filename} to vector store.")