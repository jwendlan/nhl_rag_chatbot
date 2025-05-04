import os
import sys

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceEmbeddings

os.environ['PINECONE_API_KEY'] = "" #needs to be set Pinecone API Key

index_name = "" #needs to be set to the Pinecone Index Name
# embeddings = OpenAIEmbeddings()
embeddings = HuggingFaceEmbeddings() #can be changed to OpenAI Embeddings instead

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

txt_dir = "../data/texts_carl_limit500000" #input directory containing text files

for filename in os.listdir(txt_dir):
    if filename.endswith('.txt'):
        file_path = os.path.join(txt_dir, filename)
        print(f"Processing: {filename}")
        
        loader = TextLoader(file_path)
        documents = loader.load()
        
        #chunking method, params can be modified for different size chunks
        text_splitter = CharacterTextSplitter(chunk_size=7000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        
        vectorstore.add_documents(docs)
        print(f"Added documents from {filename} to vector store.")