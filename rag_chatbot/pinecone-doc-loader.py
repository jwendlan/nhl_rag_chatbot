import os
import sys

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceEmbeddings

os.environ['OPENAI_API_KEY'] = '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'
os.environ['PINECONE_API_KEY'] = 'pcsk_2oxcYN_NPhzizrpYnXdAbvyAt38mF8iGZBNVLXRKqXhKjn94qWRUqEUAt3s8ZyetjnQTCr'

# index_name = "llama-text-embed-v2-index"
index_name = "huggingface"
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
        
        # Load, split, and add to vector store
        loader = TextLoader(file_path)
        documents = loader.load()
        
        text_splitter = CharacterTextSplitter(chunk_size=7000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        # for doc in docs:
        #     print(doc)
        # texts = [doc.page_content for doc in docs]
        # metadatas = [{"source": filename} for _ in docs]

        # metadatas = [prune_metadata(meta) for meta in metadatas]
        
        # vectorstore.add_texts(texts=texts, metadatas=metadatas)

        # if check_metadata_size(metadatas):
        #     vectorstore.add_texts(texts=texts, metadatas=metadatas)
        #     print(f"Added documents from {filename} to vector store.")
        # else:
        #     print(f"Skipping {filename} due to large metadata size.")
        # break
        vectorstore.add_documents(docs)
        print(f"Added documents from {filename} to vector store.")
        # break
        # break

# loader = TextLoader("../data/nhl_stats_full.txt")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=3100, chunk_overlap=30)
# docs = text_splitter.split_documents(documents)
# print(len(docs))

# vectorstore_from_docs = PineconeVectorStore.from_documents(
#     docs,
#     index_name=index_name,
#     embedding=embeddings
# )

# vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# vectorstore.add_documents(docs)

# import json

# # Path to your .txt file
# file_path = '../data/nhl_stats_full.txt'

# # Read and parse the JSON
# with open(file_path, 'r') as file:
#     data = json.load(file)

# print(type(data))
# print(json.dumps(data, indent=2)[:1000])

# json_string = json.dumps(data)
# # splitter = RecursiveJsonSplitter(max_chunk_size = 3000)
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=3000,
#     chunk_overlap=100
# )
# # json_chunks = splitter.split_json(json_data=json_string)
# json_string = json.dumps(data)
# docs = splitter.split_text(json_string)
# # docs = splitter.split_text(json_data=json_string)

# print(len(docs))
# print(docs[0])

# texts = ["Tonight, I call on the Senate to: Pass the Freedom to Vote Act.", "ne of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court.", "One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence."]

# vectorstore_from_texts = PineconeVectorStore.from_texts(
#     texts,
#     index_name=index_name,
#     embedding=embeddings
# )