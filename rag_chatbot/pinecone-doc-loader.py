import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

os.environ['OPENAI_API_KEY'] = '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'
os.environ['PINECONE_API_KEY'] = 'pcsk_2oxcYN_NPhzizrpYnXdAbvyAt38mF8iGZBNVLXRKqXhKjn94qWRUqEUAt3s8ZyetjnQTCr'

index_name = "llama-text-embed-v2-index"
embeddings = OpenAIEmbeddings()

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

loader = TextLoader("../data/nhl_stats_full.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=3100, chunk_overlap=30)
docs = text_splitter.split_documents(documents)

# vectorstore_from_docs = PineconeVectorStore.from_documents(
#     docs,
#     index_name=index_name,
#     embedding=embeddings
# )

# vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

vectorstore.add_documents(docs)

# texts = ["Tonight, I call on the Senate to: Pass the Freedom to Vote Act.", "ne of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court.", "One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence."]

# vectorstore_from_texts = PineconeVectorStore.from_texts(
#     texts,
#     index_name=index_name,
#     embedding=embeddings
# )