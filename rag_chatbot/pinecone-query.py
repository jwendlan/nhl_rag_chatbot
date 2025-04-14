use_serverless = True

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec, PodSpec  
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import os
import time  
# configure client
OPENAI_KEY = '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'
PINECONE_KEY = 'pcsk_2oxcYN_NPhzizrpYnXdAbvyAt38mF8iGZBNVLXRKqXhKjn94qWRUqEUAt3s8ZyetjnQTCr'
os.environ['OPENAI_API_KEY'] = OPENAI_KEY
os.environ['PINECONE_API_KEY'] = PINECONE_KEY
pc = Pinecone(api_key='pcsk_2oxcYN_NPhzizrpYnXdAbvyAt38mF8iGZBNVLXRKqXhKjn94qWRUqEUAt3s8ZyetjnQTCr')  
# if use_serverless:  
#     spec = ServerlessSpec(cloud='aws', region='us-east-1')  
# else:  
#     # if not using a starter index, you should specify a pod_type too  
#     spec = PodSpec()  
# check for and delete index if already exists  
# index_name = 'langchain-retrieval-augmentation-fast'  
# if pc.has_index(index_name):  
#     pc.delete_index(name=index_name)  
# # create a new index  
# pc.create_index(  
#     index_name,  
#     dimension=1536,  # dimensionality of text-embedding-ada-002  
#     metric='dotproduct',  
#     spec=spec  
# )  

index_name = "llama-text-embed-v2-index"
embeddings = OpenAIEmbeddings()

index = pc.Index(index_name)  
print(index.describe_index_stats())

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

query = "How many attempted shots did Boston block in the 2008 season?"  
res = vectorstore.similarity_search(  
    query,  # our search query  
    k=3  # return 3 most relevant docs  
)
for entry in res:
    print(entry)
    print('\n\n')

# from langchain_openai import ChatOpenAI  
# from langchain.chains import RetrievalQA  
# # completion llm  
# llm = ChatOpenAI(  
#     openai_api_key=OPENAI_KEY, 
#     model_name='gpt-4o-mini',  
#     temperature=0.0  
# )  
# qa = RetrievalQA.from_chain_type(  
#     llm=llm,  
#     chain_type="stuff",  
#     retriever=vectorstore.as_retriever()  
# )  
# print(qa.invoke(query))