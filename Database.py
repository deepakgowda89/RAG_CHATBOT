from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

file_path=r"C:\Users\deepa\OneDrive\Documents\Desktop\RAG\RAG_PROJECTS\policy.txt"
loader=TextLoader(file_path,encoding="utf-8")
docs=loader.load()
splitter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=200
)
chunks=splitter.split_documents(docs)

embedding=HuggingFaceEmbeddings(

    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorestore=Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="rag_db"
)

# from langchain_community.vectorstores import Chroma
# from langchain.chat_models import init_chat_model
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader,TextLoader
# from langchain_huggingface import HuggingFaceEmbeddings

# file_path=r"C:\Users\deepa\OneDrive\Documents\Desktop\RAG\RAG_PROJECTS\policy.txt"
# loader=TextLoader(file_path,encoding="utf-8")
# docs=loader.load()

# embedding_model=HuggingFaceEmbeddings()

# splitter=RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=100
# )

# chunks=splitter.split_documents(docs)

# vectorestore=Chroma.from_documents(
#     documents=chunks,
#     embedding=embedding_model,
#     persist_directory='Chroma_db'
# )
# from langchain_core.prompts import ChatPromptTemplate

# vectorestore=Chroma(
#     persist_directory='Chroma_db',
#     embedding_function=embedding_model
# )

# prompt=ChatPromptTemplate([
#     ("system","""Act as a SImple chatbot analyse the given query from user and give normal answer without conflecting it""")
#     ,("human","""context:{context},
#     query:{query}""")
# ])


# retriever=vectorestore.as_retriever(
#     search_type='mmr',
#     k=2
# )

# query=input("You : ")


# docu=retriever.invoke(query)

# context="\n\n".join([documents for documents in docu])
# final_prompt=prompt.invoke(
#     context=context,
#     query=query
# )

# llm=init_chat_model(
#     model=''
# )

# result=llm.invoke(final_prompt)

# print(result)