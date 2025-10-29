import os
from dotenv import load_dotenv
import streamlit as st
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.schema import HumanMessage,AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
load_dotenv()

os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_KEY")

st.title("RAG APPLICATION FOR Q & A")
st.session_state.message=st.chat_input("please ask your question")

def save_document_in_vectorstore():
    loader=PyPDFDirectoryLoader("resources\Attention.pdf")
    document=loader.load()
    text_spllit=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20)
    docs=text_spllit.split_documents(document)
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore=FAISS.from_documents(docs,embeddings)
    retriver=vectorstore.as_retriever(search_kwargs={"k": 3})
    st.session_state.retriver=retriver
    st.toast("Documents saved in vector")
    return retriver

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

Question: {input}

Answer:
""")

llm=ChatGroq(model="llama-3.1-8b-instant")
def getLLMOutput(documents):
    pass

if "retriver" in st.session_state:
    document_chain=create_stuff_documents_chain(llm=llm,prompt=prompt)
    retrieval_chain = create_retrieval_chain(st.session_state.retriver, document_chain)
    result=retrieval_chain.invoke({"input": st.session_state.message})
    st.chat_message("user").write(st.session_state.message)
    st.chat_message("assistant").write(result["answer"])
else:
    save_document_in_vectorstore()
    document_chain=create_stuff_documents_chain(llm=llm,prompt=prompt)
    retrieval_chain = create_retrieval_chain(st.session_state.retriver, document_chain)
    result=retrieval_chain.invoke({"input": st.session_state.message})
    st.chat_message("user").write(st.session_state.message)
    st.chat_message("assistant").write(result["answer"])