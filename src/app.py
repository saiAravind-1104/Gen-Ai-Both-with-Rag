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

st.title("BASIC RAG APPLICATION FOR Q & A")
st.session_state.message=st.chat_input("please ask your question")

def save_document_in_vectorstore():
    loader=PyPDFDirectoryLoader("D:\Git Hub\Day1\Gen-Ai-Both-with-Rag\Documents")
    document=loader.load()
    text_spllit=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20)
    docs=text_spllit.split_documents(document)
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore=FAISS.from_documents(docs,embeddings)
    st.session_state.vector=vectorstore
    st.chat_message("Message saved in vector store")

if "vector" in st.session_state:
    output=st.session_state.vector.similarity_search(query=st.session_state.message,k=1)
    st.chat_message("assistant").write(output[0].page_content)
else:
    save_document_in_vectorstore()
    output=st.session_state.vector.similarity_search(query=st.session_state.message,k=1)
    st.chat_message("assistant").write(output[0].page_content)