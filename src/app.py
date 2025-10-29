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
user_input=st.chat_input("please ask your question")

def save_document_in_vectorstore():
    # ✅ Load PDFs from a folder (not a single file)
    loader = PyPDFDirectoryLoader("resources")
    documents = loader.load()

    if not documents:
        st.error("❌ No PDF files found in the 'resources' directory.")
        return None

    # ✅ Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)

    if not docs:
        st.error("❌ No document chunks generated. Check PDF content.")
        return None

    # ✅ Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # ✅ Create FAISS vectorstore safely
    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        st.session_state.retriver = retriever
        st.toast("✅ Documents saved in vector store successfully!")
        return retriever
    except Exception as e:
        st.error(f"❌ Error creating FAISS vectorstore: {e}")
        return None

if user_input:
    st.session_state.message=user_input
    save_document_in_vectorstore()

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

Question: {input}

Answer:
""")

llm=ChatGroq(model="llama-3.1-8b-instant")
def getLLMOutput(documents):
    pass

if "retriver" in st.session_state and st.session_state.retriver is not None:
    document_chain=create_stuff_documents_chain(llm=llm,prompt=prompt)
    retrieval_chain = create_retrieval_chain(st.session_state.retriver, document_chain)
    result=retrieval_chain.invoke({"input": st.session_state.message})
    st.chat_message("user").write(st.session_state.message)
    st.chat_message("assistant").write(result["answer"])