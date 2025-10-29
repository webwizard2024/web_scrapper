import os
import warnings
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain

warnings.filterwarnings("ignore")

# Set up environment
groq_api = os.getenv("GROQ_API_KEY")

st.title("🌐 Wikipedia Chatbot using LangChain + Groq")

url = st.text_input("Enter a Wikipedia URL:")

if url:
    st.info(f"Loading data from: {url}")

    # Step 1: Load and split
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Step 2: Create vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Step 3: Create chatbot
    llm = ChatGroq(api_key=groq_api, model_name="llama-3.1-8b-instant")
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False
    )

    # Step 4: Chat interface
    st.success("✅ Chatbot is ready! Ask questions about the page below.")
    query = st.text_input("Ask a question about this page:")

    if query:
        response = chain({"question": query, "chat_history": []})
        st.write("🤖", response["answer"])
