import os
import warnings
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain

# Ignore unnecessary warnings
warnings.filterwarnings("ignore")

# Streamlit app title
st.title("🌐 Wikipedia Chatbot using LangChain + Groq")

# --- Secure API key management ---
# Do NOT paste your key directly in the code.
# Instead, store it in Streamlit secrets.
# You can set it later in .streamlit/secrets.toml (for local or cloud)

groq_api_key = st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.warning("⚠️ Please set your GROQ_API_KEY in Streamlit secrets.")
    st.stop()

# --- URL Input ---
url = st.text_input("🌍 Enter a Wikipedia URL:")

if url:
    st.info(f"🔍 Loading data from: {url}")

    # Step 1: Load and split content
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
    except Exception as e:
        st.error(f"❌ Error loading webpage: {e}")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Step 2: Create vector store
    try:
        st.write("⚙️ Creating embeddings and FAISS index...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        st.error(f"❌ Error creating vector store: {e}")
        st.stop()

    # Step 3: Initialize Groq LLM
    try:
        llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.1-8b-instant")
        st.success("✅ Groq model initialized successfully!")
    except Exception as e:
        st.error(f"❌ Error initializing Groq LLM: {e}")
        st.stop()

    # Step 4: Create conversational chain
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False
    )

    # Step 5: Chat interface
    st.success("💬 Chatbot is ready! Ask a question about this Wikipedia page below.")
    query = st.text_input("Ask something about this page:")

    if query:
        try:
            response = chain({"question": query, "chat_history": []})
            st.markdown(f"**🤖 Answer:** {response['answer']}")
        except Exception as e:
            st.error(f"❌ Error generating response: {e}")
