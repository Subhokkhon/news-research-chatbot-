import os
import streamlit as st
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load API key
load_dotenv()  # Ensure GROQ_API_KEY is set in your .env

st.title("ChatBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Inputs
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url = st.sidebar.button("Process URLs")
faiss_path = "faiss_store_groq"

# Use a supported Groq model instead of deprecated one
llm = ChatGroq(
    model="llama-3.3-70b-versatile",      # Updated model
    temperature=0.9,
    max_tokens=500
)

main = st.empty()

if process_url:
    loader = UnstructuredURLLoader(urls=[u for u in urls if u])
    main.text("Loading data... ⏳")
    docs = loader.load()

    main.text("Splitting text... ⏳")
    splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(docs)

    main.text("Generating embeddings... ⏳")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    main.text("Saving FAISS index... ⏳")
    vectorstore.save_local(faiss_path)
    main.success("FAISS index saved successfully!")

query = main.text_input("Enter your question:")
if query:
    if os.path.exists(faiss_path):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(
            faiss_path, embeddings, allow_dangerous_deserialization=True
        )

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        st.info("Querying... ⏳")
        response = chain({"question": query}, return_only_outputs=True)
        sources = response.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for src in sources.split("\n"):
                st.write(src)
    else:
        st.error("FAISS store not found. Please process URLs first.")
