A Streamlit-based AI chatbot that enables users to search, query, and retrieve information from news articles using modern retrieval-augmented generation (RAG) techniques. The tool uses embeddings, vector stores, and LLMs to provide concise answers with cited sources.

Features

News Article Processing:
Input multiple news article URLs and process them automatically.

Text Splitting & Chunking:
Articles are split into manageable chunks for better embedding and retrieval.

Embeddings & Vector Store:
Converts article text into embeddings using HuggingFaceEmbeddings and stores them in a FAISS vector index for efficient semantic search.

Retrieval-Augmented QA:
Uses RetrievalQAWithSourcesChain to answer queries by retrieving relevant chunks from the processed articles.

Sources Tracking:
Provides the source URLs or text chunks used to generate the answer.

Interactive Streamlit Interface:
Easy-to-use sidebar for URL input and main interface for asking questions and viewing results.

Technologies Used

Streamlit: For building the interactive web interface.

LangChain: For handling document retrieval and integrating LLMs.

FAISS: Facebook AI Similarity Search for fast vector-based retrieval.

HuggingFace Embeddings: For converting text into vector representations.

ChatGroq (LLM): Advanced language model for generating answers.

UnstructuredURLLoader & RecursiveCharacterTextSplitter: For loading and preprocessing article text.

Python dotenv: To securely manage API keys (GROQ_API_KEY).
