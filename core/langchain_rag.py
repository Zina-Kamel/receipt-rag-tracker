import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from core.store import FaissStore
from langchain.docstore.in_memory import InMemoryDocstore

# Initialize LLM (Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=os.environ["GEMINI_API_KEY"])

# Embedding model must match the one you use when creating embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_langchain_vectorstore():
    fs = FaissStore()
    fs.load()

    documents = [Document(page_content=text, metadata=meta) for text, meta in zip(fs.texts, fs.metadata)]

    # Use InMemoryDocstore, not a dict
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})

    index_to_docstore_id = {i: str(i) for i in range(len(documents))}

    vectorstore = FAISS(
        embedding_model.embed_query,  # embedding function
        fs.index,                     # loaded FAISS index
        docstore,                    # InMemoryDocstore instance
        index_to_docstore_id         # mapping index to docstore id
    )
    
    print(f"Loaded FAISS index size: {fs.index.ntotal}")
    print(f"Loaded documents count: {len(fs.texts)}")

    return vectorstore


def get_rag_chain():
    retriever = load_langchain_vectorstore().as_retriever(search_kwargs={"k": 3})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    return chain
