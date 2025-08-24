from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.docstore.in_memory import InMemoryDocstore
from core.store import FaissStore
from config import settings
import logging

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = settings.GEMINI_API_KEY
EMBEDDING_MODEL_NAME = settings.EMBEDDING_MODEL_NAME
GEMINI_MODEL = settings.GEMINI_MODEL
FAISS_RECEIPTS_STORE_NAME = settings.FAISS_RECEIPTS_STORE_NAME

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY
)

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"}
)

def load_langchain_vectorstore(store_name=FAISS_RECEIPTS_STORE_NAME):
    """
    Load a FAISS vectorstore and wrap it into a LangChain-compatible object.

    Args:
        store_name (str): The name of the FAISS store to load.
                         Defaults to the configured receipts vectorstore.

    Returns:
        FAISS: A LangChain FAISS vectorstore instance that supports similarity search.
    """
    fs = FaissStore(store_name=store_name)
    fs.load() 

    documents = [Document(page_content=json_str, metadata=meta)
                 for json_str, meta in zip(fs.texts, fs.metadata)]

    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}

    vectorstore = FAISS(
        embedding_model,
        fs.index,
        docstore,
        index_to_docstore_id
    )
    logger.info("FAISS index successfully loaded")
    logger.debug(f"Index size: {fs.index.ntotal}, Documents count: {len(fs.texts)}")
    return vectorstore

def get_rag_chain(store_name):
    """
    Create a RAG QA chain using LangChain.

    Args:
        store_name (str): The name of the FAISS vectorstore to query against.

    Returns:
        RetrievalQA: A LangChain RetrievalQA chain configured with Gemini LLM
                     and a retriever over the provided FAISS store.
    """
    vectorstore = load_langchain_vectorstore(store_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",             
        return_source_documents=True
    )
    
    logger.info("RAG chain created successfully")
    
    return chain
