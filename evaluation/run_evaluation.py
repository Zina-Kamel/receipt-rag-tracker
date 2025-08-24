import os
import json
import pandas as pd

from langsmith import Client
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.langchain_rag import get_rag_chain
from core.store import FaissStore
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "evaluation")))
from correctness_eval import correctness
from evaluation.groundedness_eval import groundedness
from evaluation.relevance_eval import relevance, retrieval_relevance
from dotenv import load_dotenv


load_dotenv()  
client = Client(api_key=os.environ["LANGSMITH_API_KEY"])

rag_chain = get_rag_chain("receipts_store")

dataset_name = "RAG Receipt Evaluation Data"  

def rag_bot(question: str) -> dict:
    """
    Input: question string
    Output: dictionary with keys:
        - answer (str)
        - documents (list of Document objects used by RAG)
    """
    result = rag_chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "documents": result.get("source_documents", [])
    }

def target(inputs: dict) -> dict:
    """
    Inputs: {'question': ...}
    Outputs: dict with 'answer' and 'documents'
    """
    return rag_bot(inputs["question"])

experiment_results = client.evaluate(
    target,
    data=dataset_name,
    evaluators=[correctness, groundedness, relevance, retrieval_relevance],
    experiment_prefix="rag-full-eval",
    metadata={"version": "LCEL context, gemini-1.5-flash-latest"}
)


try:
    df_results = experiment_results.to_pandas()
    df_results.to_csv("rag_evaluation_results.csv", index=False)
    print("Saved evaluation results to rag_evaluation_results.csv")
except Exception as e:
    print(f"Could not convert results to DataFrame: {e}")
