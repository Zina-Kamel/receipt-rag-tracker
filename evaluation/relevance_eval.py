from typing_extensions import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers.json import JsonOutputParser
from langchain.schema import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "Provide the score on whether the answer addresses the question"]

class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "True if the retrieved documents are relevant to the question, False otherwise"]

relevance_instructions = """You are a teacher grading a quiz. 
You will be given a QUESTION and a STUDENT ANSWER. 
Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
Avoid simply stating the correct answer at the outset."""

retrieval_relevance_instructions = """You are a teacher grading a quiz. 
You will be given a QUESTION and a set of FACTS provided by the student. 
Here is the grade criteria to follow:
(1) Your goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
Avoid simply stating the correct answer at the outset."""

relevance_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)
retrieval_relevance_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

relevance_parser = JsonOutputParser(schema=RelevanceGrade)
retrieval_parser = JsonOutputParser(schema=RetrievalRelevanceGrade)

def parse_llm_json(parser, raw_output):
    content = raw_output if isinstance(raw_output, str) else getattr(raw_output, "content", str(raw_output))
    
    import re, json
    match = re.search(r"```json\s*(\{.*\})\s*```", content, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = content.strip()
    
    try:
        return parser.parse(json_str)
    except Exception as e:
        print("Failed to parse JSON from LLM output:")
        print(content)
        raise e


def relevance(inputs: dict, outputs: dict) -> bool:
    answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
    raw_output = relevance_llm.invoke([
        SystemMessage(content=relevance_instructions),
        HumanMessage(content=answer)
    ])
    content = raw_output if isinstance(raw_output, str) else getattr(raw_output, "content", str(raw_output))
    parsed = parse_llm_json(relevance_parser, raw_output)

    return parsed["relevant"]

def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    prompt = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"
    raw_output = retrieval_relevance_llm.invoke([
        SystemMessage(content=retrieval_relevance_instructions),
        HumanMessage(content=prompt)
    ])
    content = raw_output if isinstance(raw_output, str) else getattr(raw_output, "content", str(raw_output))
    parsed = parse_llm_json(retrieval_parser, raw_output)
    return parsed["relevant"]
