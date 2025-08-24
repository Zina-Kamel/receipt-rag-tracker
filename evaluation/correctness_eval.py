from typing_extensions import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers.json import JsonOutputParser
from dotenv import load_dotenv
from langchain.schema import SystemMessage, HumanMessage
import os

load_dotenv()

class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

correctness_instructions = """You are a teacher grading a quiz. 
You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. 
Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. 
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the ground truth answer.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.
Avoid simply stating the correct answer at the outset."""

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

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


grader_parser = JsonOutputParser(schema=CorrectnessGrade)

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """An evaluator for RAG answer accuracy using Gemini"""
    answers = f"""QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {outputs['answer']}"""

    raw_output = gemini_llm.invoke([
        SystemMessage(content=correctness_instructions),
        HumanMessage(content=answers)
    ])

    parsed = parse_llm_json(grader_parser, raw_output)

    return parsed["correct"]
