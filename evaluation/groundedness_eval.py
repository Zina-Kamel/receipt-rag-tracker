from typing_extensions import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document, SystemMessage, HumanMessage
from langchain_core.output_parsers.json import JsonOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[bool, ..., "Provide the score on if the answer hallucinates from the documents"]

grounded_instructions = """You are a teacher grading a quiz. 

You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

parser = JsonOutputParser(schema=GroundedGrade)
grounded_llm = ChatGoogleGenerativeAI(
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


def groundedness(inputs: dict, outputs: dict) -> bool:
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    prompt = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"

    raw_output = grounded_llm.invoke([
        SystemMessage(content=grounded_instructions),
        HumanMessage(content=prompt)
    ])

    parsed = parse_llm_json(parser, raw_output)
    return parsed["grounded"]
