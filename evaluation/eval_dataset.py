from langsmith import Client
from dotenv import load_dotenv

load_dotenv()  

client = Client()

examples = [
    {
        "inputs": {"question": "What items did I buy at Walmart in October? "},
        "outputs": {"answer": "BANANAS, FRAP, OT 200Z TUM, M ATHLETICS, DEXAS15X20, GV OATMEAL"},
    },
    {
        "inputs": {"question": "What did I buy from walmart?"},
        "outputs": {"answer": "BANANAS, FRAP, OT 200Z TUM, M ATHLETICS, DEXAS15X20, GV OATMEAL, water"},
    },
    {
        "inputs": {"question": "Show items from my Walmart receipt on October 18th."},
        "outputs": {"answer": "OT 200Z TUM, M ATHLETICS, DEXAS15X20, GV OATMEAL"},
    }
]

dataset_name = "RAG Receipt Evaluation Data"
dataset = client.create_dataset(dataset_name=dataset_name)
client.create_examples(
    dataset_id=dataset.id,
    examples=examples
)