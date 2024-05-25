import requests

API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
API_TOKEN = ""
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")
    return response.json()

def get_answer_from_llm(question, context):
    max_context_length = 750 

    if len(context) > max_context_length:
        context = context[:max_context_length]

    payload = {
        "inputs": {
            "question": question,
            "context": context
        }
    }
    response = query(payload)
    return response['answer']