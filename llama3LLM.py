import requests
import re

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
headers = {"Authorization": "Bearer hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def get_answer_from_llama3(question, context):
    output = query({
	"inputs": """You are an AI assistant designed to answer questions based on the content of a provided PDF document.
    The text from the PDF is extracted and provided below (delimited by '[]'). 
    Your task is to use the information from this text to accurately and comprehensively respond to any queries related to the content. You are also able to sumarize the text if needed.
    Make sure to base your answers strictly on the information available in the text. If no clear answer can be inferred, you should indicate that in the response section instead of providing incorrect information.
    IMPORTANT: Response pattern is as follows: r'Response:\s+---(.+?)---' where the response is within '---' delimiters.
    After response, no other information should be provided.
    Question is surrounded by '///' delimiters.
    Response is surrounded by '---' delimiters.

    Extracted PDF Text:

    ["""+context+"""]
	
	Question: ///"""+question+"""///
    Response: """,
      
    })

    print(output)
    # Extract the generated_text from the data
    generated_text = output[0]["generated_text"]

    # Use regular expressions to find the response
    response_match = re.search(r'Response:\s+---(.+?)---', generated_text, re.DOTALL)
    if response_match:
        response = response_match.group(1).strip()
        print(response)
        return response
    else:
        print("Response not found.")
        return "LLM could not generate a response. Please make sure your question is clear and relevant to the provided context."
    

def get_pdf_summary_from_llama3(question, context):
    output = query({
	"inputs": """You are an AI assistant designed to answer questions based on the content of a provided PDF document.
    The text from the PDF is extracted and provided below (delimited by '[]'). 
    Your task is to use the information from this text to accurately and comprehensively respond to any queries related to the content. You are also able to sumarize the text if needed.
    Make sure to base your answers strictly on the information available in the text. If no clear answer can be inferred, you should indicate that in the response section instead of providing incorrect information.
    IMPORTANT: Response pattern is as follows: r'Response:\s+---(.+?)---' where the response is within '---' delimiters.
    After response, no other information should be provided.
    Question is surrounded by '///' delimiters.
    Response is surrounded by '---' delimiters.

    Extracted PDF Text:

    ["""+context+"""]
	
	Question: ///"""+question+"""///
    Response: """,
      
    })

    print(output)
    generated_text = output[0]["generated_text"]

    response_match = re.search(r'Response:\s+---(.+?)---', generated_text, re.DOTALL)
    if response_match:
        response = response_match.group(1).strip()
        print(response)
        return response
    else:
        print("Response not found.")
        return "LLM could not generate a summary. Please make sure your file is in correct format."