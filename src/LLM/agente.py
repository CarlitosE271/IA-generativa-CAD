"""Modelo utilizado como LLM"""
from langchain_ollama import OllamaLLM, OllamaEmbeddings 

model = OllamaLLM(
    model="llama3.1",
    temperature=0.7,
    num_predict=256,
)

input_text = "The meaning of life is "
response = model.invoke(input_text)
print(response)