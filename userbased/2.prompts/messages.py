from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()
# model = ChatOpenAI()
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)
model = ChatHuggingFace(llm=llm)


messages=[
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about LangChain')
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)
