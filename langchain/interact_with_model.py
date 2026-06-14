import requests
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

model = init_chat_model(
    model = "gpt-4.1-mini",
    temperature = 0.1
)

# response = model.invoke("Hi, can you describe the rules of gather step in basketball?")

# print(response.content)

# context
conversation = [
    SystemMessage("You are a helpful assistant for questions regarding general knowledge"),
    HumanMessage("What is basketball"),
    AIMessage("Basketball is a sport"),
    HumanMessage("When was it created?")
]

response2 = model.invoke(conversation)
print(response2.content)