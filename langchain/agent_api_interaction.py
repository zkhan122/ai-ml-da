import requests
from dotenv import load_dotenv

from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat: # model will answer in this format
    summary: str
    temperature_c: float
    temperature_f: float
    humidity: float

@tool("get_weather", description="This agent will get the weather for the given city", return_direct=False)
def get_weather(city: str):
    response = requests.get("https://wttr.in/{city}?format=j1")
    return response.json()

@tool("locate_user", description="Look up a user's city based on the context", return_direct=False)
def locate_user(runtime: ToolRuntime[Context]): # toolruntime will inject user_id as a dependency within the Context object
    match runtime.context.user_id:
        case "ABC123":
            return "Karachi"
        case "DEF456":
            return "Vienna"
        case "GHI789":
            return "Tokyo"
        case _: # default if no match
            return "Unknown" # can change this logic to handle unknown values
        

model = init_chat_model("gpt-4.1-mini", temperature=0.1)
checkpoint_model = InMemorySaver() # keep track of conversations


agent = create_agent(
    model = model, # needs openai api key in env, and package langchain[openai],
    tools = [get_weather, locate_user], 
    system_prompt= "You are a helpful weather assistant that provides reliable information in a concise way",
    context_schema = Context,
    response_format = ResponseFormat,
    checkpointer = checkpoint_model
)

config = {"configurable": {"thread_id": 1}}


agent_response = agent.invoke({
    'messages': [
        {"role": "user", "content": "What is the weather like?"}
    ]},
    config=config,
    context=Context(user_id='GHI789')
)

# print(agent_response)
# print(agent_response["messages"][-1].content)

print(agent_response["structured_response"])