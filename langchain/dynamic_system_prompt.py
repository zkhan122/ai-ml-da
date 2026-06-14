from dotenv import load_dotenv
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, dynamic_prompt

"""
Middleware provides a way to more tightly control what happens inside the agent. Middleware is useful for the following:

    Tracking agent behavior with logging, analytics, and debugging.
    Transforming prompts, tool selection, and output formatting.
    Adding retries, fallbacks, and early termination logic.
    Applying rate limits, guardrails, and PII detection.

"""

load_dotenv()

@dataclass
class Context:
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role

    base_prompt = "You are a helpful and concise assistant"

    match user_role:
        case 'expert':
            return f'{base_prompt}. Provide detailed and technical responses'
        
        case 'beginner':
            return f'{base_prompt}. Provide basic responses'
        
        case 'child':
            return f'{base_prompt}. Provide child-friendly responses'
        
        case _:
            return base_prompt
        
agent = create_agent(
    model="gpt-4.1-mini",
    middleware= [user_role_prompt],
    context_schema= Context
)


agent_response = agent.invoke({
    'messages': [
        {"role": "user", "content": "Can you explain what a pointer is in C?"}
    ]},
    context=Context(user_role="beginner")
)

print(agent_response["messages"][-1].content)

