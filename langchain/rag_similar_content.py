from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS # vector database
from langchain_core.tools import create_retriever_tool
from langchain.agents import create_agent

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
texts = [
    "I like basketball.",
    "It is time to sleep.",
    "I don't like football.",
    "I like football.",
    "Basketball is cool."
]

vector_store = FAISS.from_texts(texts, embedding=embeddings)
print(vector_store.similarity_search("Basketball is a good sport", k=1)) # top element with highest similarity score
print(vector_store.similarity_search("morning", k=1)) 


retriever = vector_store.as_retriever(search_kwards={"k" : 1})
retriever_tool = create_retriever_tool(retriever, "similarity_search", description="Search the knowledge base for information")

agent = create_agent(
    model = "gpt-4.1-mini",
    tools = [retriever_tool], 
    system_prompt= "Call the similarity_search tool to retrieve cibtext then answer the prompt",
)

agent_response = agent.invoke({
    'messages': [
        {"role": "user", "content": "What sport does the user like and dislike? And what sport would you recommend them to try next?"}
    ]},
)

print(agent_response["messages"][-1].content)