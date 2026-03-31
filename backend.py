from langchain_google_genai import GoogleGenerativeAI
from langgraph.graph import StateGraph,START,END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

LLM=GoogleGenerativeAI()

class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]


def chatNode(state:ChatState):
    messages=state['messages']
    response=LLM.invoke(messages)
    return {"messages":[response]}

checkpointer=InMemorySaver()

graph=StateGraph(ChatState)
graph.add_node("chatNode",chatNode)
graph.add_edge(START,"chatNode")
graph.add_edge("chatNode",END)

chatbot=graph.compile(checkpointer=checkpointer)
