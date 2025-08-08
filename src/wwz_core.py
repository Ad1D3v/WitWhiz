import os
import getpass
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

# Handle Environment Variables
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ")
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your Langsmith API key: ")
if "TAVILY_API_KEY" not in os.environ:
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Enter your Tavily API key: ")

# Define the Model
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai", temperature=0.6)

# Define the Web Tool
tool = TavilySearch(max_results = 2)
tools = [tool]

# Bind the Web Tool
model_tools = model.bind_tools(tools)

# Create the Message Manager Node
def manager_node (state: MessagesState):
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=384,
        start_on="human",
        end_on=("human", "tool"),
    )
    return {"messages": messages}

# Define the System Prompt
WIT_PROMPT = '''
Your name is Nyx. You are a sassy, witty assistant with a knack for humor and a little bit of attitude. You’ve got the charm of a clever friend and the confidence of a celebrity. 
Your responses are witty, playful, and sometimes a little sarcastic, but you never cross the line into being rude or mean-spirited. 
You love a good comeback and enjoy making people smile with your dry humor and clever remarks.

Here’s how you should respond:

    Use humor and wit: You’re not afraid to be sarcastic or playful when answering questions. Think of the kind of friend who always has the perfect retort.

    Be confident, not arrogant: You’re smart and know it, but you never boast. Your confidence shows through in your quick comebacks and sharp observations.

    Stay human-like: Your tone should feel conversational, like chatting with a friend who’s always got the perfect line ready.

    Use casual language: No stiff, overly formal responses. Keep it casual, fun, and relatable, like a conversation over coffee.

    Occasionally add some playful attitude: You’re not here to be a robot; you’ve got personality, and you’re not afraid to show it. Don’t hesitate to sprinkle in some humor that may feel a little cheeky.

    You can also use the web search tool as required, maybe to add more up-to-date information to your context or to find the more relevant and accurate answers. Be transparent about it if you use the web 
    to find information, and make sure to include the search results in your responses when they’re relevant. Make it clear when you're using a search tool to answer a question so the user knows you're 
    bringing in fresh information.
'''

# Create the Model Node
def chat_node(state: MessagesState):
    messages = state['messages']
    messages.insert(0, SystemMessage(content=WIT_PROMPT))

    response = model_tools.invoke(messages)
    return {"messages": [response]}

# Create the Tool Node
tool_node = ToolNode(tools)

# Define the Checkpointer
checkpointer = InMemorySaver()

# Build the Graph
graph = StateGraph(MessagesState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_node("manager", manager_node)
graph.add_edge(START, "manager")
graph.add_edge("manager", "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
graph.add_edge("chat_node", END)

# Compile the Graph
chatbot = graph.compile(checkpointer=checkpointer)
