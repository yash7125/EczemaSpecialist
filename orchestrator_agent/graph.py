import json
import requests
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from orchestrator_agent.prompt import ORCHESTRATOR_PROMPT

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
# Define the A2A tool for consulting the specialist
def consult_specialist(image_path: str, prompt: str = "Analyze the morphology of the lesion in this image.") -> str:
    """Consult the Dermatology Specialist Agent via the A2A protocol. 
    Use this tool ONLY when an image path is provided by the patient.
    
    Args:
        image_path: The local path to the image file provided by the patient.
        prompt: Additional instructions for the specialist (optional).
    """
    import base64
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            
        # A2A tasks/send payload
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "skillId": "analyze_skin_lesion",
                "parameters": {
                    "image_b64": encoded_string,
                    "prompt": prompt
                }
            },
            "id": "1"
        }
        
        # Call the specialist A2A server
        # Default port is 8001 as defined in a2a_server.py
        response = requests.post("http://localhost:8001", json=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        if "error" in data:
            return f"Specialist Agent Error: {data['error']}"
            
        # Assuming the A2A server returns findings in the result object
        # In a real A2A implementation with EventQueue, the client might need to connect to an SSE stream
        # For simplicity in this local HTTP call, we'll assume the server returns the final result synchronously 
        # or we read the immediate response. 
        # (Note: In a pure A2A async setup, this client would use the a2a SDK to invoke the task)
        return json.dumps(data.get("result", data))
        
    except FileNotFoundError:
        return f"Error: Image file not found at path '{image_path}'."
    except Exception as e:
        return f"Error communicating with Specialist Agent: {str(e)}"

def create_diagnosis_graph(llm, mcp_tools: List):
    """Create and compile the LangGraph for the Eczema Orchestrator."""
    
    # Combine our custom A2A tool with the dynamically loaded MCP tools
    all_tools = [consult_specialist] + mcp_tools
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(all_tools)
    
    def chatbot(state: State):
        # Inject system prompt if not present
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=ORCHESTRATOR_PROMPT)] + messages
            
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
        
    # Tool execution node
    tool_node = ToolNode(all_tools)
    
    # Define conditional routing
    def should_continue(state: State):
        messages = state["messages"]
        last_message = messages[-1]
        
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"
            
    # Build graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("agent", chatbot)
    workflow.add_node("action", tool_node)
    
    # Set entry point
    workflow.add_edge(START, "agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END
        }
    )
    
    # Add normal edge from tools back to agent
    workflow.add_edge("action", "agent")
    
    return workflow.compile()
