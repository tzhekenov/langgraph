import time
import json # For printing extracted data nicely
from typing import Annotated, TypedDict, List, Optional # Import List and Optional
import operator # For MessagesState if using the custom approach

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage # Added ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph # Removed START as set_entry_point is used
# Use the prebuilt MessagesState for simplicity
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Use try-except for robust imports relative to project structure
try:
    # Note: Adjusted import path assuming email_agent.py is in the same 'graphs' dir
    from .notice_extraction import NOTICE_EXTRACTION_GRAPH, GraphState as NoticeGraphState # Import the graph and its state
    from utils.logging_config import LOGGER
except ImportError:
    print("Attempting import relative to project root for graphs/email_agent.py...")
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Import graph and its state type alias for clarity
    from graphs.notice_extraction import NOTICE_EXTRACTION_GRAPH
    from graphs.notice_extraction import GraphState as NoticeGraphState
    from utils.logging_config import LOGGER


# Load environment variables (ensure .env is present)
from dotenv import load_dotenv
load_dotenv()

# --- Agent State ---
# Using the prebuilt MessagesState is often simpler
# class AgentState(TypedDict):
#    messages: Annotated[list[BaseMessage], add_messages]

# Using MessagesState prebuilt class
from langgraph.graph import MessagesState

# --- Tools ---

@tool
def forward_email(email_message: str, send_to_email: str) -> str:
    """
    Forward an email_message to the address or comma-separated addresses of send_to_email.
    Returns a success or error message.
    Note: This tool only forwards the email to internal departments - it does not reply to the sender.
    """
    LOGGER.info(f"--- TOOL: Forwarding Email ---")
    LOGGER.info(f"Attempting to forward to: {send_to_email}")
    # Simulate potential multiple recipients if comma-separated
    recipients = [email.strip() for email in send_to_email.split(',') if email.strip()]
    if not recipients:
        LOGGER.warning("No valid recipient email provided.")
        return "Error: No valid recipient email provided."
    try:
        for recipient in recipients:
             LOGGER.info(f"---> Simulating forward to: {recipient}")
             time.sleep(0.5 + random.random() * 0.5) # Simulate network delay per recipient
        LOGGER.info("Email forwarded successfully!")
        return f"Successfully forwarded email to {', '.join(recipients)}."
    except Exception as e:
        LOGGER.error(f"Failed to forward email: {e}", exc_info=True)
        return f"Error: Failed to forward email. Details: {e}"


@tool
def send_wrong_email_notification_to_sender(
    sender_email: str, correct_department: str
) -> str:
    """
    Send an email back to the sender_email informing them that they have the wrong address.
    Inform them the email should be sent to the correct_department address instead.
    """
    LOGGER.info(f"--- TOOL: Sending Wrong Email Notification ---")
    LOGGER.info(f"Attempting to send notification to: {sender_email} about dept: {correct_department}")
    try:
        # Simulate sending email
        time.sleep(1 + random.random())
        LOGGER.info(f"Wrong email notification sent successfully to {sender_email}!")
        return f"Successfully sent wrong email notification to {sender_email}, advising them to use {correct_department}."
    except Exception as e:
        LOGGER.error(f"Failed to send notification: {e}", exc_info=True)
        return f"Error: Failed to send notification. Details: {e}"

@tool
def extract_notice_data(
    email: str,
    escalation_criteria: str = "Escalate if mentions safety violations, structural issues, or fines over $50,000" # Example default
) -> str:
    """
    Extract structured fields from a regulatory notice email using a specialized graph.
    Use ONLY when the email clearly comes from a regulatory body, government agency,
    or auditor regarding a property or construction site.
    Provide the 'escalation_criteria' based on the initial request or default policy.
    After calling this tool, the process is complete for this email. Do not call other tools after this one.
    Returns a summary of the extracted data or an error message.
    """
    LOGGER.info(f"--- TOOL: Extracting Notice Data ---")
    LOGGER.info(f"Using escalation criteria: {escalation_criteria}")
    try:
        # Prepare the initial state for the notice extraction graph
        # Ensure all keys required by NoticeGraphState are present
        initial_state: NoticeGraphState = {
            "notice_message": email,
            "notice_email_extract": None,
            "escalation_text_criteria": escalation_criteria,
            "escalation_dollar_criteria": 50000.0, # Example threshold, could be configurable
            "requires_escalation": False, # Will be set by the graph
            "escalation_emails": ["legal-team@example.com", "compliance-dept@example.com"], # Example emails
            "follow_ups": None,
            "current_follow_up": None,
        }

        # Invoke the notice extraction graph
        # Use stream to observe sub-graph execution if needed, invoke for final result
        LOGGER.info("Invoking NOTICE_EXTRACTION_GRAPH...")
        results = NOTICE_EXTRACTION_GRAPH.invoke(initial_state)
        LOGGER.info("NOTICE_EXTRACTION_GRAPH finished.")

        extracted_data = results.get("notice_email_extract")
        final_follow_ups = results.get("follow_ups")

        # Prepare response string
        response_lines = []
        if extracted_data:
             response_lines.append("Notice data extracted successfully.")
             # Convert Pydantic model to string for agent response
             response_lines.append(extracted_data.model_dump_json(indent=2))
        else:
             response_lines.append("Error: Failed to extract notice data from the email.")

        if final_follow_ups:
            response_lines.append("\nFollow-up questions answered:")
            response_lines.append(json.dumps(final_follow_ups, indent=2))

        if results.get("requires_escalation"):
             response_lines.append("\nNotice required escalation.")
        else:
             response_lines.append("\nNotice did not require escalation.")

        return "\n".join(response_lines)

    except Exception as e:
        LOGGER.error(f"Error calling notice extraction graph: {e}", exc_info=True)
        return f"Error: An exception occurred during notice extraction: {e}"


@tool
def determine_email_action(email: str) -> str:
    """
    Call ONLY as a last resort to determine the action for an email IF AND ONLY IF no other tools
    (like extract_notice_data) seem relevant. Do not call if extract_notice_data was called.
    Provides routing GUIDELINES based on common scenarios. The agent should then call the
    appropriate tools (forward_email, send_wrong_email_notification_to_sender) based on these guidelines.
    """
    LOGGER.info(f"--- TOOL: Determining Email Action (Fallback) ---")
    # In a real scenario, this might involve another LLM call or complex rules.
    # For this tutorial, it returns static guidelines.
    return """
    Routing Guidelines Provided:
    1. Invoice/Billing: If the email appears to be an invoice, billing statement, or payment query:
       - Use 'forward_email' tool to send it ONLY to billing@company.com.
       - Use 'send_wrong_email_notification_to_sender' tool, informing the sender the correct department is billing@company.com.
    2. Customer Support: If the email appears to be from a customer reporting an issue, asking for help, or requesting maintenance/refunds:
       - Use 'forward_email' tool to send it to ALL of these addresses: support@company.com, cdetuma@company.com, ctu@abc.com.
       - Use 'send_wrong_email_notification_to_sender' tool, informing the sender the correct department is support@company.com.
    3. Regulatory Notice: If the email looks like a regulatory notice (from OSHA, Building Dept, etc.) but wasn't caught earlier:
       - Use the 'extract_notice_data' tool.
    4. Other: For emails that don't fit above, attempt to infer the correct department from context (e.g., job application -> humanresources@company.com).
       - If unsure, use 'send_wrong_email_notification_to_sender' suggesting a likely department (e.g., support@company.com or general-info@company.com).
    Provide a brief final response indicating the action taken based *after* calling the necessary tools.
    """

# --- Agent Setup ---

tools = [
    determine_email_action,
    forward_email,
    send_wrong_email_notification_to_sender,
    extract_notice_data,
]

# ToolNode executes tools based on the agent's output
tool_node = ToolNode(tools)

# Agent LLM (bind tools)
EMAIL_AGENT_MODEL = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

# --- Node Functions ---

def call_agent_model_node(state: MessagesState) -> dict[str, List[BaseMessage]]:
    """Node that calls the main LLM agent model."""
    LOGGER.info("--- NODE: Calling Agent Model ---")
    messages = state["messages"]
    # Invoke the LLM with the current conversation history
    # The response will be an AIMessage, potentially with tool_calls
    response = EMAIL_AGENT_MODEL.invoke(messages)
    LOGGER.info(f"Agent model response received. Tool calls: {bool(response.tool_calls)}")
    # Return value adheres to MessagesState structure
    return {"messages": [response]}

# --- Edge Functions ---

def route_agent_graph_edge(state: MessagesState) -> str:
    """Determines whether to continue calling tools or end the graph."""
    LOGGER.info("--- EDGE: Routing Agent Action ---")
    # Get the last message added to the state
    last_message = state["messages"][-1]
    # Check if the last message contains tool calls requested by the LLM
    if last_message.tool_calls:
        # If there are tool calls, route to the tool node
        LOGGER.info(f"Decision: Agent requested {len(last_message.tool_calls)} tool call(s) -> Route to call_tools")
        return "call_tools" # Name of the tool node
    # If no tool calls, the agent has finished (provided a final response)
    LOGGER.info("Decision: Agent finished (no tool calls) -> Route to END")
    return END

# --- Build the Graph ---

LOGGER.info("Building Email Agent Graph...")
workflow = StateGraph(MessagesState) # Use the prebuilt MessagesState

# Add the agent node
workflow.add_node("agent", call_agent_model_node)
# Add the tool execution node
workflow.add_node("call_tools", tool_node)

# Set the entry point: the agent node
workflow.set_entry_point("agent")

# Add the conditional edge: after the agent runs, decide to call tools or end
workflow.add_conditional_edges(
    "agent", # Starting node is the agent
    route_agent_graph_edge, # Function to determine the route
    {
        "call_tools": "call_tools", # Route to tool node if tool calls exist
        END: END # Route to END if no tool calls
    }
)

# Add the edge to loop back from the tool node to the agent node
# After tools run, their output (ToolMessage) is added to state,
# and we go back to the agent to process the tool results.
workflow.add_edge("call_tools", "agent")

# Compile the graph
email_agent_graph = workflow.compile()
LOGGER.info("Email Agent Graph compiled successfully.")

# --- Testing --- (Optional: Keep for standalone testing)
if __name__ == "__main__":
    import random # Ensure random is imported if not already
    try:
        from example_emails import EMAILS
    except ImportError:
        print("Run this script from the project root directory or adjust import path.")
        exit()

    print("\n--- TESTING EMAIL AGENT GRAPH ---")

    test_emails = {
        "Invoice": EMAILS[1],
        "Support Request": EMAILS[2],
        "Regulatory Notice (LA)": EMAILS[3],
        "Regulatory Notice (OSHA)": EMAILS[0],
    }

    for name, email_content in test_emails.items():
        print(f"\n--- Running Test Case: {name} ---")
        print(f"Input Email:\n{email_content[:200]}...") # Print snippet

        # Add escalation criteria specifically for notice tests if needed
        if "Regulatory Notice" in name:
             escalation_criteria = "Escalate if mentions safety violations, structural issues, electrical problems, fire risks, or fines over $10,000."
             input_content = f"""
Please process the following email.
The escalation criteria for regulatory notices is: {escalation_criteria}

--- Email Start ---
{email_content}
--- Email End ---
"""
        else:
            input_content = email_content

        initial_state = MessagesState(messages=[HumanMessage(content=input_content)])
        final_state_agent = None

        # Use stream to observe the flow
        for step in email_agent_graph.stream(initial_state, stream_mode="values", config={"recursion_limit": 10}): # Add recursion limit
            last_msg = step["messages"][-1]
            node_ran = list(step.keys())[0]
            print(f"\n -> Output from Node: {node_ran} | Message Type: {type(last_msg).__name__}")
            # last_msg.pretty_print() # Can be verbose
            if isinstance(last_msg, ToolMessage):
                 print(f"Tool Output: {last_msg.content[:300]}...")
            elif hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                 print(f"Agent wants to call: {[tc['name'] for tc in last_msg.tool_calls]}")
            else:
                 print(f"Agent Response: {last_msg.content[:300]}...")

            final_state_agent = step # Keep track of the last state

        print(f"\n--- Final State (Test Case: {name}) ---")
        if final_state_agent and final_state_agent.get('messages'):
             print("Final Agent Message:")
             final_state_agent['messages'][-1].pretty_print()
        else:
             print("Could not determine final agent state.")
        print("="*30)

    print("\n--- Agent Testing Complete ---") 