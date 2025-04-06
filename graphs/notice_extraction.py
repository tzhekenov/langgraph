from typing import TypedDict, Dict, List, Optional # Use concrete types
from pydantic import EmailStr # Ensure EmailStr is imported if used in GraphState
import datetime # Added for default_serializer
import json # Added for default_serializer

# Use try-except for robust imports relative to project structure
try:
    from chains.binary_questions import BINARY_QUESTION_CHAIN
    from chains.escalation_check import ESCALATION_CHECK_CHAIN
    from chains.notice_extraction import NOTICE_PARSER_CHAIN, NoticeEmailExtract
    from utils.graph_utils import create_legal_ticket, send_escalation_email
    from utils.logging_config import LOGGER
except ImportError:
    print("Attempting import relative to project root for graphs/notice_extraction.py...")
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from chains.binary_questions import BINARY_QUESTION_CHAIN
    from chains.escalation_check import ESCALATION_CHECK_CHAIN
    from chains.notice_extraction import NOTICE_PARSER_CHAIN, NoticeEmailExtract
    from utils.graph_utils import create_legal_ticket, send_escalation_email
    from utils.logging_config import LOGGER

from langgraph.graph import END, START, StateGraph

# Load environment variables (should be done in chains, but ensure chains do it)
from dotenv import load_dotenv
load_dotenv()

# Define the state dictionary for the graph
class GraphState(TypedDict):
    notice_message: str
    notice_email_extract: Optional[NoticeEmailExtract]
    escalation_text_criteria: str
    escalation_dollar_criteria: float
    requires_escalation: bool
    escalation_emails: Optional[List[EmailStr]]
    follow_ups: Optional[Dict[str, bool]]
    current_follow_up: Optional[str]

# --- Node Functions ---

def parse_notice_message_node(state: GraphState) -> Dict[str, Optional[NoticeEmailExtract]]:
    """Use the notice parser chain to extract fields from the notice."""
    LOGGER.info("--- NODE: Parsing Notice Message ---")
    try:
        notice_email_extract = NOTICE_PARSER_CHAIN.invoke(
            {"message": state["notice_message"]}
        )
        LOGGER.info(f"Parsing successful. Extracted: {notice_email_extract.model_dump_json(indent=2)}")
        return {"notice_email_extract": notice_email_extract}
    except Exception as e:
        LOGGER.error(f"Error parsing notice message: {e}", exc_info=True)
        return {"notice_email_extract": None}

def check_escalation_status_node(state: GraphState) -> Dict[str, bool]:
    """Determine whether a notice needs escalation based on text and fine amount."""
    LOGGER.info("--- NODE: Checking Escalation Status ---")
    notice_extract = state.get("notice_email_extract")
    if not notice_extract:
        LOGGER.warning("Cannot check escalation: notice_email_extract is missing. Defaulting to False.")
        return {"requires_escalation": False}

    needs_escalation = False
    try:
        # Check text criteria
        text_check = ESCALATION_CHECK_CHAIN.invoke(
            {
                "escalation_criteria": state["escalation_text_criteria"],
                "message": state["notice_message"],
            }
        ).needs_escalation
        LOGGER.info(f"Text escalation check result: {text_check}")

        # Check dollar criteria (only if max_potential_fine exists)
        fine_check = False
        if notice_extract.max_potential_fine is not None:
            fine_check = notice_extract.max_potential_fine >= state["escalation_dollar_criteria"]
            LOGGER.info(f"Fine escalation check result (> {state['escalation_dollar_criteria']}): {fine_check}")
        else:
            LOGGER.info("No maximum potential fine found in notice for escalation check.")

        needs_escalation = text_check or fine_check

    except Exception as e:
        LOGGER.error(f"Error checking escalation status: {e}", exc_info=True)
        needs_escalation = False

    LOGGER.info(f"Final Escalation Required: {needs_escalation}")
    return {"requires_escalation": needs_escalation}

def send_escalation_email_node(state: GraphState) -> Dict:
    """Sends an escalation email if required data is present."""
    LOGGER.info("--- NODE: Sending Escalation Email ---")
    notice_extract = state.get("notice_email_extract")
    escalation_emails = state.get("escalation_emails")

    if notice_extract and escalation_emails:
        send_escalation_email(
            notice_email_extract=notice_extract,
            escalation_emails=escalation_emails,
        )
    else:
        LOGGER.warning("Cannot send escalation email: missing notice_email_extract or escalation_emails in state.")
    return {}

def create_legal_ticket_node(state: GraphState) -> Dict[str, Optional[str]]:
    """Creates a legal ticket, potentially returning a follow-up question."""
    LOGGER.info("--- NODE: Creating Legal Ticket ---")
    notice_extract = state.get("notice_email_extract")
    if not notice_extract:
        LOGGER.warning("Cannot create legal ticket: missing notice_email_extract in state.")
        return {"current_follow_up": None}

    try:
        follow_up = create_legal_ticket(
            current_follow_ups=state.get("follow_ups"),
            notice_email_extract=notice_extract,
        )
        return {"current_follow_up": follow_up}
    except Exception as e:
        LOGGER.error(f"Error creating legal ticket: {e}", exc_info=True)
        return {"current_follow_up": None}

def answer_follow_up_question_node(state: GraphState) -> Dict[str, Optional[Dict[str, bool]]]:
    """Answers follow-up questions about the notice using BINARY_QUESTION_CHAIN."""
    LOGGER.info("--- NODE: Answering Follow-up Question ---")
    current_follow_up = state.get("current_follow_up")
    notice_message = state.get("notice_message")
    current_answers = state.get("follow_ups") or {}

    updated_answers = current_answers.copy()

    if current_follow_up and notice_message:
        LOGGER.info(f"Answering follow-up: '{current_follow_up}'")
        try:
            answer_obj = BINARY_QUESTION_CHAIN.invoke({
                "question": current_follow_up,
                "context": notice_message
                })
            answer = answer_obj.is_true
            updated_answers[current_follow_up] = answer
            LOGGER.info(f"---> Answered '{current_follow_up}': {answer}")
        except Exception as e:
             LOGGER.error(f"Error answering follow-up '{current_follow_up}': {e}", exc_info=True)
             updated_answers[current_follow_up] = None
             LOGGER.warning(f"Could not answer follow-up '{current_follow_up}'. Storing None.")

    else:
        LOGGER.warning("Cannot answer follow-up: missing current_follow_up or notice_message in state.")

    return {"follow_ups": updated_answers, "current_follow_up": None}

# --- Edge Functions ---

def route_escalation_status_edge(state: GraphState) -> str:
    """Determine whether to send an escalation email or create a legal ticket."""
    LOGGER.info("--- EDGE: Routing Escalation Status ---")
    if state.get("requires_escalation", False):
        LOGGER.info("Decision: Escalation needed -> Route to send_escalation_email")
        return "send_escalation_email"
    else:
        LOGGER.info("Decision: No escalation needed -> Route to create_legal_ticket")
        return "create_legal_ticket"

def route_follow_up_edge(state: GraphState) -> str:
    """Determine whether a follow-up question is required from create_legal_ticket."""
    LOGGER.info("--- EDGE: Routing Follow-up Status ---")
    if state.get("current_follow_up"):
        LOGGER.info(f"Decision: Follow-up '{state['current_follow_up']}' received -> Route to answer_follow_up_question")
        return "answer_follow_up_question"
    else:
        LOGGER.info("Decision: No follow-up question -> Route to END")
        return END

# --- Build the Graph ---

LOGGER.info("Building Notice Extraction Graph...")
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("parse_notice_message", parse_notice_message_node)
workflow.add_node("check_escalation_status", check_escalation_status_node)
workflow.add_node("send_escalation_email", send_escalation_email_node)
workflow.add_node("create_legal_ticket", create_legal_ticket_node)
workflow.add_node("answer_follow_up_question", answer_follow_up_question_node)

# Add edges
workflow.set_entry_point("parse_notice_message")
workflow.add_edge("parse_notice_message", "check_escalation_status")

# Conditional edge for escalation
workflow.add_conditional_edges(
    "check_escalation_status",
    route_escalation_status_edge,
    {
        "send_escalation_email": "send_escalation_email",
        "create_legal_ticket": "create_legal_ticket",
    },
)

# Edge after sending email (if needed)
workflow.add_edge("send_escalation_email", "create_legal_ticket")

# Conditional edge AFTER creating ticket - determines cycle or end
workflow.add_conditional_edges(
    "create_legal_ticket",
    route_follow_up_edge,
    {
        "answer_follow_up_question": "answer_follow_up_question",
        END: END,
    },
)

# Edge AFTER answering follow-up - ALWAYS go back to try creating ticket again
workflow.add_edge("answer_follow_up_question", "create_legal_ticket")

# Compile the graph
NOTICE_EXTRACTION_GRAPH = workflow.compile()
LOGGER.info("Notice Extraction Graph compiled successfully.")


# --- Testing --- (Optional: Keep for standalone testing)
if __name__ == "__main__":
    try:
        from example_emails import EMAILS
    except ImportError:
        print("Run this script from the project root directory or adjust import path.")
        exit()

    print("\n--- TESTING NOTICE_EXTRACTION_GRAPH --- ")

    test_state_1 = {
        "notice_message": EMAILS[0],
        "notice_email_extract": None,
        "escalation_text_criteria": "Workers explicitly violating safety protocols",
        "escalation_dollar_criteria": 20000.0,
        "requires_escalation": False,
        "escalation_emails": ["manager1@example.com", "ceo@example.com"],
        "follow_ups": None,
        "current_follow_up": None,
    }

    print("\n--- Running Test Case 1 (Should Escalate & Cycle) --- ")
    final_state_1 = None
    for event in NOTICE_EXTRACTION_GRAPH.stream(test_state_1, stream_mode="values"):
        print("\n--- Graph Step Output --- Key: ", list(event.keys())[0]) # Show node that produced output
        final_state_1 = event
        pass

    print("\n--- Final State (Test Case 1) ---")
    def default_serializer(obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        if hasattr(obj, 'model_dump_json'):
             return json.loads(obj.model_dump_json())
        return str(obj)

    print(json.dumps(final_state_1, indent=2, default=default_serializer))

    print("\n--- Running Test Case 2 (Should NOT Escalate) ---")
    test_state_2 = {
        "notice_message": EMAILS[1],
        "notice_email_extract": None,
        "escalation_text_criteria": "Mentions fire or structural damage",
        "escalation_dollar_criteria": 1000000.0,
        "requires_escalation": False,
        "escalation_emails": ["manager1@example.com"],
        "follow_ups": None,
        "current_follow_up": None,
    }
    final_state_2 = None
    for event in NOTICE_EXTRACTION_GRAPH.stream(test_state_2, stream_mode="values"):
        print("\n--- Graph Step Output --- Key: ", list(event.keys())[0]) # Show node that produced output
        final_state_2 = event
        pass

    print("\n--- Final State (Test Case 2) ---")
    print(json.dumps(final_state_2, indent=2, default=default_serializer)) 