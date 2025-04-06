# Building a LangGraph Agent for Email Processing

Looking at the code and documentation, I can see you're building an email processing agent using LangGraph. Let me explain the key components and architecture of this system.

## Agent Architecture

Your email agent is structured as a graph with two main nodes:

*   **Agent Node** - The "brain" that analyzes emails and decides what to do
*   **Tool Node** - Executes actions requested by the agent

The flow works like this:

## Key Components

### 1. Tools Available to the Agent

The agent has access to these tools:

*   `determine_email_action`: Gets routing guidelines for unclear emails.
*   `forward_email`: Forwards emails internally.
*   `send_wrong_email_notification_to_sender`: Notifies senders of the correct address.
*   `extract_notice_data`: Processes regulatory notices using the `NOTICE_EXTRACTION_GRAPH`.

These tools allow the agent to:

*   Get routing guidelines for different email types
*   Forward emails to appropriate departments
*   Send notifications to senders who used the wrong email address
*   Extract and process regulatory notices (which triggers your `NOTICE_EXTRACTION_GRAPH`)

### 2. Agent Model

An OpenAI model (`gpt-4o-mini` by default) is used as the agent's brain. It's configured using `.bind_tools(tools)` so it knows which tools are available and how to use them based on their docstrings and type hints.

---

# Email Processing System with LangGraph

## Project Overview

This project demonstrates a sophisticated email processing system built with LangGraph, a Python library designed for creating stateful, cyclic, and multi-actor Large Language Model (LLM) applications. The system automatically processes incoming emails, identifies their type, and takes appropriate actions based on the content.

## Key Features

*   **Email Classification:** Automatically identifies different types of emails (regulatory notices, invoices, customer support requests, etc.).
*   **Regulatory Notice Processing:** Extracts structured data from regulatory notices including dates, violation types, and compliance deadlines using Pydantic models.
*   **Escalation Handling:** Identifies high-priority notices that require immediate attention based on configurable text and dollar amount criteria.
*   **Email Routing:** Forwards emails to appropriate internal departments and notifies external senders when they've used the wrong email address (simulated).
*   **Stateful Processing:** Maintains context throughout multi-step workflows (like handling follow-up questions during ticket creation) using LangGraph's state management (`GraphState`, `MessagesState`).
*   **Cyclic Workflows:** Demonstrates loops, such as retrying ticket creation after answering required follow-up questions.

## Architecture

The system consists of two main LangGraph components:

1.  **Notice Extraction Graph (`graphs/notice_extraction.py`):** A specialized workflow for processing regulatory notices. It:
    *   Parses the email using `NOTICE_PARSER_CHAIN` to extract structured data into a `NoticeEmailExtract` object.
    *   Evaluates if escalation is needed using `ESCALATION_CHECK_CHAIN` based on text criteria and extracted fine amounts.
    *   Simulates sending escalation emails (`send_escalation_email_node`) if needed.
    *   Simulates creating a legal ticket (`create_legal_ticket_node`), handling a potential cycle of follow-up questions using `BINARY_QUESTION_CHAIN` via the `answer_follow_up_question_node`.
2.  **Email Agent Graph (`graphs/email_agent.py`):** The main entry point for processing emails.
    *   Uses `MessagesState` to track the conversation/processing steps.
    *   The core `agent` node decides the next action using the `EMAIL_AGENT_MODEL`.
    *   Uses a `tool_node` to execute chosen actions (tools):
        *   `forward_email` (simulated)
        *   `send_wrong_email_notification_to_sender` (simulated)
        *   `determine_email_action` (provides routing guidelines)
        *   `extract_notice_data` (invokes the `NOTICE_EXTRACTION_GRAPH`)
    *   Cycles between the `agent` and `tool_node` until processing is complete.

## Installation

### Prerequisites

*   Python 3.9+
*   [Poetry](https://python-poetry.org/docs/#installation) (for dependency management)
*   An OpenAI API key

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url> # Replace with the actual URL
    cd langgraph-project # Or your chosen project name
    ```
2.  **Install dependencies using Poetry:**
    ```bash
    poetry install
    ```
    This command reads the `pyproject.toml` file, creates a virtual environment (if one doesn't exist), and installs the required libraries (`langgraph`, `langchain-openai`, `pydantic`, `python-dotenv`).
3.  **Activate the virtual environment:**
    ```bash
    poetry shell
    ```
4.  **Set up your OpenAI API key:**
    *   Create a file named `.env` in the project root directory (`langgraph-project/`).
    *   Add your API key to the `.env` file like this:
        ```dotenv
        OPENAI_API_KEY=sk-your_actual_api_key_here
        ```
    *   **Important:** Add `.env` to your `.gitignore` file to avoid committing your API key.

## Usage

### Testing Individual Components

You can test the individual chains and the notice extraction graph by running their respective files directly from the project root directory (while the Poetry environment is active):

```bash
# Ensure you are in the project root directory
# and run 'poetry shell' first

python chains/notice_extraction.py
python chains/escalation_check.py
python chains/binary_questions.py
python graphs/notice_extraction.py
```
Observe the output logs to ensure each component behaves as expected.

### Running the Main Email Agent

The primary way to use the system is through the `graphs/email_agent.py` script. It includes a test block (`if __name__ == "__main__":`) that runs through the example emails.

1.  **Run the agent script:**
    ```bash
    # Ensure you are in the project root directory
    # and run 'poetry shell' first

    python graphs/email_agent.py
    ```

2.  **Observe the Output:**
    *   The script will print the type of test case being run (Invoice, Support Request, Regulatory Notice).
    *   It will show the flow of execution through the agent graph's nodes (`--- NODE:` logs) and edges (`--- EDGE:` logs).
    *   You'll see which tools the agent decides to call (`--- TOOL:` logs) and the simulated results.
    *   For regulatory notices, you will see the logs from the nested `NOTICE_EXTRACTION_GRAPH` execution.
    *   The final agent message summarizing the actions taken will be printed for each test case.

### Processing a Custom Email (using Python REPL)

1.  Activate the environment: `poetry shell`
2.  Start Python: `python`
3.  Run the following code, replacing `"Your custom email text here..."`:

    ```python
    from graphs.email_agent import email_agent_graph, MessagesState
    from langchain_core.messages import HumanMessage
    from dotenv import load_dotenv
    import os

    load_dotenv() # Load API key

    custom_email = """
    Subject: Urgent: Safety Concern on Elm Street Site

    Hi Team,

    Reporting an issue at the Elm Street site (Project 98765). Saw workers on the 5th floor without safety harnesses this morning.
    Seems like a major fall risk.

    Please address immediately.

    Thanks,
    Concerned Citizen
    """

    # Add escalation criteria if relevant to the test
    escalation_criteria_for_notice = "Escalate if mentions safety violations, structural issues, or fines over $10,000."
    input_content = f"""
    Please process the following email.
    The escalation criteria for regulatory notices is: {escalation_criteria_for_notice}

    --- Email Start ---
    {custom_email}
    --- Email End ---
    """

    initial_state = MessagesState(messages=[HumanMessage(content=input_content)])

    print("--- Processing Custom Email --- STREAM START ---")
    final_output = None
    for step in email_agent_graph.stream(initial_state, stream_mode="values", config={"recursion_limit": 10}):
        last_msg = step["messages"][-1]
        node_ran = list(step.keys())[0]
        print(f"\n -> Output from Node: {node_ran} | Type: {type(last_msg).__name__}")
        last_msg.pretty_print()
        final_output = step
    print("--- STREAM END ---")

    print("\n--- Final Agent Response ---")
    if final_output:
        final_output['messages'][-1].pretty_print()
    ```

## Project Structure

```
langgraph-project/
├─ chains/                     # LangChain components (LCEL chains)
│  ├─ __init__.py
│  ├─ binary_questions.py   # Chain for yes/no questions during ticketing
│  ├─ escalation_check.py   # Chain to check if notice text needs escalation
│  └─ notice_extraction.py  # Chain to extract structured data from notices
├─ graphs/                     # LangGraph workflows (StateGraphs)
│  ├─ __init__.py
│  ├─ email_agent.py        # Main email processing agent graph
│  └─ notice_extraction.py  # Graph for detailed notice processing & ticketing
├─ utils/                      # Helper functions and configurations
│  ├─ __init__.py
│  ├─ graph_utils.py        # Simulated email sending, ticket creation
│  └─ logging_config.py     # Logging setup
├─ .env                      # Stores API keys (!!! ADD TO .gitignore !!!)
├─ .gitignore                # Specify files to ignore for Git
├─ example_emails.py         # Sample emails for testing
├─ pyproject.toml            # Poetry project configuration and dependencies
└─ README.md                 # This file
```

## Customization

### Escalation Criteria

*   **Text Criteria:** Modify the `escalation_text_criteria` string passed into `NOTICE_EXTRACTION_GRAPH` within the `extract_notice_data` tool in `graphs/email_agent.py`.
*   **Dollar Threshold:** Modify the `escalation_dollar_criteria` value passed into `NOTICE_EXTRACTION_GRAPH` (same location as above), or change the default value within `graphs/notice_extraction.py` if preferred.
*   **Logic:** Adjust the logic within `check_escalation_status_node` in `graphs/notice_extraction.py` for more complex rules.

### Email Routing & Handling

*   **Guidelines:** Update the static guidelines returned by the `determine_email_action` tool in `graphs/email_agent.py`.
*   **Forwarding Addresses:** Change the hardcoded email addresses in the `determine_email_action` guidelines and potentially in the test/example invocations within `graphs/email_agent.py`.
*   **New Email Types:** Add new tools specifically designed to handle other email categories (e.g., a `process_job_application` tool) and update the agent model binding and potentially the `determine_email_action` tool.

### Follow-up Questions

*   Modify the `follow_ups_pool` list within the `create_legal_ticket` function in `utils/graph_utils.py` to change the potential questions asked during ticketing.

### Models

*   Change the underlying LLMs used (e.g., `gpt-4o` instead of `gpt-4o-mini`) by modifying the `ChatOpenAI(...)` instantiations in the `chains/` files and `graphs/email_agent.py`.

## How It Works (Detailed Flow)

1.  **Input:** An email string is provided as a `HumanMessage` in the initial `MessagesState` passed to `email_agent_graph.invoke()` or `.stream()`.
2.  **Agent Entry:** The graph enters at the `agent` node (`call_agent_model_node`).
3.  **Agent Decision:** The `EMAIL_AGENT_MODEL` receives the current message list. Based on the latest message (the input email initially) and the descriptions of the bound tools, it decides:
    *   **Call a Tool:** If it thinks a tool is needed (e.g., it recognizes regulatory language suggesting `extract_notice_data`, or it's unsure and needs `determine_email_action`), it outputs an `AIMessage` containing `tool_calls`.
    *   **Respond Directly:** If it thinks no tools are needed or the task is complete, it outputs a final `AIMessage` without `tool_calls`.
4.  **Routing Edge (`route_agent_graph_edge`):**
    *   If the last message has `tool_calls`, route to the `call_tools` node.
    *   If no `tool_calls`, route to `END`.
5.  **Tool Execution (`call_tools` node):**
    *   The `ToolNode` receives the `tool_calls` from the `AIMessage`.
    *   It executes the corresponding Python functions (e.g., `extract_notice_data(...)`, `forward_email(...)`) with the arguments provided by the agent.
    *   The return value of each executed tool function is packaged into a `ToolMessage`.
6.  **Loop Back:** The graph flows from `call_tools` back to the `agent` node. The `ToolMessage`(s) are appended to the `MessagesState`.
7.  **Agent Re-evaluation:** The `agent` node runs again (`call_agent_model_node`). The `EMAIL_AGENT_MODEL` now receives the original message(s) *plus* the `ToolMessage`(s) containing the results of the previous tool calls. It uses this new context to decide the next action (call another tool, or finish).
8.  **Sub-Graph Invocation (`extract_notice_data`):** If this tool is called:
    *   It prepares an initial `GraphState` for the `NOTICE_EXTRACTION_GRAPH`.
    *   It calls `NOTICE_EXTRACTION_GRAPH.invoke()`.
    *   The notice graph runs its *own* internal nodes and edges (parsing -> escalation check -> maybe email -> ticketing loop -> end).
    *   The final state of the notice graph is returned.
    *   The `extract_notice_data` tool formats a summary of the notice graph's results into a string.
    *   This summary string becomes the content of the `ToolMessage` returned to the main agent.
9.  **Completion:** The loop continues until the agent node outputs an `AIMessage` with no `tool_calls`, at which point the `route_agent_graph_edge` directs the flow to `END`.

## Limitations

*   **Simulation:** Email sending and ticket creation are simulated with `time.sleep()` and logging; they don't interact with external systems.
*   **Attachment Handling:** The system only processes email text content. Attachments are ignored.
*   **LLM Dependence:** Accuracy heavily relies on the LLM's ability to understand the email, follow instructions, choose the correct tools, and provide valid arguments (especially for `extract_notice_data`). Prompt engineering in tool docstrings is crucial.
*   **Error Handling:** Basic error handling exists (e.g., `try...except` in tool calls), but more sophisticated retry logic or fallback paths could be added.
*   **Scalability:** For very high volumes, the synchronous nature and reliance on potentially rate-limited APIs (like OpenAI) might become bottlenecks.

## Future Enhancements

*   **Real Email Integration:** Use `smtplib` (for sending) and `imaplib` or Microsoft Graph API (for receiving) to interact with actual mail servers.
*   **Real Ticketing Integration:** Replace `create_legal_ticket` simulation with API calls to JIRA, ServiceNow, etc.
*   **Attachment Processing:** Add tools using libraries like `pypdf`, `python-docx`, `ocr` to extract text from attachments and include it in the context for the agent.
*   **Configuration:** Move hardcoded values (email addresses, escalation thresholds, model names) to a configuration file (`config.yaml`, `.env`).
*   **Async Operations:** Convert nodes involving I/O (LLM calls, simulated API calls) to use `async` for potential performance improvements.
*   **Human-in-the-Loop:** Add steps where uncertain decisions or outputs are flagged for human review and approval before proceeding.

## License

This project is licensed under the MIT License. (You should create a `LICENSE` file with the MIT License text).

## Acknowledgments

*   Built with [LangGraph](https://github.com/langchain-ai/langgraph)
*   Utilizes concepts and components from [LangChain](https://github.com/langchain-ai/langchain)
*   Powered by OpenAI's language models. 