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

These tools allow the agent to:

*   Get routing guidelines for different email types
*   Forward emails to appropriate departments
*   Send notifications to senders who used the wrong email address
*   Extract and process regulatory notices (which triggers your `NOTICE_EXTRACTION_GRAPH`)

### 2. Agent Model

This binds the tools to the LLM, allowing it to "generate readme text with detailed description of a project, how to use it and all other supporting information"

---

# Email Processing System with LangGraph

## Project Overview

This project demonstrates a sophisticated email processing system built with LangGraph, a Python library designed for creating stateful, cyclic, and multi-actor Large Language Model (LLM) applications. The system automatically processes incoming emails, identifies their type, and takes appropriate actions based on the content.

## Key Features

*   **Email Classification:** Automatically identifies different types of emails (regulatory notices, invoices, customer support requests, etc.)
*   **Regulatory Notice Processing:** Extracts structured data from regulatory notices including dates, violation types, and compliance deadlines
*   **Escalation Handling:** Identifies high-priority notices that require immediate attention based on configurable criteria
*   **Email Routing:** Forwards emails to appropriate departments and notifies senders when they've used the wrong email address
*   **Stateful Processing:** Maintains context throughout multi-step workflows using LangGraph's state management

## Architecture

The system consists of two main components:

1.  **Notice Extraction Graph:** A specialized workflow for processing regulatory notices that:
    *   Extracts structured data from notice emails
    *   Evaluates whether escalation is needed
    *   Sends escalation emails when necessary
    *   Creates legal tickets with extracted information
2.  **Email Agent:** A LangGraph agent that:
    *   Acts as the entry point for all incoming emails
    *   Determines the appropriate action for each email
    *   Forwards non-regulatory emails to the correct department
    *   Invokes the Notice Extraction Graph for regulatory notices

## Installation

### Prerequisites

*   Python 3.9+
*   Poetry (for dependency management)
*   OpenAI API key

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd langgraph-project
    ```
2.  **Create a virtual environment and activate it:**
    ```bash
    poetry install
    poetry shell
    ```
3.  **Set up your OpenAI API key:**
    *   Create a file named `.env` in the project root directory.
    *   Add your API key to the `.env` file:
        ```dotenv
        OPENAI_API_KEY=your-api-key-here
        ```

## Usage

### Test individual chains:

You can test each chain in isolation to ensure they work correctly before testing the full graphs. Each chain file (chains/notice_extraction.py, chains/escalation_check.py, chains/binary_questions.py) includes an if __name__ == "__main__": block.

What to check: Observe the print output. Ensure the chains correctly parse information, check escalation, or answer binary questions based on the test inputs defined within those files. Check for any errors.

```python   
    python chains/notice_extraction.py
    python chains/escalation_check.py
    python chains/binary_questions.py
```

### Processing a Single Email

You can process an email by running a script that invokes the agent graph. (Example script needed here)

```python
# Example: (Assuming you create a main.py or similar)
from graphs.email_agent import email_agent_graph, AgentState
from langchain_core.messages import HumanMessage
from example_emails import EMAILS

# Choose an email to process
email_content = EMAILS[1] # Example: Invoice email

# Prepare the initial state
initial_state = AgentState(messages=[HumanMessage(content=email_content)])

# Invoke the graph
final_state = email_agent_graph.invoke(initial_state)

print("--- Final Agent Response ---")
print(final_state['messages'][-1].content)
```

### Watching the Processing Steps

To see each step the agent takes, you can stream the graph execution:

```python
# Example:
#... (imports and initial_state setup as above) ...

print(f"--- Processing Email ---\n{email_content}\n--- Start Stream ---")

for step in email_agent_graph.stream(initial_state, stream_mode="values"):

last_message = step["messages"][-1]
print(f"\n----- Last Message ({type(last_message).name}) -----")

last_message.pretty_print()
print("----- End Step -----")
print("--- End Stream ---")
```

### Testing with Example Emails

The `example_emails.py` file contains several emails for testing different scenarios:

*   `EMAILS[0]`: OSHA Regulatory Notice (should trigger notice extraction)
*   `EMAILS[1]`: Invoice (should be forwarded to billing)
*   `EMAILS[2]`: Customer Support Request (should be forwarded to support)
*   `EMAILS[3]`: LA Building Dept. Notice (should trigger notice extraction)

## Project Structure
```sql
langgraph-project/
├─ chains/
│ ├─ init.py
│ ├─ binary_questions.py # Chain for yes/no questions
│ ├─ escalation_check.py # Chain to check if notice needs escalation
│ └─ notice_extraction.py # Chain to extract data from notices
├─ graphs/
│ ├─ init.py
│ ├─ email_agent.py # Main email processing agent graph
│ └─ notice_extraction.py # Graph for detailed notice processing
├─ utils/
│ ├─ init.py
│ ├─ graph_utils.py # Helper functions (email sending, ticket creation sims)
│ └─ logging_config.py # Logging setup
├─ .env # Stores API keys (!!! ADD TO .gitignore !!!)
├─ example_emails.py # Sample emails for testing
├─ pyproject.toml # Poetry dependency management
└─ README.md # This file
```

## Customization

### Escalation Criteria

You can customize when notices are escalated by modifying the criteria passed into the `NOTICE_EXTRACTION_GRAPH` when it's invoked by the `extract_notice_data` tool within `graphs/email_agent.py`. You can also adjust the logic within `graphs/notice_extraction.py` (`check_escalation_status_node`).

### Email Routing

To modify where different types of emails are routed, update the `determine_email_action` tool in `graphs/email_agent.py`, specifically the routing guidelines returned by the tool. You might also adjust the logic within the agent node (`call_agent_model_node`) or add more specific tools.

## How It Works

1.  **Email Ingestion:** The system receives an email as input (currently via script).
2.  **Agent Decision:** The `email_agent_graph` starts. The agent node (`call_agent_model_node`) uses the LLM and the available tools' descriptions to decide the first action (e.g., call `determine_email_action` or `extract_notice_data`).
3.  **Tool Execution:** If a tool is chosen, the `tool_node` executes it (e.g., calls the `extract_notice_data` function).
4.  **Notice Extraction (if applicable):** If `extract_notice_data` is called, it invokes the `NOTICE_EXTRACTION_GRAPH`. This graph runs through its own nodes (parsing, escalation check, potential follow-up loop, ticket creation). The final result is returned to the agent.
5.  **Routing/Notification (if applicable):** If the agent determines routing is needed (based on `determine_email_action` or other logic), it calls `forward_email` and/or `send_wrong_email_notification_to_sender`.
6.  **Looping:** The results of tool executions are fed back to the agent node. The agent decides on the next step (call another tool or finish).
7.  **Completion:** Once the agent determines no more tools are needed, it provides a final summary message, and the graph ends.

## Limitations

*   The system currently only processes the text content of emails (no attachment handling).
*   Performance and accuracy depend heavily on the quality and capabilities of the underlying LLM (currently configured for OpenAI's `gpt-4o-mini`). Different models might require prompt adjustments.
*   Email sending and ticket creation are simulated. Real integration would require using actual email libraries (e.g., `smtplib`) and APIs for ticketing systems.
*   Processing time can vary depending on the email complexity and the number of LLM calls and tool executions required.

## Future Enhancements

*   Add support for processing email attachments (PDFs, images, etc.).
*   Integrate with real ticketing systems (JIRA, ServiceNow, Zendesk, etc.) via their APIs.
*   Develop more specialized tools and chains for handling other specific email categories (e.g., job applications, spam filtering).
*   Implement a monitoring and analytics dashboard to track processing volume, accuracy, and common issues.
*   Add more robust error handling and fallback mechanisms.

## License

This project is licensed under the MIT License - see the LICENSE file for details. (Note: You would need to add a LICENSE file).

## Acknowledgments

*   Built with [LangGraph](https://github.com/langchain-ai/langgraph)
*   Powered by [LangChain](https://github.com/langchain-ai/langchain)
*   Uses OpenAI's language models for natural language processing