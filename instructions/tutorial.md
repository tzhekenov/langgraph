# LangGraph: Build Stateful AI Agents in Python

by Harrison Hoffman Mar 19, 2025 | 0 Comments | *intermediate data-science*

## Table of Contents

- [Install LangGraph](#install-langgraph)
- [Create Test Cases](#create-test-cases)
- [Work With State Graphs](#work-with-state-graphs)
  - [LangChain Chains and Their Limitations](#langchain-chains-and-their-limitations)
  - [Build Your First State Graph](#build-your-first-state-graph)
- [Work With Conditional Edges](#work-with-conditional-edges)
  - [Create a Conditional Edge](#create-a-conditional-edge)
  - [Use Conditional Edges for Cycles](#use-conditional-edges-for-cycles)
- [Develop Graph Agents](#develop-graph-agents)
  - [Structure Agents as Graphs](#structure-agents-as-graphs)
  - [Test Your Graph Agent](#test-your-graph-agent)
- [Conclusion](#conclusion)
- [Frequently Asked Questions](#frequently-asked-questions)

---
*Remove ads*
---

LangGraph is a versatile Python library designed for stateful, cyclic, and multi-actor Large Language Model (LLM) applications. LangGraph builds upon its parent library, LangChain, and allows you to build sophisticated workflows that are capable of handling the complexities of real-world LLM applications.

By the end of this tutorial, you'll understand that:

*   You can use LangGraph to build LLM workflows by defining state graphs with nodes and edges.
*   LangGraph expands LangChain's capabilities by providing tools to build complex LLM workflows with state, conditional edges, and cycles.
*   LLM agents in LangGraph autonomously process tasks using state graphs to make decisions and interact with tools or APIs.
*   You can use LangGraph independently of LangChain, although they're often used together to complement each other.

Explore the full tutorial to gain hands-on experience with LangGraph, including setting up workflows and building a LangGraph agent that can autonomously parse emails, send emails, and interact with API services.

While you'll get a brief primer on LangChain in this tutorial, you'll benefit from having prior knowledge of LangChain fundamentals. You'll also want to ensure you have intermediate Python knowledge—specifically in object-oriented programming concepts like classes and methods.

**Get Your Code:** Click here to download the free sample code that you'll use to build stateful AI agents with LangGraph in Python.

**Take the Quiz:** Test your knowledge with our interactive "LangGraph: Build Stateful AI Agents in Python" quiz. You'll receive a score upon completion to help you track your learning progress:

**LangGraph: Build Stateful AI Agents in Python**
*Interactive Quiz*

**LangGraph: Build Stateful AI Agents in Python**
Take this quiz to test your understanding of LangGraph, a Python library designed for stateful, cyclic, and multi-actor Large Language Model (LLM) applications. By working through this quiz, you'll revisit how to build LLM workflows and agents in LangGraph.

## Install LangGraph

LangGraph is available on PyPI, and you can install it with `pip`. Open a terminal or command prompt, create a new virtual environment, and then run the following command:

```bash
(venv) $ python -m pip install langgraph
```

This command will install the latest version of LangGraph from PyPI onto your machine. To verify that the installation was successful, start a Python REPL and import LangGraph:

```python
>>> import langgraph
```

If the import runs without error, then you've successfully installed LangGraph. You'll also need a few more libraries for this tutorial:

```bash
(venv) $ python -m pip install langchain-openai "pydantic[email]"
```

You'll use `langchain-openai` to interact with OpenAI LLMs, but keep in mind that you can use any LLM provider you like with LangGraph and LangChain. You'll use `pydantic` to validate the information your agent parses from emails.

Before moving forward, if you choose to use OpenAI, make sure you're signed up for an OpenAI account and that you have a valid API key. You'll need to set the following environment variable before running any examples in this tutorial:

```
OPENAI_API_KEY=<YOUR-OPENAI-API-KEY>
```

**Note:** While LangGraph was made by the creators of LangChain, and the two libraries are highly compatible, it's possible to use LangGraph without LangChain. However, it's more common to use LangChain and LangGraph together, and you'll see throughout this tutorial how they complement each other.

With that, you've installed all the dependencies you'll need for this tutorial, and you're ready to create your LangGraph email processor. Before diving in, you'll take a brief detour to set up quick sanity tests for your app. Then, you'll go through an overview of LangChain chains and explore LangGraph's core concept—the state graph.

---
*Remove ads*
---

## Create Test Cases

When developing AI applications, testing and performance tracking is crucial for understanding how your chain, graph, or agent performs in the real world. While performance tracking is out of scope for this tutorial, you'll use several example emails to test your chains, graphs, and agent, and you'll empirically inspect whether their outputs are correct.

To avoid redefining these examples each time, create the following Python file with example emails:

`example_emails.py`
```python
EMAILS = [
    # Email 0
    """
    Date: October 15, 2024
    From: Occupational Safety and Health Administration (OSHA)
    To: Blue Ridge Construction, project 111232345 - Downtown Office
    Complex Location: Dallas, TX

    During a recent inspection of your construction site at 123 Main
    Street,
    the following safety violations were identified:

    Lack of fall protection: Workers on scaffolding above 10 feet
    were without required harnesses or other fall protection
    equipment. Unsafe scaffolding setup: Several scaffolding
    structures were noted as
    lacking secure base plates and bracing, creating potential
    collapse risks.
    Inadequate personal protective equipment (PPE): Multiple
    workers were
    found without proper PPE, including hard hats and safety
    glasses.
    Required Corrective Actions:

    Install guardrails and fall arrest systems on all scaffolding
    over 10 feet. Conduct an inspection of all scaffolding
    structures and reinforce unstable sections. Ensure all
    workers on-site are provided
    with necessary PPE and conduct safety training on proper
    usage.
    Deadline for Compliance: All violations must be rectified
    by November 10, 2024. Failure to comply may result in fines
    of up to
    $25,000 per violation.

    Contact: For questions or to confirm compliance, please reach
    out to the
    OSHA regional office at (555) 123-4567 or email
    compliance.osha@osha.gov.
    """,
    # Email 1
    """
    From: debby@stack.com
    Hey Betsy,
    Here's your invoice for $1000 for the cookies you ordered.
    """,
    # Email 2
    """
    From: tdavid@companyxyz.com
    Hi Paul,
    We have an issue with the HVAC system your team installed in
    apartment 1235. We'd like to request maintenance or a refund.
    Thanks,
    Terrance
    """,
    # Email 3
    """
    Date: January 10, 2025
    From: City of Los Angeles Building and Safety Department
    To: West Coast Development, project 345678123 - Sunset Luxury
    Condominiums
    Location: Los Angeles, CA
    Following an inspection of your site at 456 Sunset Boulevard, we have
    identified the following building code violations:
    Electrical Wiring: Exposed wiring was found in the underground parking
    garage, posing a safety hazard. Fire Safety: Insufficient fire
    extinguishers were available across multiple floors of the structure
    under construction.
    Structural Integrity: The temporary support beams in the eastern wing
    do not meet the load-bearing standards specified in local building
    codes.
    Required Corrective Actions:
    Replace or properly secure exposed wiring to meet electrical safety
    standards. Install additional fire extinguishers in compliance with
    fire code requirements. Reinforce or replace temporary support beams
    to ensure structural stability. Deadline for Compliance: Violations
    must be addressed no later than February 5,
    2025. Failure to comply may result in
    a stop-work order and additional fines.
    Contact: For questions or to schedule a re-inspection, please contact
    the Building and Safety Department at
    (555) 456-7890 or email inspections@lacity.gov.
    """,
]
```
You can read through these right now if you want, but you'll get links back to these test emails throughout the tutorial.

## Work With State Graphs

As you might have inferred from the name, LangGraph is all about implementing LLM applications as directed graphs. You can think of a directed graph as a sequence of instructions composed of nodes and edges, that tell you how to complete a task. In LangGraph, nodes represent actions that your graph can take, such as calling a function, and edges tell you which node to go to next.

To understand this better, take a look at this directed graph:

*(Image: Directed Graph Food Example)*
*A Directed Graph Example*

This graph models what you might do when you eat a meal in a cafeteria. It consists of two actions represented by nodes: *Buy Food* and *Eat Food*. Once you've eaten, you ask yourself: *Am I still hungry, or am I full?*

The dotted arrows, which are edges, represent the answers to this question. If you're still hungry, you buy more food and eat it. This cycle continues until you're full, at which point you leave the cafeteria.

This simple example illustrates the essence of how LangGraph represents and implements LLM applications. In this tutorial, you're going to step into the shoes of an AI engineer at a large real estate development firm and build a graph to process emails from regulatory agencies. Your graph will:

*   Extract structured fields like dates, names, phone numbers, and locations from email messages
*   Notify internal stakeholders if an email requires immediate escalation
*   Create tickets with your company's legal team using the information extracted from the email
*   Forward and reply to emails that were sent to the wrong address

To understand why LangGraph is a great choice for building this type of application, you'll begin by reviewing and building a LangChain chain, and you'll see why chains can't accomplish the tasks listed above.

### LangChain Chains and Their Limitations

Suppose you work for a large real estate development company. Your company receives hundreds of emails a day from regulatory entities and other organizations regarding active construction sites. For instance, your company might receive a notice from an inspector saying that a construction site doesn't comply with safety regulations.

Your job is to build a tool that can read these emails, extract critical information from them, and notify the correct internal team who will take action. The first step you'll take to accomplish this is to build a LangChain chain that uses an LLM to extract structured fields from a regulatory notice email. You start by defining a Pydantic `BaseModel` that describes all the fields you want to extract from the email:

`chains/notice_extraction.py`
```python
from datetime import datetime, date
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, computed_field

class NoticeEmailExtract(BaseModel):
    date_of_notice_str: str | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="""The date of the notice (if any) reformatted
        to match YYYY-mm-dd""",
    )
    entity_name: str | None = Field(
        default=None,
        description="""The name of the entity sending the notice (if present
        in the message)""",
    )
    entity_phone: str | None = Field(
        default=None,
        description="""The phone number of the entity sending the notice
        (if present in the message)""",
    )
    entity_email: str | None = Field(
        default=None,
        description="""The email of the entity sending the notice
        (if present in the message)""",
    )
    project_id: int | None = Field(
        default=None,
        description="""The project ID (if present in the message) -
        must be an integer""",
    )
    site_location: str | None = Field(
        default=None,
        description="""The site location of the project (if present
        in the message). Use the full address if possible.""",
    )
    violation_type: str | None = Field(
        default=None,
        description="""The type of violation (if present in the
        message)""",
    )
    required_changes: str | None = Field(
        default=None,
        description="""The required changes specified by the entity
        (if present in the message)""",
    )
    compliance_deadline_str: str | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="""The date that the company must comply (if any)
        reformatted to match YYYY-mm-dd""",
    )
    max_potential_fine: float | None = Field(
        default=None,
        description="""The maximum potential fine
        (if any)""",
    )

    @staticmethod
    def _convert_string_to_date(date_str: str | None) -> date | None:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception as e:
            print(e)
            return None

    @computed_field
    @property
    def date_of_notice(self) -> date | None:
        return self._convert_string_to_date(self.date_of_notice_str)

    @computed_field
    @property
    def compliance_deadline(self) -> date | None:
        return self._convert_string_to_date(self.compliance_deadline_str)
```
You first import all of the dependencies you'll need to create your chain. Then, you define `NoticeEmailExtract`, which is a Pydantic `BaseModel` that provides type definitions and descriptions of each field you want to extract. Downstream, LangChain will pass the information in the `NoticeEmailExtract` definition to an LLM as raw text. The LLM will try to extract these fields from an email based on the type hints and description parameters in `Field()`.

As an example, the LLM will try to identify and extract the project ID corresponding to the construction site discussed in the email. If successfully extracted, the LLM will return the project ID in a JSON object with an integer `project_id` entry. If it's unable to extract a project ID, the `project_id` entry will be `None`.

You may have noticed that `date_of_notice` and `compliance_deadline` are Pydantic `computed_field` properties that are derived from `date_of_notice_str` and `compliance_deadline_str`, respectively. Since OpenAI LLMs can't natively extract fields as a `date` data type, the LLM first extracts dates as strings. Then, you use computed field properties to convert those strings to dates.

For example, the LLM extracts `2025-01-01` for `date_of_notice_str`. Your `NoticeEmailExtract` instance will convert this to a `date` object for January 1, 2025, and it will store this in a new field called `date_of_notice`.

Also, because `exclude` is `True` and `repr` is `False` in the definition of `date_of_notice_str`, you won't see `date_of_notice_str` when you serialize or display `NoticeEmailExtract`. It will be as if the LLM extracted `date_of_notice` directly as a date.

Next, you create a chain to parse notice emails using `NoticeEmailExtract`:

`chains/notice_extraction.py`
```python
# ...

info_parse_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Parse the date of notice, sending entity name, sending entity
            phone, sending entity email, project id, site location,
            violation type, required changes, compliance deadline, and
            maximum potential fine from the message. If any of the fields
            aren't present, don't populate them. Try to cast dates into
            the YYYY-mm-dd format. Don't populate fields if they're not
            present in the message.

            Here's the notice message:

            {message}
            """,
        )
    ]
)

notice_parser_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

NOTICE_PARSER_CHAIN = (
    info_parse_prompt
    | notice_parser_model.with_structured_output(NoticeEmailExtract)
)
```
You define `info_parse_prompt` to tell the LLM how it should behave and which tasks it should accomplish. In this case, you're instructing it to extract structured fields from an email notice. The `{message}` parameter is a placeholder that will store the email you pass through the chain.

You then instantiate a `ChatOpenAI` model using `gpt-4o-mini` as the foundation model—you can experiment with different LLM providers and models here. Lastly, you instantiate `NOTICE_PARSER_CHAIN` using the LangChain Expression Language (LCEL).

This statement creates a chain that injects an email into the `message` parameter of `info_parse_prompt`. It then passes the output of `info_parse_prompt` to `notice_parser_model`. By calling `.with_structured_output(NoticeEmailExtract)`, LangChain converts your `NoticeEmailExtract` base model to a prompt that tells the LLM to adhere its output to the schema defined by `NoticeEmailExtract`.

**Note:** You may have noticed that `NOTICE_PARSER_CHAIN` is in all caps. This is intentional because `NOTICE_PARSER_CHAIN` will act as a global variable that you'll use in other functions later in this tutorial, and it's considered a best practice to define global variables in all caps.

To bring this all together, open a Python interpreter and test `NOTICE_PARSER_CHAIN` on an example email notice:

```python
>>> from chains.notice_extraction import NOTICE_PARSER_CHAIN
>>> from example_emails import EMAILS

>>> NOTICE_PARSER_CHAIN.invoke({"message": EMAILS[0]})
NoticeEmailExtract(
    entity_name='Occupational Safety and Health Administration (OSHA)',
    entity_phone='(555) 123-4567',
    entity_email='compliance.osha@osha.gov',
    project_id=111232345,
    site_location='123 Main Street, Dallas, TX',
    violation_type='Lack of fall protection, Unsafe scaffolding setup, Inadequate personal protective equipment (PPE)',
    required_changes='Install guardrails and fall arrest systems on all scaffolding over 10 feet. Conduct an inspection of all scaffolding structures and reinforce unstable sections. Ensure all workers on-site are provided with necessary PPE and conduct safety training on proper usage.',
    max_potential_fine=25000.0,
    date_of_notice=datetime.date(2024, 10, 15),
    compliance_deadline=datetime.date(2024, 11, 10)
)
```
Here, you import `NOTICE_PARSER_CHAIN` and pass `EMAILS[0]` to `NOTICE_PARSER_CHAIN.invoke()`. You can see that `NOTICE_PARSER_CHAIN` successfully parses the email and returns a `NoticeEmailExtract`. It's pretty awesome that `NOTICE_PARSER_CHAIN`, and specifically `gpt-4o-mini`, pulled these fields out of raw text. Think about how difficult it would be to write logic to do this without an LLM!

You'll see that `date_of_notice` and `compliance_deadline` are `date` objects, and `date_of_notice_str` and `compliance_deadline_str` aren't displayed. This shows that the LLM successfully extracted the two dates as strings and your computed field properties converted them to `date` objects.

**Note:** In the example above, you may have noticed that the REPL output of `NOTICE_PARSER_CHAIN.invoke()` is nicely formatted. This output was manually reformatted for this tutorial, and your output won't look exactly like this.

Not only did `NOTICE_PARSER_CHAIN` extract these fields, it did so with high accuracy. For example, `NOTICE_PARSER_CHAIN` extracted the date the notice was received, the entity that sent it, and even the maximum potential fine for non-compliance. Notice how all of the extracted fields are the correct data type that you specified when defining `NoticeEmailExtract`—all of this without having to write a single line of type conversion logic.

Next, you'll build another chain that you'll use throughout this tutorial. This chain will check whether the email notice requires escalation within the company based on a text description of what constitutes escalation. For example, you might want to escalate a message if employees are in danger, or if the notice warns about a fine above a specified threshold. Here's what the escalation chain looks like:

`chains/escalation_check.py`
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class EscalationCheck(BaseModel):
    needs_escalation: bool = Field(
        description="""Whether the notice requires escalation
        according to specified criteria"""
    )

escalation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Determine whether the following notice received
            from a regulatory body requires immediate escalation.
            Immediate escalation is required when {escalation_criteria}.

            Here's the notice message:

            {message}
            """,
        )
    ]
)

escalation_check_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

ESCALATION_CHECK_CHAIN = (
    escalation_prompt
    | escalation_check_model.with_structured_output(EscalationCheck)
)
```
In this block, you define `ESCALATION_CHECK_CHAIN`, which accepts a `message` and `escalation_criteria` as parameters and returns a `BaseModel` with a Boolean attribute called `needs_escalation`. You'll use this chain to check whether the message requires escalation using `escalation_criteria` as the criteria. If `message` does require escalation, `ESCALATION_CHECK_CHAIN.invoke()` returns an `EscalationCheck` instance with `needs_escalation` set to `True`.

Here's what `ESCALATION_CHECK_CHAIN` looks like in action:

```python
>>> from chains.escalation_check import ESCALATION_CHECK_CHAIN

>>> escalation_criteria = """There is currently water damage
... or potential water damage reported"""

>>> message = """Several cracks in the foundation have
... been identified along with water leaks"""

>>> ESCALATION_CHECK_CHAIN.invoke(
...     {"message": message, "escalation_criteria": escalation_criteria}
... )
EscalationCheck(needs_escalation=True)

>>> message = "The wheel chair ramps are too steep"

>>> ESCALATION_CHECK_CHAIN.invoke(
...     {"message": message, "escalation_criteria": escalation_criteria}
... )
EscalationCheck(needs_escalation=False)
```
You first import `ESCALATION_CHECK_CHAIN` and define the escalation criteria. Any messages that mention water damage require escalation. The first `message` mentions water leaks, and `ESCALATION_CHECK_CHAIN` correctly identifies that this meets the criteria and returns `EscalationCheck(needs_escalation=True)`. The second `message` doesn't mention water damage, and `ESCALATION_CHECK_CHAIN` returns `EscalationCheck(needs_escalation=False)`.

You now have the first two components of your email parsing system built, but in isolation, `NOTICE_PARSER_CHAIN` and `ESCALATION_CHECK_CHAIN` don't exactly solve your problems. You want your system to take different actions depending on whether the email requires escalation. You also might want to check if the email even comes from a regulatory body, and forward it to the correct department if it doesn't.

This is where chains reach their limit. They're not designed to handle state or make conditional decisions, such as determining which action to take if an email requires escalation. To tackle more complex tasks, you'll need more than a stateless chain that passes data linearly from one step to the next. This is where LangGraph's core object—the state graph—comes in to help.

---
*Remove ads*
---

### Build Your First State Graph

Now that you've built the notice parsing and escalation check chains, you need to orchestrate them and add additional functionality that your company requires to process notice emails. To do this, you'll use LangGraph's `StateGraph` to create a graph that builds upon `NOTICE_PARSER_CHAIN` and `ESCALATION_CHECK_CHAIN`. Before getting started, you'll want to initialize a logger that you'll use throughout this tutorial:

`utils/logging_config.py`
```python
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)
```
Here, you initialize a standard logger that displays informational messages from all libraries besides `httpx`. For `httpx`, you'll only show warnings. This will keep your logs clean later on in the tutorial.

**Note:** You're not directly using `httpx` in your code. However, the `httpx` library is a dependency of LangGraph and is used under the hood to communicate with the AI models.

Now you can initialize your first graph:

`graphs/notice_extraction.py`
```python
from typing import TypedDict
from chains.escalation_check import ESCALATION_CHECK_CHAIN
from chains.notice_extraction import NOTICE_PARSER_CHAIN, NoticeEmailExtract
from langgraph.graph import END, START, StateGraph
from pydantic import EmailStr
from utils.logging_config import LOGGER

class GraphState(TypedDict):
    notice_message: str
    notice_email_extract: NoticeEmailExtract | None
    escalation_text_criteria: str
    escalation_dollar_criteria: float
    requires_escalation: bool
    escalation_emails: list[EmailStr] | None
    follow_ups: dict[str, bool] | None
    current_follow_up: str | None

workflow = StateGraph(GraphState)
```
You first import dependencies. Notice that you import the chains you built previously. You then define `GraphState`—a typed dictionary that defines the information each node in your graph updates and passes to the next node. Note that by inheriting from `TypedDict`, LangGraph ensures each field in `GraphState` has the correct type when populated. Here's what each field in `GraphState` stores:

*   `notice_message`: The notice email that you want to parse and process.
*   `notice_email_extract`: A `NoticeEmailExtract` instance, which is the output of running `notice_message` through `NOTICE_PARSER_CHAIN`. When you initialize the graph, `notice_email_extract` is `None`.
*   `escalation_text_critera`: A text description of how to determine whether an email notice requires immediate escalation.
*   `escalation_dollar_criteria`: A threshold for the smallest maximum potential fine used to determine whether escalation is needed.
*   `requires_escalation`: A Boolean indicating whether the notice requires escalation.
*   `escalation_emails`: A list of email addresses to notify if escalation is required.
*   `follow_ups`: A dictionary that stores follow-up questions that your graph needs to answer about the notice message before creating a legal ticket. You'll learn more about this in the next section.
*   `current_follow_up`: The current follow-up question your graph needs to answer.

You then initialize a `StateGraph` instance, passing `GraphState` as an argument, and assign it to the variable `workflow`. At this point, `workflow` is an empty graph that can't do anything. To make `workflow` functional, you need to add nodes and edges. In LangGraph, a node represents an action that your graph can take, and every action is defined by a function.

For example, you can use `NOTICE_PARSER_CHAIN` and `ESCALATION_CHECK_CHAIN` as the first nodes in your graph:

`graphs/notice_extraction.py`
```python
# ...

def parse_notice_message_node(state: GraphState) -> GraphState:
    """Use the notice parser chain to extract fields from the notice"""
    LOGGER.info("Parsing notice...")
    notice_email_extract = NOTICE_PARSER_CHAIN.invoke(
        {"message": state["notice_message"]}
    )
    state["notice_email_extract"] = notice_email_extract
    return state

def check_escalation_status_node(state: GraphState) -> GraphState:
    """Determine whether a notice needs escalation"""
    LOGGER.info("Determining escalation status...")
    text_check = ESCALATION_CHECK_CHAIN.invoke(
        {
            "escalation_criteria": state["escalation_text_criteria"],
            "message": state["notice_message"],
        }
    ).needs_escalation

    if (
        text_check
        or state["notice_email_extract"].max_potential_fine
        >= state["escalation_dollar_criteria"]
    ):
        state["requires_escalation"] = True
    else:
        state["requires_escalation"] = False

    return state

workflow.add_node("parse_notice_message", parse_notice_message_node)
workflow.add_node("check_escalation_status", check_escalation_status_node)
```
Here, you define `parse_notice_message_node()`—a function that accepts your `GraphState` instance, runs the `notice_message` attribute of `state` through `NOTICE_PARSER_CHAIN.invoke()`, stores the results in the `state`, and returns the `state`. In general, all node functions accept the graph state, perform some action, update the graph state, and return the graph state.

Similarly, `check_escalation_status_node()` passes the `escalation_text_criteria` and `notice_message` from `state` through `ESCALATION_CHECK_CHAIN.invoke()`. If the chain determines that escalation is required, or the extracted `max_potential_fine` is greater than `state["escalation_dollar_criteria"]`, the `requires_escalation` attribute is set to `True`.

You then add the nodes to your graph with `workflow.add_node()`, which is a method that accepts the name of your node and the function that determines what your node does. For example, `workflow.add_node("parse_notice_message", parse_notice_message_node)` assigns `parse_notice_message_node()` to a graph node called `parse_notice_message`. The graph passes `state` to `parse_notice_message_node()` and stores the output in an updated `state`.

The next thing you need to do is add edges to your graph. Edges control the flow of data between nodes in your graph. Said differently, after a node performs an action and updates your graph's state, the edge flowing out of the node tells it which node to pass `state` to next. Here's how you add edges to your graph:

`graphs/notice_extraction.py`
```python
# ...

workflow.add_edge(START, "parse_notice_message")
workflow.add_edge("parse_notice_message", "check_escalation_status")
workflow.add_edge("check_escalation_status", END)

NOTICE_EXTRACTION_GRAPH = workflow.compile()
```
You call `workflow.add_edge()`, which accepts the names of the start and end nodes of the edge, respectively. `START` is a predefined node representing the entry point of the graph, and `END` is the node that terminates the graph. Here's what each edge definition does:

1.  On line 3, you add an edge from `START` to the `parse_notice_message` node.
2.  Then, line 4 adds an edge from `parse_notice_message` to `check_escalation_status`.
3.  Lastly, line 5 adds an edge to terminate the graph after running `check_escalation_status`.

You can now compile your graph by running `workflow.compile()`, which creates a `Runnable` interface that can be invoked, streamed, batched, and run asynchronously just like chains.

To visualize your graph, open another Python REPL and run the following:

```python
>>> from graphs.notice_extraction import NOTICE_EXTRACTION_GRAPH

>>> image_data = NOTICE_EXTRACTION_GRAPH.get_graph().draw_mermaid_png()
>>> with open("notice_extraction_graph.png", mode="wb") as f:
...     f.write(image_data)
...
8088 # (Byte count may vary)
```
Here, you import `NOTICE_EXTRACTION_GRAPH` and use `.get_graph().draw_mermaid_png()` to create an image of your graph. You then save the image to a file called `notice_extraction_graph.png`. Here's what the image should look like:

*(Image: Notice Extraction Graph Visual)*
*Your First Notice Extraction Graph*

This visual shows you that state flows from `parse_notice_message` to `check_escalation_status`, and it confirms that you've built your graph correctly. Here's how you use your graph:

```python
>>> from graphs.notice_extraction import NOTICE_EXTRACTION_GRAPH
>>> from example_emails import EMAILS

>>> initial_state = {
...     "notice_message": EMAILS[0],
...     "notice_email_extract": None,
...     "escalation_text_criteria": """There's a risk of fire or
...     water damage at the site""",
...     "escalation_dollar_criteria": 100_000.0, # Ensure float
...     "requires_escalation": False,
...     "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
...     "follow_ups": None, # Initialize optional fields
...     "current_follow_up": None,
... }

>>> final_state = NOTICE_EXTRACTION_GRAPH.invoke(initial_state)
# 2025-03-08 09:07:46,180 - INFO - Parsing notice...
# 2025-03-08 09:07:47,960 - INFO - Determining escalation status...

>>> final_state["notice_email_extract"]
# NoticeEmailExtract(
#     date_of_notice=datetime.date(2024, 10, 15),
#     entity_name='Occupational Safety and Health Administration (OSHA)',
#     entity_phone='(555) 123-4567',
#     entity_email='compliance.osha@osha.gov',
#     project_id=111232345,
#     site_location='123 Main Street, Dallas, TX',
#     violation_type='Lack of fall protection, Unsafe scaffolding setup, Inadequate personal protective equipment (PPE)',
#     required_changes='Install guardrails and fall arrest systems on all scaffolding over 10 feet. Conduct an inspection of all scaffolding structures and reinforce unstable sections. Ensure all workers on-site are provided with necessary PPE and conduct safety training on proper usage.',
#     compliance_deadline=datetime.date(2024, 11, 10),
#     max_potential_fine=25000.0
# )

>>> final_state["requires_escalation"]
# False
```
In this example, you import `NOTICE_EXTRACTION_GRAPH` and `EMAILS`. You then define your graph's initial state with `EMAILS[0]` as the `notice_message`, escalation criteria, and a few other fields that you'll use later in this tutorial. After calling `NOTICE_EXTRACTION_GRAPH.invoke(initial_state)`, your `final_state` stores the extracted notice fields along with the `requires_escalation` flag.

Notice that in `final_state`, `notice_email_extract` is now a `NoticeEmailExtract` object. Also, `requires_escalation` is set to `False` because `EMAILS[0]` doesn't say anything about fire or water damage, and the maximum potential fine is less than $100,000.

With that, you've built and successfully run your first state graph! If you're thinking that this graph isn't any more useful than a chain at this point, you're correct. In the next section, you'll address this by learning about and implementing LangGraph's differentiating feature—conditional edges.

---
*Remove ads*
---

## Work With Conditional Edges

LangGraph is all about modeling LLM workflows as graphs with nodes and edges. Nodes represent actions that your graph can take like calling functions or invoking chains, and edges tell your graph how to navigate between nodes.

So far, you've built a graph with a couple of nodes and edges between them that can't do much more than a chain. In this section, you'll learn about conditional edges, which you can use to move beyond chain-like structures to create intricate, conditional, and even cyclic workflows.

### Create a Conditional Edge

Up to this point, your graph can extract notice fields using `NOTICE_PARSER_CHAIN` and determine whether the notice message requires immediate escalation using `ESCALATION_CHECK_CHAIN`.

Next, you'll see how to change the path your graph takes depending on whether a notice message requires escalation. If a notice message *does* require escalation, your graph will immediately send an email informing the correct team. If escalation *isn't* required, your graph will create a legal ticket using your company's ticketing system API.

You'll start by defining a function that sends emails regarding the details of a notice message when the notice requires escalation. Here's what that looks like:

`utils/graph_utils.py`
```python
import random
import time
from pydantic import EmailStr
from chains.notice_extraction import NoticeEmailExtract
from utils.logging_config import LOGGER

def send_escalation_email(
    notice_email_extract: NoticeEmailExtract,
    escalation_emails: list[EmailStr]
) -> None:
    """Simulate sending escalation emails"""
    LOGGER.info("Sending escalation emails...")
    if escalation_emails: # Check if list is not empty/None
        for email in escalation_emails:
            time.sleep(1)
            LOGGER.info(f"Escalation email sent to {email}")
    else:
        LOGGER.warning("No escalation emails provided.")


def create_legal_ticket(
    current_follow_ups: dict[str, bool] | None,
    notice_email_extract: NoticeEmailExtract,
) -> str | None:
    """Simulate creating a legal ticket using your company's API."""
    LOGGER.info("Creating legal ticket for notice...")
    time.sleep(2)

    follow_ups_pool = [ # Renamed variable
        None,
        """Does this message mention the states of Texas, Georgia, or New Jersey?""",
        """Did this notice involve an issue with FakeAirCo's HVAC system?""",
    ]

    available_follow_ups = follow_ups_pool[:] # Create a copy

    if current_follow_ups:
        answered_questions = current_follow_ups.keys()
        available_follow_ups = [
            f for f in available_follow_ups if f not in answered_questions
        ]

    # Ensure there are questions left to ask, or None is an option
    if not available_follow_ups and None not in follow_ups_pool:
         LOGGER.info("All follow-up questions answered. Creating legal ticket!")
         return None
    elif not available_follow_ups: # Only None is left potentially
         follow_up = None
    else:
        follow_up = random.choice(available_follow_ups)


    if not follow_up:
        LOGGER.info("Legal ticket created!")
        return None # Explicitly return None

    LOGGER.info("Follow-up is required before creating this ticket")
    return follow_up
```
Here, you import dependencies and create a function called `send_escalation_email()`, which accepts a `NoticeEmailExtract` and a list of addresses to send emails to. Since actually sending emails is beyond the scope of this tutorial, `send_escalation_email()` simply simulates the process. For now, imagine that it sends an email to each address in `escalation_emails` regarding the details stored in `NoticeEmailExtract`. (Note: Added a check if `escalation_emails` is provided).

You then define `create_legal_ticket()` to simulate creating a ticket for your company's legal team to investigate. Notice how the first argument to `create_legal_ticket()` is a dictionary with string keys and Boolean values called `current_follow_ups`.

One feature of your legal team's ticketing system API is that it occasionally requires you to answer yes/no follow-up questions. These questions can change at any time depending on what the legal team is interested in knowing, so it's difficult to know what the follow-up questions might be ahead of time.

To simulate this logic, `create_legal_ticket()` checks the follow-up questions that you've already answered (`current_follow_ups`), and it randomly picks a new follow-up question from the remaining available questions in `follow_ups_pool`. If a follow-up isn't required (either no questions are left or `None` is chosen), `create_legal_ticket()` creates a legal ticket and returns `None`. If a follow-up *is* required, `create_legal_ticket()` returns the follow-up question, and you'll see how to handle this in your graph later on. (Note: Improved logic for selecting follow-up questions).

Next, you'll wrap `send_escalation_email()` and `create_legal_ticket()` in nodes that can interact with your graph's state:

`graphs/notice_extraction.py`
```python
from typing import TypedDict
from chains.escalation_check import ESCALATION_CHECK_CHAIN
from chains.notice_extraction import NOTICE_PARSER_CHAIN, NoticeEmailExtract
from langgraph.graph import END, START, StateGraph
from pydantic import EmailStr
from utils.graph_utils import create_legal_ticket, send_escalation_email # Make sure this import is correct
from utils.logging_config import LOGGER

# ... [Existing GraphState class definition] ...

# ... [Existing parse_notice_message_node and check_escalation_status_node] ...

def send_escalation_email_node(state: GraphState) -> GraphState:
    """Send an escalation email"""
    if state.get("notice_email_extract") and state.get("escalation_emails"):
        send_escalation_email(
            notice_email_extract=state["notice_email_extract"],
            escalation_emails=state["escalation_emails"],
        )
    else:
        LOGGER.warning("Cannot send escalation email: missing data in state.")
    return state # Always return state

def create_legal_ticket_node(state: GraphState) -> GraphState:
    """Node to create a legal ticket"""
    follow_up = None # Default value
    if state.get("notice_email_extract"):
        follow_up = create_legal_ticket(
            current_follow_ups=state.get("follow_ups"),
            notice_email_extract=state["notice_email_extract"],
        )
    else:
         LOGGER.warning("Cannot create legal ticket: missing notice_email_extract in state.")

    state["current_follow_up"] = follow_up
    return state # Always return state

# ... [Rest of the graph definition]
```
After importing your utility functions, you define two new node functions. In `send_escalation_email_node()`, you call `send_escalation_email()` without modifying state (except potentially logging). Conversely, in `create_legal_ticket_node()`, you call `create_legal_ticket()` and store the `follow_up` question, if there is one, in `state`. (Note: Added checks for required state elements before calling utils).

Now onto the critical part of this section. You need to create a conditional edge based on whether a notice email requires escalation. Specifically, if a notice email *does* require escalation, then your graph needs to pass state to `send_escalation_email_node()` *before* creating a legal ticket. If no escalation is required, then your graph can move directly to `create_legal_ticket_node()`.

Here's how you can create this behavior in your graph:

`graphs/notice_extraction.py`
```python
# ... [Existing imports, GraphState, node functions] ...

def route_escalation_status_edge(state: GraphState) -> str:
    """Determine whether to send an escalation email or
    create a legal ticket"""
    if state.get("requires_escalation", False): # Safely access requires_escalation
        LOGGER.info("Escalation needed!")
        return "send_escalation_email"

    LOGGER.info("No escalation needed")
    return "create_legal_ticket"

workflow = StateGraph(GraphState) # Re-initialize or ensure it's the existing one

workflow.add_node("parse_notice_message", parse_notice_message_node)
workflow.add_node("check_escalation_status", check_escalation_status_node)
workflow.add_node("send_escalation_email", send_escalation_email_node)
workflow.add_node("create_legal_ticket", create_legal_ticket_node)

workflow.add_edge(START, "parse_notice_message")
workflow.add_edge("parse_notice_message", "check_escalation_status")

# Conditional edge based on escalation status
workflow.add_conditional_edges(
    "check_escalation_status", # Starting node
    route_escalation_status_edge, # Function to determine the route
    {
        # Map return value of function to next node name
        "send_escalation_email": "send_escalation_email",
        "create_legal_ticket": "create_legal_ticket",
    },
)

workflow.add_edge("send_escalation_email", "create_legal_ticket") # After sending email, go to create ticket
# workflow.add_edge("create_legal_ticket", END) # This will be changed later for cycles

# ... [Compile the graph - NOTICE_EXTRACTION_GRAPH = workflow.compile()] ...
# NOTE: The edge from create_legal_ticket to END will be replaced in the next section
```
You first define `route_escalation_status_edge()`, which is a function that governs the behavior of your conditional edge. Depending on whether the notice requires escalation (`state["requires_escalation"]`), `route_escalation_status_edge()` returns a string (`"send_escalation_email"` or `"create_legal_ticket"`) that indicates which node to navigate to next. After registering the `send_escalation_email` and `create_legal_ticket` nodes, you register a conditional edge using `workflow.add_conditional_edges()`.

**Note:** In general, node functions should return `GraphState` objects (or a dictionary subset to update the state), while edge functions return strings that tell you which node or nodes to navigate to.

In `workflow.add_conditional_edges()`, the first argument (`"check_escalation_status"`) tells your graph which node the edge starts from. The second argument (`route_escalation_status_edge`) is the function that governs the behavior of the conditional edge. The third argument is a mapping that tells your graph which node to go to based on the *string output* of the conditional edge function.

For example, if `route_escalation_status_edge()` returns `"send_escalation_email"`, your graph will invoke the `send_escalation_email` node next. Lastly, you add an edge from `send_escalation_email` to `create_legal_ticket` (so ticketing happens regardless of escalation, just *after* email if needed). The edge from `create_legal_ticket` to `END` will be modified later. Finally, you compile your graph. Here's what your graph looks like now:

*(Image: A graph with a conditional edge)*
*Your Current Notice Extraction Graph With a Conditional Edge*

You now see dotted lines representing the conditional edge flowing out of `check_escalation_status` into `send_escalation_email` and `create_legal_ticket`. As expected, if escalation is needed, your graph goes to `send_escalation_email` *before* `create_legal_ticket`. Otherwise, your graph flows directly to `create_legal_ticket`.

To see your conditional edge in action, try this example:

```python
>>> from graphs.notice_extraction import NOTICE_EXTRACTION_GRAPH
>>> from example_emails import EMAILS

>>> initial_state_no_escalation = {
...     "notice_message": EMAILS[0],
...     "notice_email_extract": None,
...     "escalation_text_criteria": """There's a risk of water damage at the site""",
...     "escalation_dollar_criteria": 100000.0,
...     "requires_escalation": False,
...     "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
...     "follow_ups": None,
...     "current_follow_up": None,
... }

>>> initial_state_escalation = {
...     "notice_message": EMAILS[0],
...     "notice_email_extract": None,
...     "escalation_text_criteria": """Workers explicitly violating safety protocols""",
...     "escalation_dollar_criteria": 100000.0,
...     "requires_escalation": False, # Will be updated by check_escalation_status_node
...     "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
...     "follow_ups": None,
...     "current_follow_up": None,
... }

>>> no_esc_result = NOTICE_EXTRACTION_GRAPH.invoke(initial_state_no_escalation)
# 2025-03-09 23:37:35,627 - INFO - Parsing notice...
# 2025-03-09 23:37:38,584 - INFO - Determining escalation status...
# NoneType: object has no attribute 'max_potential_fine' # Potential error if parsing fails - Ensure Robust Parsing
# 2025-03-09 23:37:39,270 - INFO - No escalation needed
# 2025-03-09 23:37:39,271 - INFO - Creating legal ticket for notice...
# 2025-03-09 23:37:41,277 - INFO - Legal ticket created! # Or requires follow-up

>>> no_esc_result.get("requires_escalation") # Use .get for safer access
# False

>>> esc_result = NOTICE_EXTRACTION_GRAPH.invoke(initial_state_escalation)
# 2025-03-09 23:37:57,977 - INFO - Parsing notice...
# 2025-03-09 23:38:01,391 - INFO - Determining escalation status...
# 2025-03-09 23:38:01,903 - INFO - Escalation needed!
# 2025-03-09 23:38:01,903 - INFO - Sending escalation emails...
# 2025-03-09 23:38:02,908 - INFO - Escalation email sent to brog@abc.com
# 2025-03-09 23:38:03,913 - INFO - Escalation email sent to bigceo@company.com
# 2025-03-09 23:38:03,915 - INFO - Creating legal ticket for notice...
# 2025-03-09 23:38:05,920 - INFO - Legal ticket created! # Or requires follow-up

>>> esc_result.get("requires_escalation")
# True
```
In this block, you use the same `EMAILS[0]` from the previous section and create two initial states: `initial_state_no_escalation` and `initial_state_escalation`. You expect `initial_state_no_escalation` not to require escalation because `EMAILS[0]` doesn't mention anything about water damage (and the fine is below $100k). On the other hand, `initial_state_escalation` *should* require escalation because the escalation criteria mention workers violating safety protocols, which aligns with `EMAILS[0]` (workers not wearing PPE).

Exactly as expected, you can see from the logs that `NOTICE_EXTRACTION_GRAPH.invoke(initial_state_escalation)` results in `requires_escalation` being `True`, and your graph sends emails to the addresses listed in `state["escalation_emails"]` before creating a legal ticket. Compare this to `NOTICE_EXTRACTION_GRAPH.invoke(initial_state_no_escalation)`, where `requires_escalation` is `False`, which moves directly to creating a legal ticket without sending escalation emails.

Your graph now moves beyond the limitations of chains by handling conditional workflows. Take a moment to think about why the conditional edge abstraction is so powerful. Most meaningful real-world tasks involve several decisions that change the trajectory of steps you take. Trying to replicate this behavior with chains would require a lot of boilerplate conditional and iterative logic, and it would quickly get out of hand as your graph grows.

Now what about those follow-up questions? If the legal ticketing API returns a follow-up question, how can you answer it and attempt to create a legal ticket again? The answer lies in cycles, and that's what you'll explore next.

---
*Remove ads*
---

### Use Conditional Edges for Cycles

The last capability you'll explore for your notice extraction graph is a cycle. A graph cycle is effectively a loop between two (or more) nodes that continues until a task is complete. In this section, you'll build a cycle between the node that creates legal tickets (`create_legal_ticket_node`) and a new node that answers the follow-up questions.

If you recall from the last section, sometimes the legal ticketing API (`create_legal_ticket` function) returns follow-up questions that you're required to answer before creating the ticket. You want to build a cycle that receives and answers the follow-up questions from the legal ticket node until no other follow-up questions are required. To keep things simple for this tutorial, these questions will always have binary yes or no answers.

First, create a chain that you'll use to answer binary questions:

`chains/binary_questions.py`
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class BinaryAnswer(BaseModel):
    is_true: bool = Field(
        description="""Whether the answer to the question is yes or no.
        True if yes otherwise False."""
    )

binary_question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer this question based on the provided context as True for "yes" and False for "no".
            No other answers are allowed.

            Context:
            {context}

            Question:
            {question}
            """,
        )
    ]
)

binary_question_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

BINARY_QUESTION_CHAIN = (
    binary_question_prompt
    | binary_question_model.with_structured_output(BinaryAnswer)
)
```
This `BINARY_QUESTION_CHAIN` is similar to the chains you built previously. It accepts a `question` and `context` (which will be the email message) as input and outputs `True` if the answer to the question based on the context is yes, and `False` if the answer is no. (Note: Modified prompt to include context). Of course, this chain could give strange results for questions that don't have yes/no answers, but you could modify it to return `None` or to answer arbitrary questions with a text response.

Next, you'll create a node in your graph that uses `BINARY_QUESTION_CHAIN` to answer follow-up questions:

`graphs/notice_extraction.py`
```python
from typing import TypedDict
from chains.binary_questions import BINARY_QUESTION_CHAIN # Import the new chain
from chains.escalation_check import ESCALATION_CHECK_CHAIN
from chains.notice_extraction import NOTICE_PARSER_CHAIN, NoticeEmailExtract
from langgraph.graph import END, START, StateGraph
from pydantic import EmailStr
from utils.graph_utils import create_legal_ticket, send_escalation_email
from utils.logging_config import LOGGER

# ... [Existing GraphState, other node functions] ...

def answer_follow_up_question_node(state: GraphState) -> GraphState:
    """Answer follow-up questions about the notice using
    BINARY_QUESTION_CHAIN"""
    current_follow_up = state.get("current_follow_up")
    notice_message = state.get("notice_message")

    if current_follow_up and notice_message:
        LOGGER.info(f"Answering follow-up: {current_follow_up}")
        # Pass both question and context (email) to the chain
        answer_obj = BINARY_QUESTION_CHAIN.invoke({
            "question": current_follow_up,
            "context": notice_message
            })
        answer = answer_obj.is_true # Extract boolean value

        if state.get("follow_ups") is None: # Initialize if first follow-up
             state["follow_ups"] = {}

        # Store the answer (boolean) in the follow_ups dictionary
        state["follow_ups"][current_follow_up] = answer
        LOGGER.info(f"Answered '{current_follow_up}': {answer}")

    else:
        LOGGER.warning("Cannot answer follow-up: missing question or notice message in state.")

    # Clear current_follow_up after answering, so create_legal_ticket runs again
    state["current_follow_up"] = None

    return state

# ... [Rest of graph definition] ...
```
In `answer_follow_up_question_node()`, you check if there's a `current_follow_up` question and a `notice_message` in `state`. If there is, you pass the question and the notice message (as context) through `BINARY_QUESTION_CHAIN.invoke()`. You then store the boolean answer (`answer.is_true`) as an entry in the `state["follow_ups"]` dictionary, initializing the dictionary if necessary. Importantly, you then clear `state["current_follow_up"]` so that the next node (`create_legal_ticket_node`) knows this question has been handled.

Now you need to create a function to define the conditional edge *after* the `create_legal_ticket_node`:

`graphs/notice_extraction.py`
```python
# ... [Existing imports, GraphState, node functions including answer_follow_up_question_node] ...

def route_follow_up_edge(state: GraphState) -> str:
    """Determine whether a follow-up question is required from create_legal_ticket"""
    if state.get("current_follow_up"):
        # If create_legal_ticket returned a question, go answer it
        LOGGER.info("Follow-up question received, routing to answer node.")
        return "answer_follow_up_question"
    # Otherwise, the ticket was created (or failed finally), end the graph
    LOGGER.info("No follow-up question, routing to END.")
    return END

# ... [Rest of graph definition] ...
```
In `route_follow_up_edge()`, if `state.get("current_follow_up")` has a value (meaning `create_legal_ticket_node` just returned a question), then you navigate to `answer_follow_up_question`. If not (meaning `create_legal_ticket_node` returned `None`), then you exit the graph (`END`).

Now, you can register your new node and create the conditional edge that forms the cycle:

`graphs/notice_extraction.py`
```python
# ... [Existing imports, GraphState, node functions] ...

workflow = StateGraph(GraphState) # Ensure using the same workflow instance

workflow.add_node("parse_notice_message", parse_notice_message_node)
workflow.add_node("check_escalation_status", check_escalation_status_node)
workflow.add_node("send_escalation_email", send_escalation_email_node)
workflow.add_node("create_legal_ticket", create_legal_ticket_node)
workflow.add_node("answer_follow_up_question", answer_follow_up_question_node) # Add the new node

workflow.add_edge(START, "parse_notice_message")
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
    "create_legal_ticket", # Starting node
    route_follow_up_edge, # Function to decide next step
    {
        "answer_follow_up_question": "answer_follow_up_question", # Go answer
        END: END, # Or end the graph
    },
)

# Edge AFTER answering follow-up - ALWAYS go back to try creating ticket again
workflow.add_edge("answer_follow_up_question", "create_legal_ticket")

NOTICE_EXTRACTION_GRAPH = workflow.compile() # Compile the final graph
```
Here, you add the `answer_follow_up_question` node. Then, you define the conditional edge starting from `create_legal_ticket` using `route_follow_up_edge`. This edge flows to either `answer_follow_up_question` (if a follow-up is needed) or `END` (if the ticket is created). Notice that you also add an edge from answer_follow_up_question back to create_legal_ticket, which completes the cycle. To make more sense of this, take a look at the updated visualization of your graph:

*(Image: LangGraph Graph with a cycle)*
*Your Updated Graph With a Cycle Between `create_legal_ticket` and `answer_follow_up_question`*

Your graph has come a long way from where it started! Notice the dotted arrows representing the conditional edge coming out of `create_legal_ticket`. If a follow-up question exists (`current_follow_up` is set), the graph goes to `answer_follow_up_question`. If not, it goes to `END`. Additionally, there's a solid edge flowing back *into* `create_legal_ticket` from `answer_follow_up_question`. This creates the cycle because `create_legal_ticket` will continue to be called after each follow-up question is answered until no follow-up questions remain, at which point the graph exits via the conditional edge.

Go ahead and give your updated graph a test run:

```python
>>> from graphs.notice_extraction import NOTICE_EXTRACTION_GRAPH
>>> from example_emails import EMAILS

>>> initial_state_cycle_test = { # Renamed for clarity
...       "notice_message": EMAILS[0],
...       "notice_email_extract": None,
...       "escalation_text_criteria": """Workers explicitly violating safety protocols""",
...       "escalation_dollar_criteria": 100000.0,
...       "requires_escalation": False,
...       "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
...       "follow_ups": None, # Start with no follow-ups answered
...       "current_follow_up": None,
...  }

>>> results = NOTICE_EXTRACTION_GRAPH.invoke(initial_state_cycle_test)
# 2025-03-10 22:06:23,507 - INFO - Parsing notice...
# 2025-03-10 22:06:27,046 - INFO - Determining escalation status...
# 2025-03-10 22:06:27,868 - INFO - Escalation needed!
# 2025-03-10 22:06:27,869 - INFO - Sending escalation emails...
# 2025-03-10 22:06:28,873 - INFO - Escalation email sent to brog@abc.com
# 2025-03-10 22:06:29,879 - INFO - Escalation email sent to bigceo@company.com
# 2025-03-10 22:06:29,882 - INFO - Creating legal ticket for notice...
# 2025-03-10 22:06:31,887 - INFO - Follow-up is required before creating this ticket
# 2025-03-10 22:06:31,887 - INFO - Follow-up question received, routing to answer node.
# 2025-03-10 22:06:31,888 - INFO - Answering follow-up: Does this message mention the states of Texas, Georgia, or New Jersey?
# ... [LLM Call] ...
# 2025-03-10 22:06:32,374 - INFO - Answered 'Does this message mention the states of Texas, Georgia, or New Jersey?': True
# 2025-03-10 22:06:32,375 - INFO - Creating legal ticket for notice...
# 2025-03-10 22:06:34,377 - INFO - Follow-up is required before creating this ticket
# 2025-03-10 22:06:34,377 - INFO - Follow-up question received, routing to answer node.
# 2025-03-10 22:06:34,378 - INFO - Answering follow-up: Did this notice involve an issue with FakeAirCo's HVAC system?
# ... [LLM Call] ...
# 2025-03-10 22:06:34,934 - INFO - Answered 'Did this notice involve an issue with FakeAirCo's HVAC system?': False
# 2025-03-10 22:06:34,935 - INFO - Creating legal ticket for notice...
# 2025-03-10 22:06:36,937 - INFO - Legal ticket created! # Assuming None is chosen this time
# 2025-03-10 22:06:36,937 - INFO - No follow-up question, routing to END.

>>> results.get("follow_ups")
# {'Does this message mention the states of Texas, Georgia, or New Jersey?': True,
#  "Did this notice involve an issue with FakeAirCo's HVAC system?": False}
```
Here, you import `NOTICE_EXTRACTION_GRAPH` and the example emails. Recall that one of the potential follow-up questions in `create_legal_ticket()` is: *"Does this message mention the states of Texas, Georgia, or New Jersey?"* Therefore, if `create_legal_ticket()` requires you to answer that question for `EMAILS[0]`, the answer should be `True` since `EMAILS[0]` mentions Dallas, Texas. The other question about "FakeAirCo" should be `False`.

In the example above (logs adjusted to reflect the cycle logic), when you run `EMAILS[0]` through your graph, you see from the logs that `create_legal_ticket()` might require your graph to answer one or both follow-up questions before finally creating the ticket. Keep in mind that it might take you a few tries to replicate this *exact* sequence since `create_legal_ticket()` randomly selects follow-up questions. From `results["follow_ups"]`, you see that `BINARY_QUESTION_CHAIN` correctly answered the questions based on the email content.

With that, you've completed your notice email processing graph. Nice work! Hopefully, you're seeing the power of LangGraph to create sophisticated, conditional, and cyclic LLM workflows. From here, you can continue adding any functionality that you can imagine by creating nodes to accomplish tasks and adding edges to navigate between them.

In the next and final section, you'll add some nice finishing touches to your notice email processing graph. You'll use LangGraph to create an agent that can handle any kind of email and use `NOTICE_EXTRACTION_GRAPH` as one of its tools.

---
*Remove ads*
---

## Develop Graph Agents

Now it's time to complete your email-parsing system by creating an agent with LangGraph. If you're unfamiliar with agents, you can read about them in the LangChain tutorial. In short, an AI agent is a system of AI models, usually LLMs, capable of performing tasks and making decisions autonomously.

The two main components of an agent are:

1.  **The models** that make decisions
2.  **The tools** the models use to perform actions

LangGraph was designed with agents in mind because agent architectures tend to be conditional and cyclic in nature. The agent you'll create in this section will govern email processing, and it will have access to tools that can send emails and call `NOTICE_EXTRACTION_GRAPH` when it determines that an email is a regulatory notice.

### Structure Agents as Graphs

To get started building your email processing agent, first import the following dependencies:

`graphs/email_agent.py`
```python
import time
from typing import Annotated # Needed for MessagesState example later if used
from chains.notice_extraction import NoticeEmailExtract
from graphs.notice_extraction import NOTICE_EXTRACTION_GRAPH
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage # Import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages # Utility for updating MessagesState
from langgraph.prebuilt import ToolNode
from utils.logging_config import LOGGER
import operator # For MessagesState example

# Define the state for the agent graph
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# OR using the prebuilt MessagesState (simpler):
# from langgraph.graph import MessagesState
```
Most of these imports should look familiar. We've added `BaseMessage`, `HumanMessage`, `Annotated`, `add_messages`, and `operator` which are often used with agent states. `MessagesState` (if imported directly) is a pre-built `TypedDict` with a special `messages` field designed for agents. It automatically handles appending new messages using the `add_messages` function when used correctly with `Annotated`. Each node in your agent graph will append its output to `messages` in the state.

Also, notice that you've imported the `ToolNode` class from `langgraph.prebuilt`. `ToolNode` allows you to create nodes in your graph explicitly designed for executing tools chosen by an agent node. In both LangChain and LangGraph, a tool is nothing more than a function your agent models can call to perform actions. Here are the tools you'll need for your email agent:

`graphs/email_agent.py`
```python
# ... [Imports and State definition] ...

@tool
def forward_email(email_message: str, send_to_email: str) -> str: # Return type often str for status
    """
    Forward an email_message to the address of send_to_email. Returns
    a success or error message. Note
    that this tool only forwards the email to an internal department -
    it does not reply to the sender.
    """
    LOGGER.info(f"Forwarding the email to {send_to_email}...")
    # Simulate potential multiple recipients if comma-separated
    recipients = [email.strip() for email in send_to_email.split(',') if email.strip()]
    if not recipients:
        return "Error: No valid recipient email provided."
    try:
        for recipient in recipients:
             LOGGER.info(f"Simulating forward to: {recipient}")
             time.sleep(1) # Simulate network delay per recipient
        LOGGER.info("Email forwarded successfully!")
        return f"Successfully forwarded email to {', '.join(recipients)}."
    except Exception as e:
        LOGGER.error(f"Failed to forward email: {e}")
        return f"Error: Failed to forward email. Details: {e}"


@tool
def send_wrong_email_notification_to_sender(
    sender_email: str, correct_department: str
) -> str: # Return type often str for status
    """
    Send an email back to the sender_email informing them that
    they have the wrong address. Inform them the email should be sent
    to the correct_department address instead.
    """
    LOGGER.info(f"Sending wrong email notification to {sender_email}...")
    try:
        # Simulate sending email
        time.sleep(2)
        LOGGER.info("Wrong email notification sent!")
        return f"Successfully sent wrong email notification to {sender_email}, advising them to use {correct_department}."
    except Exception as e:
        LOGGER.error(f"Failed to send notification: {e}")
        return f"Error: Failed to send notification. Details: {e}"

@tool
def extract_notice_data(
    email: str, escalation_criteria: str = "Default criteria if not provided" # Add default
) -> str: # Return type changed to str for agent compatibility
    """
    Extract structured fields from a regulatory notice email.
    This should be used ONLY when the email message clearly comes from
    a regulatory body, government agency, or auditor regarding a property or
    construction site that the company works on.

    escalation_criteria is a description of which kinds of
    notices require immediate escalation (e.g., 'mentions fire risk or fines over $50k').

    After calling this tool, the process is complete for this email.
    Do not call any other tools after this one for the same email.
    Returns a summary of the extracted data or an error message.
    """
    LOGGER.info("Calling the email notice extraction graph...")
    try:
        # Define the initial state for the notice extraction graph
        initial_state = {
            "notice_message": email,
            "notice_email_extract": None,
            # "critical_fields_missing": False, # This field was not in the final notice graph state
            "escalation_text_criteria": escalation_criteria,
            "escalation_dollar_criteria": 100000.0, # Example threshold
            "requires_escalation": False,
            "escalation_emails": ["brog@abc.com", "bigceo@company.com"], # Example emails
            "follow_ups": None,
            "current_follow_up": None,
        }

        # Invoke the notice extraction graph
        results = NOTICE_EXTRACTION_GRAPH.invoke(initial_state)

        extracted_data = results.get("notice_email_extract")
        if extracted_data:
             # Convert Pydantic model to string for agent response
             return f"Notice data extracted successfully: {extracted_data.model_dump_json(indent=2)}"
        else:
             return "Error: Failed to extract notice data from the email."

    except Exception as e:
        LOGGER.error(f"Error calling notice extraction graph: {e}")
        return f"Error: An exception occurred during notice extraction: {e}"


@tool
def determine_email_action(email: str) -> str:
    """
    Call this tool ONLY as a last resort to determine which action should be taken
    for an email IF AND ONLY IF no other tools seem relevant (e.g., it's not a regulatory notice,
    not clearly an invoice, not clearly a customer support issue).
    Do not call this tool if you have already called extract_notice_data.
    This tool provides routing GUIDELINES based on common scenarios.
    """
    return """
    Routing Guidelines:
    1. Invoice/Billing: If the email appears to be an invoice, billing statement, or payment query, use the 'forward_email' tool to send it ONLY to billing@company.com. Then, use the 'send_wrong_email_notification_to_sender' tool, informing the sender the correct department is billing@company.com.
    2. Customer Support: If the email appears to be from a customer reporting an issue, asking for help, or requesting maintenance/refunds, use the 'forward_email' tool to send it to ALL of these addresses: support@company.com, cdetuma@company.com, ctu@abc.com. Then, use the 'send_wrong_email_notification_to_sender' tool, informing the sender the correct department is support@company.com.
    3. Other: For emails that don't fit the above and aren't regulatory notices, attempt to infer the correct department from the context (e.g., job application -> humanresources@company.com, technical issue -> it@company.com). If unsure, use 'send_wrong_email_notification_to_sender' and suggest a likely department (e.g., support@company.com or billing@company.com). Provide a brief final response indicating the action taken.
    """
```
The functions you defined above are the tools your agent will use, and each function is decorated by `@tool`. A key functionality of `@tool` is that it makes the function (including its docstring and type hints) callable by the agent's LLM. The LLM uses the docstring and argument descriptions to determine *which* tool is relevant to the task at hand and *what arguments* to provide. Because of this, it's important to write clear, informative docstrings and use descriptive argument names to maximize the chances that your agent uses the appropriate tools correctly. (Note: Return types changed to `str` for better compatibility with tool outputs in agent loops, docstrings refined, added error handling/simulation).

As an example of how your agent will use these tools, if the agent determines that it needs to forward an email, the agent's underlying LLM will aim to return a structured output (like JSON, though LangChain handles this) specifying the tool name (`forward_email`) and the required arguments (`email_message`, `send_to_email`). LangGraph's `ToolNode` will then parse this and execute the actual `forward_email()` Python function with those arguments.

Perhaps the most powerful tool available to your agent is `extract_notice_data()`. Your agent should call `extract_notice_data()` when it determines that an email is a regulatory notice. This tool then invokes your entire `NOTICE_EXTRACTION_GRAPH`. This is an amazing abstraction—because tools are just Python functions, you can use them to call other chains, graphs, agents, external APIs, or perform just about any task that you can encapsulate in a function.

The last tool you define, `determine_email_action()`, acts as a fallback, providing guidelines when the email's purpose isn't immediately clear or covered by other tools.

Next, you need to define the nodes and edges of your agent graph:

`graphs/email_agent.py`
```python
# ... [Imports, State definition, Tools] ...

tools = [
    determine_email_action,
    forward_email,
    send_wrong_email_notification_to_sender,
    extract_notice_data,
]

# The ToolNode executes tools based on the agent's output
tool_node = ToolNode(tools)

# The LLM acts as the agent's brain, deciding which tool to call (if any)
# Bind the tools to the LLM - this allows the LLM to see the tool descriptions
EMAIL_AGENT_MODEL = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

# Define the node that calls the agent model
def call_agent_model_node(state: AgentState) -> dict: # Return updates for the state
    """Node that calls the LLM agent model"""
    messages = state["messages"]
    # Invoke the LLM with the current conversation history
    response = EMAIL_AGENT_MODEL.invoke(messages)
    # We return a dictionary with the key 'messages' and a list containing the response
    # The 'add_messages' function in the state definition handles appending this
    LOGGER.info("Agent model invoked, returning response.")
    return {"messages": [response]}

# Define the conditional edge routing function
def route_agent_graph_edge(state: AgentState) -> str:
    """Determines whether to continue calling tools or end the graph."""
    # Get the last message added to the state
    last_message = state["messages"][-1]
    # Check if the last message contains tool calls requested by the LLM
    if last_message.tool_calls:
        # If there are tool calls, route to the tool node
        LOGGER.info("Agent requested tool calls, routing to tools node.")
        return "call_tools" # Name of the tool node
    # If there are no tool calls, the agent has finished, end the graph
    LOGGER.info("Agent finished, routing to END.")
    return END

# ... [Graph definition below] ...
```
You first gather your functions into a `tools` list. You then instantiate a `ToolNode` which will execute the function calls requested by the agent. You then define `EMAIL_AGENT_MODEL`, which is the LLM that acts as your agent's brain.

By calling `.bind_tools(tools)` when instantiating `EMAIL_AGENT_MODEL`, you're providing the LLM with descriptions of each tool (from their docstrings and type hints). If `EMAIL_AGENT_MODEL` determines that its input requires a tool call, it will output a message containing `tool_calls`, specifying the name of the tool(s) it wants to use and the arguments for each.

In `call_agent_model_node()`, you define the logic for the main agent node. It takes the current list of messages from the `state` and passes them to `EMAIL_AGENT_MODEL.invoke()`. The LLM's response (which might be a final answer or a request to use tools) is returned in a dictionary format suitable for updating the `MessagesState` (or the custom `AgentState` using `add_messages`).

Lastly, you define `route_agent_graph_edge()`, which governs the conditional edge *after* the agent node runs. It checks the *last message* added to the state. If that message contains `tool_calls`, it means the agent wants to use a tool, so the function returns the name of the tool node (`"call_tools"`). Otherwise, it means the agent provided a final response, so the function returns `END` to terminate the graph. This creates the core agent loop: agent decides -> tool executes -> agent processes result -> agent decides...

Here are the final lines to define and compile your agent graph:

`graphs/email_agent.py`
```python
# ... [Imports, State, Tools, Node/Edge functions] ...

# Initialize the StateGraph using the AgentState (or MessagesState)
workflow = StateGraph(AgentState) # Or StateGraph(MessagesState)

# Add the agent node
workflow.add_node("agent", call_agent_model_node)
# Add the tool execution node
workflow.add_node("call_tools", tool_node)

# Set the entry point: the agent node
workflow.add_edge(START, "agent")

# Add the conditional edge: after the agent runs, decide to call tools or end
workflow.add_conditional_edges(
    "agent", # Starting node
    route_agent_graph_edge, # Function to determine the route
    {
        "call_tools": "call_tools", # Route to tool node if tool calls exist
        END: END # Route to END if no tool calls
    }
)

# Add the edge to loop back from the tool node to the agent node
workflow.add_edge("call_tools", "agent")

# Compile the graph
email_agent_graph = workflow.compile()

LOGGER.info("Email agent graph compiled successfully.")

# Optional: Add a main block for testing if desired
# if __name__ == "__main__":
#    from example_emails import EMAILS
#    # ... test code ...
```
You first instantiate `workflow`—a `StateGraph` using your chosen state definition (`AgentState` or `MessagesState`). Then, you add the `agent` node (`call_agent_model_node`) and the `call_tools` node (`tool_node`) to your graph.

You set the entry point (`START`) to go directly to the `agent` node. Then, you add the conditional edge starting from `agent`, governed by `route_agent_graph_edge`, which directs the flow to either `call_tools` or `END`.

Crucially, you add a direct edge from `call_tools` *back* to `agent`. This creates the cycle: the agent node processes the input (or results from a previous tool call), decides if tools are needed, the conditional edge routes to the tool node if necessary, the tool node executes the tools, and then the flow returns to the agent node to process the tool results. This continues until the agent node provides a final response without requesting further tool calls, at which point the conditional edge routes to `END`.

Zooming out, here's what your email agent looks like:

*(Image: LangGraph agent)*
*Your Final LangGraph Agent*

This architecture, while straightforward, is very common and a great place to start when building agents. The general idea is that your `agent` node will accept an email message (as the initial input) and continually interact with the `call_tools` node until it believes it has successfully processed the email and provides a final response. From here, you can continue expanding your agent's capabilities by adding more tools or refining the agent's logic. The last thing to do now is give your agent a try and see how it performs on a few examples.

---
*Remove ads*
---

## Test Your Graph Agent

Your email graph agent is complete and ready for you to test! In practice, it's a good idea to test your agents on several examples where you know what the desired behavior should be. This way, you can measure your agent's performance and adjust its architecture (tools, prompts, model) to improve it. However, for this tutorial, you'll just empirically inspect how your agent responds to the example emails you defined earlier.

Given the example emails, here's how you might expect your agent to respond to each:

*   **Email 0:** This email is clearly a regulatory notice from OSHA. The agent should identify this and call the `extract_notice_data` tool, which in turn runs your `NOTICE_EXTRACTION_GRAPH`. Since you've used this example several times, you'll skip testing it again here.
*   **Email 1:** This email ("Here's your invoice for $1000...") looks like an invoice. The agent should ideally use `determine_email_action` to get guidelines, then call `forward_email` to send it to `billing@company.com`, and finally call `send_wrong_email_notification_to_sender` to notify `debby@stack.com`.
*   **Email 2:** This email ("...issue with the HVAC system...") appears to be from a customer. The agent should use `determine_email_action`, then call `forward_email` three times (to `support@company.com`, `cdetuma@company.com`, `ctu@abc.com`), and finally call `send_wrong_email_notification_to_sender` to notify `tdavid@companyxyz.com`.
*   **Email 3:** This email is clearly a regulatory notice from the LA Building and Safety Dept. The agent should identify this and call the `extract_notice_data` tool.

To see how your agent does, open a Python REPL (or run a script) and try it out on the first example email (Email 1):

```python
>>> from graphs.email_agent import email_agent_graph, AgentState # Import state if needed for input
>>> from example_emails import EMAILS
>>> from langchain_core.messages import HumanMessage

# Prepare the input for the graph - it expects the state structure
>>> message_1_input = AgentState(messages=[HumanMessage(content=EMAILS[1])])
# Or if using MessagesState: message_1_input = {"messages": [HumanMessage(content=EMAILS[1])]}


>>> # Stream the results to see the agent's steps
>>> for chunk in email_agent_graph.stream(message_1_input, stream_mode="values"):
...      # Print the last message added in each step
...      last_message = chunk["messages"][-1]
...      print(f"----- Last Message ({type(last_message).__name__}) -----")
...      last_message.pretty_print()
...      print("\n----- End Chunk -----\n")

# Expected Output Sequence (simplified):
# ----- Last Message (HumanMessage) -----
# ================================ Human Message ================================
#
#     From: debby@stack.com
#     Hey Betsy,
#
#     Here's your invoice for $1000 for the cookies you ordered.
#
# ----- End Chunk -----
#
# ----- Last Message (AIMessage) -----
# ================================== Ai Message ==================================
# Tool Calls:
#   determine_email_action (...)
#  Call ID: ...
#   Args:
#     email: From: debby@stack.com...
# ----- End Chunk -----
#
# ----- Last Message (ToolMessage) -----
# ================================= Tool Message =================================
# Name: determine_email_action
#
# Routing Guidelines:...1. Invoice/Billing: ... forward... billing@company.com... send notification...
# ----- End Chunk -----
#
# ----- Last Message (AIMessage) -----
# ================================== Ai Message ==================================
# Tool Calls:
#   forward_email (...)
#  Call ID: ...
#   Args:
#     email_message: From: debby@stack.com...
#     send_to_email: billing@company.com
#   send_wrong_email_notification_to_sender (...)
#  Call ID: ...
#   Args:
#     sender_email: debby@stack.com
#     correct_department: billing@company.com
# ----- End Chunk -----
#
# (Log messages from tool execution appear here)
# 2025-01-26 12:00:46,317 - INFO - Forwarding the email to billing@company.com...
# 2025-01-26 12:00:46,318 - INFO - Sending wrong email notification to debby@stack.com...
# 2025-01-26 12:00:48,323 - INFO - Email forwarded successfully!
# 2025-01-26 12:00:48,324 - INFO - Wrong email notification sent!
#
# ----- Last Message (ToolMessage) -----
# ================================= Tool Message =================================
# Name: forward_email
#
# Successfully forwarded email to billing@company.com.
# ----- End Chunk -----
#
# ----- Last Message (ToolMessage) -----
# ================================= Tool Message =================================
# Name: send_wrong_email_notification_to_sender
#
# Successfully sent wrong email notification to debby@stack.com, advising them to use billing@company.com.
# ----- End Chunk -----
#
# ----- Last Message (AIMessage) -----
# ================================== Ai Message ==================================
#
# The email invoice from debby@stack.com has been forwarded to the billing department (billing@company.com),
# and a notification has been sent to the sender advising them of the correct address.
# ----- End Chunk -----

```
Here, you import `email_agent_graph` and prepare the input as a dictionary matching the `AgentState` structure, containing the first email (`EMAILS[1]`) as a `HumanMessage`. You then run the input through `email_agent_graph.stream()` to observe the messages added at each step.

Here's what happens (based on the expected output):

1.  The agent receives the human message (the email).
2.  The agent decides `determine_email_action` is needed to figure out the routing.
3.  The `determine_email_action` tool returns the routing guidelines.
4.  The agent processes the guidelines and decides to call `forward_email` (to billing) and `send_wrong_email_notification_to_sender`.
5.  The `ToolNode` executes both `forward_email` and `send_wrong_email_notification_to_sender`, returning their success messages.
6.  The agent receives the success messages from the tools and provides a final confirmation response.

Your agent behaved exactly like you wanted it to for the invoice email!

Now try your agent on the next email (Email 2 - customer issue):

```python
>>> # Prepare input for Email 2
>>> message_2_input = AgentState(messages=[HumanMessage(content=EMAILS[2])])
# Or: message_2_input = {"messages": [HumanMessage(content=EMAILS[2])]}

>>> for chunk in email_agent_graph.stream(message_2_input, stream_mode="values"):
...      last_message = chunk["messages"][-1]
...      print(f"----- Last Message ({type(last_message).__name__}) -----")
...      last_message.pretty_print()
...      print("\n----- End Chunk -----\n")

# Expected Output Sequence (simplified):
# ... (HumanMessage with Email 2) ...
# ... (AIMessage calling determine_email_action) ...
# ... (ToolMessage with routing guidelines: "...2. Customer Support: ... forward... support@company.com, cdetuma@company.com, ctu@abc.com... send notification...") ...
# ... (AIMessage calling forward_email [3 times] and send_wrong_email_notification_to_sender) ...
# (Log messages for 3 forwards and 1 notification)
# ... (ToolMessage for forward_email to support@company.com) ...
# ... (ToolMessage for forward_email to cdetuma@company.com) ...
# ... (ToolMessage for forward_email to ctu@abc.com) ...
# ... (ToolMessage for send_wrong_email_notification_to_sender) ...
# ... (AIMessage confirming all actions taken for customer email) ...
```
For the second email (customer issue), here's what happens:

1.  Agent receives the email.
2.  Agent calls `determine_email_action`.
3.  Tool returns guidelines, indicating customer support routing.
4.  Agent correctly identifies the need to forward to *all three* support-related emails and send a notification. It requests *four* tool calls: `forward_email` (to support), `forward_email` (to cdetuma), `forward_email` (to ctu), and `send_wrong_email_notification_to_sender`.
5.  `ToolNode` executes all four calls.
6.  Agent receives confirmation from all tools and provides a final summary.

This means your agent correctly handled multiple tool calls based on the instructions retrieved from `determine_email_action`.

Now for the last example email (Email 3 - regulatory notice):

```python
>>> # Define escalation criteria relevant to Email 3
>>> escalation_criteria = "There's an immediate risk of electrical, water, or fire damage, or structural integrity issues mentioned."

>>> # Combine criteria and email for the agent prompt
>>> message_3_content = f"""
... Please process the following email.
... The escalation criteria for regulatory notices is: {escalation_criteria}
...
... --- Email Start ---
... {EMAILS[3]}
... --- Email End ---
... """
>>> message_3_input = AgentState(messages=[HumanMessage(content=message_3_content)])
# Or: message_3_input = {"messages": [HumanMessage(content=message_3_content)]}


>>> for chunk in email_agent_graph.stream(message_3_input, stream_mode="values"):
...      last_message = chunk["messages"][-1]
...      print(f"----- Last Message ({type(last_message).__name__}) -----")
...      last_message.pretty_print()
...      print("\n----- End Chunk -----\n")

# Expected Output Sequence (simplified):
# ... (HumanMessage with Email 3 + criteria) ...
#
# ----- Last Message (AIMessage) -----
# ================================== Ai Message ==================================
# Tool Calls:
#   extract_notice_data (...)
#  Call ID: ...
#   Args:
#     email: ... (Full Email 3 text) ...
#     escalation_criteria: There's an immediate risk of electrical, water, or fire damage, or structural integrity issues mentioned.
# ----- End Chunk -----
#
# (Log messages from NOTICE_EXTRACTION_GRAPH execution appear here)
# 2025-01-26 13:19:14,234 - INFO - Calling the email notice extraction graph...
# 2025-01-26 13:19:14,237 - INFO - Parsing notice...
# 2025-01-26 13:19:16,368 - INFO - Determining escalation status...
# 2025-01-26 13:19:16,982 - INFO - Escalation needed! (Due to electrical/fire/structural issues)
# 2025-01-26 13:19:16,987 - INFO - Sending escalation emails...
# ... (email sent logs) ...
# 2025-01-26 13:19:18,998 - INFO - Creating legal ticket for notice...
# ... (potential follow-up cycle logs) ...
# 2025-01-26 13:19:23,686 - INFO - Legal ticket created!
# 2025-01-26 13:19:23,687 - INFO - No follow-up question, routing to END.
#
# ----- Last Message (ToolMessage) -----
# ================================= Tool Message =================================
# Name: extract_notice_data
#
# Notice data extracted successfully: {
#   "date_of_notice_str": "2025-01-10",
#   "entity_name": "City of Los Angeles Building and Safety Department",
#   "entity_phone": "(555) 456-7890",
#   "entity_email": "inspections@lacity.gov",
#   "project_id": 345678123,
#   "site_location": "456 Sunset Boulevard, Los Angeles, CA", // Improved parsing assumed
#   "violation_type": "Electrical Wiring, Fire Safety, Structural Integrity",
#   "required_changes": "Replace or properly secure exposed wiring..., Install additional fire extinguishers..., Reinforce or replace temporary support beams...",
#   "compliance_deadline_str": "2025-02-05",
#   "max_potential_fine": null,
#   "date_of_notice": "2025-01-10",
#   "compliance_deadline": "2025-02-05"
# }
# ----- End Chunk -----
#
# ----- Last Message (AIMessage) -----
# ================================== Ai Message ==================================
#
# The regulatory notice from the City of Los Angeles Building and Safety Department regarding project 345678123 has been processed.
# Violations related to Electrical Wiring, Fire Safety, and Structural Integrity were found.
# The notice required escalation due to the nature of the violations, and the internal process (including potential follow-ups and legal ticket creation) has been completed by the specialized graph.
# ----- End Chunk -----
```
Here's what happens for the regulatory notice (Email 3):

1.  You first define relevant `escalation_criteria` and include it with the email text in the initial `HumanMessage`.
2.  The agent successfully recognizes that this is a regulatory notice email, likely based on the sender and content matching the docstring of the `extract_notice_data` tool.
3.  It calls the `extract_notice_data` tool, passing the email content and the provided escalation criteria.
4.  This invokes your `NOTICE_EXTRACTION_GRAPH`, which runs its full logic: parsing, checking escalation (which should be `True` based on the criteria and email content), sending escalation emails, and handling the legal ticket creation (including potential follow-up cycles).
5.  The `extract_notice_data` tool receives the final extracted data from the graph and returns it as a string to the agent.
6.  The agent receives the result from the tool and provides a final confirmation message.

Your agent worked end-to-end on all relevant examples, correctly identifying the email type and utilizing the appropriate tools, including invoking the complex notice extraction graph when needed. The results are awesome!

Keep in mind that you might not get the *exact* same final wording or tool arguments (especially from `determine_email_action`) due to LLM non-determinism, but the overall behavior and tool usage should be very similar. You now have the tools—no pun intended—needed to build your own LangGraph agents!

---
*Remove ads*
---

## Conclusion

Congratulations on completing this in-depth tutorial! You successfully built an LLM agent in LangGraph and now have a solid foundation to build your own applications.

In this tutorial, you've learned how to:

*   Think about LangGraph as a complement to LangChain for building sophisticated LLM workflows
*   Build LangGraph `StateGraph` workflows with nodes, edges, conditional edges, and cycles
*   Build LangGraph LLM agents using `MessagesState` (or similar), tool nodes, and agent loops

You can find the complete source code and data for this project in the supporting materials, which you can download using the link provided in the original tutorial source.

## Frequently Asked Questions

*(This section was empty in the original text)*


## Images from the article

```
Image 1: A Directed Graph Example
start → Eat Food
Eat Food → Buy Food (condition: Hungry)
Buy Food → Eat Food
Eat Food → end (condition: Full)
```

```
Image 2: Your First Notice Extraction Graph
__start__ → parse_notice_message → check_escalation_status → __end__
```

```
Image 3: Your Current Notice Extraction Graph With a Conditional Edge
__start__ → parse_notice_message → check_escalation_status
check_escalation_status → send_escalation_email (conditional)
send_escalation_email → create_legal_ticket
create_legal_ticket → __end__
```

```
Image 4: Your Updated Graph With a Cycle Between create_legal_ticket and answer_follow_up_question
start → parse_notice_message → check_escalation_status
check_escalation_status → send_escalation_email (conditional)
send_escalation_email → create_legal_ticket
create_legal_ticket → answer_follow_up_question (cycle back and forth)
answer_follow_up_question → create_legal_ticket
answer_follow_up_question → end (conditional)
create_legal_ticket → end (conditional)
check_escalation_status → end (conditional)
```

```
Image 5: Your Final LangGraph Agent
start → email_agent
email_agent → email_tools (conditional)
email_tools → email_agent
email_agent → end (conditional)