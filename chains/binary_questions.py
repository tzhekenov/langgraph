from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Load environment variables (ensure .env file is present)
from dotenv import load_dotenv
import os

load_dotenv()

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

# Example usage for testing
if __name__ == "__main__":
    print("Testing BINARY_QUESTION_CHAIN...")

    # Test with EMAILS[0]
    try:
        from example_emails import EMAILS
    except ImportError:
        print("Run this script from the project root directory or adjust import path.")
        exit()

    context_0 = EMAILS[0]
    question_texas = "Does this message mention the states of Texas, Georgia, or New Jersey?"
    question_hvac = "Did this notice involve an issue with FakeAirCo's HVAC system?"
    question_compliance = "Is the compliance deadline November 10, 2024?"

    result_texas = BINARY_QUESTION_CHAIN.invoke({
        "question": question_texas,
        "context": context_0
        })
    print(f"Q: '{question_texas}' -> A: {result_texas.is_true}")

    result_hvac = BINARY_QUESTION_CHAIN.invoke({
        "question": question_hvac,
        "context": context_0
        })
    print(f"Q: '{question_hvac}' -> A: {result_hvac.is_true}")

    result_compliance = BINARY_QUESTION_CHAIN.invoke({
        "question": question_compliance,
        "context": context_0
        })
    print(f"Q: '{question_compliance}' -> A: {result_compliance.is_true}") 