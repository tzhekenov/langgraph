from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Load environment variables (ensure .env file is present)
from dotenv import load_dotenv
import os

load_dotenv()

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


# Example usage for testing
if __name__ == "__main__":
    print("Testing ESCALATION_CHECK_CHAIN...")

    escalation_criteria_water = "There is currently water damage or potential water damage reported"
    message_water = "Several cracks in the foundation have been identified along with water leaks"
    message_no_water = "The wheel chair ramps are too steep"

    result_water = ESCALATION_CHECK_CHAIN.invoke(
        {"message": message_water, "escalation_criteria": escalation_criteria_water}
    )
    print(f"Message: '{message_water}' -> Escalates (water): {result_water.needs_escalation}")

    result_no_water = ESCALATION_CHECK_CHAIN.invoke(
        {"message": message_no_water, "escalation_criteria": escalation_criteria_water}
    )
    print(f"Message: '{message_no_water}' -> Escalates (water): {result_no_water.needs_escalation}")

    # Test with EMAILS[0]
    try:
        from example_emails import EMAILS
    except ImportError:
        print("Run this script from the project root directory or adjust import path.")
        exit()

    escalation_criteria_safety = "Workers explicitly violating safety protocols"
    result_safety = ESCALATION_CHECK_CHAIN.invoke(
        {"message": EMAILS[0], "escalation_criteria": escalation_criteria_safety}
    )
    print(f"Message: EMAILS[0] -> Escalates (safety): {result_safety.needs_escalation}") 