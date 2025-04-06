from datetime import datetime, date
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, computed_field, EmailStr # Added EmailStr

# Load environment variables (ensure .env file is present)
from dotenv import load_dotenv
import os

load_dotenv()
# Optional: Check if the key is loaded
# if not os.getenv("OPENAI_API_KEY"):
#     print("Warning: OPENAI_API_KEY not found in .env file.")

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
    entity_email: EmailStr | None = Field( # Changed to EmailStr for validation
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
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception as e:
            # Consider logging the error instead of printing
            # print(f"Error converting date string '{date_str}': {e}")
            return None

    @computed_field(repr=True) # Changed repr to True to show in output
    @property
    def date_of_notice(self) -> date | None:
        return self._convert_string_to_date(self.date_of_notice_str)

    @computed_field(repr=True) # Changed repr to True to show in output
    @property
    def compliance_deadline(self) -> date | None:
        return self._convert_string_to_date(self.compliance_deadline_str)


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


# Example usage for testing
if __name__ == "__main__":
    # Make sure to run this from the root of the project or adjust path
    try:
        from example_emails import EMAILS
    except ImportError:
        print("Run this script from the project root directory or adjust import path.")
        exit()

    print("Testing NOTICE_PARSER_CHAIN on EMAILS[0]...")
    result = NOTICE_PARSER_CHAIN.invoke({"message": EMAILS[0]})
    print(result)
    print("\nTesting NOTICE_PARSER_CHAIN on EMAILS[3]...")
    result_3 = NOTICE_PARSER_CHAIN.invoke({"message": EMAILS[3]})
    print(result_3) 