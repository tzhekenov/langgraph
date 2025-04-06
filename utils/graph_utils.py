import random
import time
from pydantic import EmailStr
# Use try-except for conditional import based on relative path
try:
    from chains.notice_extraction import NoticeEmailExtract
    from utils.logging_config import LOGGER
except ImportError:
    # Handle case where script is run directly or imports fail
    # This might happen if run from utils/ directory directly
    print("Attempting import relative to project root...")
    import sys
    import os
    # Add project root to path assuming utils is one level down
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from chains.notice_extraction import NoticeEmailExtract
    from utils.logging_config import LOGGER


def send_escalation_email(
    notice_email_extract: NoticeEmailExtract,
    escalation_emails: list[EmailStr] | None # Allow None
) -> None:
    """Simulate sending escalation emails"""
    if not escalation_emails:
        LOGGER.warning("No escalation emails provided. Skipping email simulation.")
        return

    LOGGER.info(f"Simulating sending escalation emails to: {', '.join(escalation_emails)}")
    for email in escalation_emails:
        # Simulate API call delay
        time.sleep(0.5 + random.random() * 0.5) # Shorter delay
        LOGGER.info(f"---> Escalation details sent to {email}")
    LOGGER.info("Finished sending all escalation emails.")

def create_legal_ticket(
    current_follow_ups: dict[str, bool] | None,
    notice_email_extract: NoticeEmailExtract,
) -> str | None:
    """Simulate creating a legal ticket using your company's API.
    Returns a follow-up question if required, otherwise None.
    """
    LOGGER.info("Attempting to create legal ticket for notice...")
    # Simulate API call delay
    time.sleep(1 + random.random())

    # Pool of potential follow-up questions (including None for no question)
    follow_ups_pool = [
        None,
        "Does this message mention the states of Texas, Georgia, or New Jersey?",
        "Did this notice involve an issue with FakeAirCo's HVAC system?",
        # Add more potential questions here if desired
    ]

    # Filter out questions already answered
    answered_questions = set(current_follow_ups.keys()) if current_follow_ups else set()
    available_follow_ups = [q for q in follow_ups_pool if q not in answered_questions]

    # If only None is left (or pool was just None), or no questions remain
    if not available_follow_ups or all(q is None for q in available_follow_ups):
        # Random chance of *still* asking a question even if answered, for simulation realism?
        # Let's keep it simple: if no available questions, create the ticket.
        LOGGER.info("*** Legal ticket successfully created (simulation). ***")
        return None

    # Choose a follow-up question randomly from the available ones
    follow_up = random.choice(available_follow_ups)

    if follow_up is None:
        LOGGER.info("*** Legal ticket successfully created (simulation). ***")
        return None
    else:
        LOGGER.info(f"---> Follow-up required before creating ticket: '{follow_up}'")
        return follow_up

# Example usage for testing
if __name__ == "__main__":
    # Create a dummy NoticeEmailExtract for testing
    dummy_extract = NoticeEmailExtract(
        date_of_notice_str="2024-10-15",
        entity_name="Test OSHA",
        entity_email="test@osha.gov",
        project_id=12345
    )

    print("\n--- Testing send_escalation_email ---")
    send_escalation_email(dummy_extract, ["manager@example.com", "legal@example.com"])
    send_escalation_email(dummy_extract, None)

    print("\n--- Testing create_legal_ticket --- ")
    follow_ups_answered = {}
    print("Attempt 1:")
    q1 = create_legal_ticket(follow_ups_answered, dummy_extract)
    if q1:
        follow_ups_answered[q1] = random.choice([True, False]) # Simulate answering
        print(f"Answered Q1: {q1} -> {follow_ups_answered[q1]}")

    print("\nAttempt 2:")
    q2 = create_legal_ticket(follow_ups_answered, dummy_extract)
    if q2:
        follow_ups_answered[q2] = random.choice([True, False])
        print(f"Answered Q2: {q2} -> {follow_ups_answered[q2]}")

    print("\nAttempt 3 (should likely create ticket now):")
    q3 = create_legal_ticket(follow_ups_answered, dummy_extract)
    if q3:
        follow_ups_answered[q3] = random.choice([True, False])
        print(f"Answered Q3: {q3} -> {follow_ups_answered[q3]}")

    print(f"\nFinal answered follow-ups: {follow_ups_answered}") 