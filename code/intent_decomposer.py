import json
import os
from typing import List


def decompose_ticket(ticket_text: str) -> List[str]:
    """
    Innovation 1: Multi-Intent Decomposer
    
    Splits a compound support ticket into distinct sub-questions/intents.
    Most support tickets contain a single intent, but some (like "I can't log in 
    AND was charged twice") contain multiple support problems that should be 
    retrieved and answered independently.
    
    Returns: List of sub-questions (minimum 1, maximum 3)
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        # Fallback: return ticket as single intent
        return [ticket_text]
    
    try:
        import anthropic
        
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            temperature=0,
            system="""You are a support ticket analyzer. Extract the distinct support intents from this ticket.
Return a JSON array of strings, each being one clear sub-question or intent.
Maximum 3 items. If the ticket has only one intent, return a single-item array.
Return ONLY valid JSON, no explanation.
Example: ["Can't reset password", "Account locked after failed attempts"]""",
            messages=[{"role": "user", "content": ticket_text[:2000]}],
        )
        
        response_text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        ).strip()
        
        # Parse JSON response
        try:
            intents = json.loads(response_text)
            if isinstance(intents, list) and intents:
                # Filter to non-empty strings and limit to 3
                intents = [str(i).strip() for i in intents if str(i).strip()][:3]
                if intents:
                    return intents
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: return ticket as single intent
        return [ticket_text]
    
    except Exception:
        # API error or other issue: return ticket as single intent
        return [ticket_text]
