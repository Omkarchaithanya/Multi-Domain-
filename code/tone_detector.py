import re
from typing import Tuple


def detect_tone(ticket_text: str) -> str:
    """
    Innovation 5: Tone and Urgency Adaptation
    
    Classifies the emotional state of the ticket using simple rule-based signals.
    This allows response generation to be adapted to the user's emotional context.
    
    Returns: One of ("frustrated", "urgent", "confused", "angry", "neutral")
    """
    text_lower = ticket_text.lower()
    
    # Check for urgent tone (highest priority)
    urgent_patterns = r"\b(urgent|asap|immediately|right now|losing money|today|before|deadline|critical)\b"
    if re.search(urgent_patterns, text_lower):
        return "urgent"
    
    # Check for angry tone
    angry_patterns = r"\b(worst|terrible|disgusting|scam|fraud|lawsuit|complaint|unacceptable|ridiculous|disgusted|furious|outrageous)\b"
    if re.search(angry_patterns, text_lower):
        return "angry"
    
    # Check for frustrated tone
    # Signals: repeated action words, exclamation marks, strong negative words
    frustrated_patterns = r"\b(still|again|already|multiple times|keeps|cannot|can't|won't|doesn't|frustrated|annoyed|fed up)\b"
    frustrated_intensity = len(re.findall(r"!!|\.\.\.|\?\?", text_lower))
    if re.search(frustrated_patterns, text_lower) or frustrated_intensity >= 2:
        return "frustrated"
    
    # Check for confused tone
    # Signals: question words, uncertainty markers, lack of frustration signals
    confused_patterns = r"\b(don't understand|not sure|confused|unclear|what does|how do i|why is|don't know|unsure|help me understand)\b"
    if re.search(confused_patterns, text_lower):
        # Make sure it's not also angry/frustrated
        if not re.search(angry_patterns, text_lower) and not re.search(frustrated_patterns, text_lower):
            return "confused"
    
    # Check for multiple question marks indicating confusion/frustration
    question_count = text_lower.count("?")
    if question_count >= 3:
        return "confused"
    
    # Default to neutral
    return "neutral"


def get_tone_instruction(tone: str) -> str:
    """
    Map tone classification to a response generation instruction.
    These instructions are passed to the LLM to adapt response generation.
    """
    tone_instructions = {
        "frustrated": (
            "Acknowledge the user's frustration explicitly in the first sentence. "
            "Be brief and action-oriented. Get to the solution quickly."
        ),
        "urgent": (
            "Lead immediately with the fastest resolution path. "
            "No preamble. Be direct and concise."
        ),
        "confused": (
            "Use simple, plain language. Explain each step clearly. "
            "Break complex information into small, digestible chunks."
        ),
        "angry": (
            "Stay calm and professional. Acknowledge the issue sincerely. "
            "De-escalate before explaining the solution."
        ),
        "neutral": (
            "Use a warm, professional, and helpful tone."
        ),
    }
    return tone_instructions.get(tone, tone_instructions["neutral"])
