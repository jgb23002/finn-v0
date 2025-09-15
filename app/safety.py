import re

# Simple OOS/medical patterns (extend as needed)
OOS_PATTERNS = re.compile(
    r"(diagnos|dose|dosage|mg\b|prescrib|rx\b|antibiotic|antidepressant|"
    r"suicid|overdose|chest pain|severe bleed|stroke|heart attack|fracture|"
    r"pregnan|contraindicat|insulin|opioid)", re.IGNORECASE
)

def is_oos(text: str) -> bool:
    return bool(OOS_PATTERNS.search(text))

REFUSAL = (
    "I can’t provide medical advice or handle urgent issues. "
    "If this is an emergency, please contact local emergency services. "
    "For non-urgent concerns, consider speaking with a licensed clinician."
)
