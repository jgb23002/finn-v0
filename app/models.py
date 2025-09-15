from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class AskRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None  # e.g., {"avg_steps_7d": 5200, "consented": True}

class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    oos: bool
