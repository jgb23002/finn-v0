SYSTEM = """You are Finn, a supportive wellness guide.
- Use only the supplied knowledge and (optional) consented app context.
- Do not provide medical diagnoses, medication guidance, or emergency advice.
- If the request is out-of-scope, decline briefly and suggest contacting a clinician.
- Be concise, empathetic, and action-oriented.
"""

def compose_prompt(snippets: list[str], user_msg: str, ctx: dict | None) -> str:
    ctx_line = ""
    if ctx and ctx.get("consented"):
        safe_ctx = {k: v for k, v in ctx.items() if k != "consented"}
        ctx_line = f"\n[User context]: {safe_ctx}"
    kb = "\n\n".join(f"[Doc {i+1}]\n{chunk}" for i, chunk in enumerate(snippets))
    return f"{SYSTEM}\n\n[Knowledge]\n{kb}{ctx_line}\n\n[User]\n{user_msg}\n\n[Assistant]"
