import json
import subprocess

def generate_ollama(prompt: str, model: str = "llama3.1:8b-instruct", max_tokens: int = 350) -> str:
    """
    Streams JSON lines from `ollama generate`.
    Requires Ollama installed and the model pulled: `ollama pull llama3.1:8b-instruct`.
    """
    try:
        proc = subprocess.Popen(
            ["ollama", "generate", model, "--json"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(prompt.encode("utf-8"))
        proc.stdin.close()

        text = []
        for line in proc.stdout:
            try:
                obj = json.loads(line.decode("utf-8"))
                if "response" in obj:
                    text.append(obj["response"])
            except json.JSONDecodeError:
                continue
        proc.wait()
        return "".join(text).strip()
    except Exception as e:
        # Fallback message if Ollama isn't available
        return ("(LLM unavailable) Here’s a concise, general wellness tip based on the provided docs: "
                "Focus on consistent routines, small habit changes, and adequate rest and hydration.")
