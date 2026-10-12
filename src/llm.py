"""Optional AI-written answers, grounded in the retrieved passages (the 'generation' half of RAG).

The model only sees numbered passages from Hansard and must cite them as [1], [2]. Every answer is checked afterwards:
citations must exist, and any number in the answer must appear in the cited passages. If no key is configured the app
simply keeps showing the verbatim sentences.

Settings come from the .env file in the project root (see .env.example). Keys are never logged or shown.
"""

import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

SYSTEM_PROMPT = (
    "You answer questions about what Australian MPs said in the House of Representatives, using ONLY the numbered passages provided. "
    "Rules: (1) Use plain English a non-expert can follow, in at most 120 words. "
    "(2) Put a citation like [1] or [2][3] after every claim. (3) Never add facts, numbers, dates or names that are not in the passages. "
    "(4) Say who said it when the passage names a speaker. (5) If the passages do not answer the question, reply exactly: "
    '"The passages do not answer this question." (6) Do not give your own opinion or judge anyone.'
)

NO_ANSWER = "The passages do not answer this question."


def provider() -> dict | None:
    """Pick the first usable provider from the environment, or None."""
    if os.getenv("AI_ANSWERS", "1") == "0":
        return None
    if os.getenv("OPENAI_API_KEY", "").strip():
        return {"name": "OpenAI", "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini")}
    if os.getenv("ANTHROPIC_API_KEY", "").strip():
        return {"name": "Anthropic", "model": os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")}
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        tags = requests.get(host + "/api/tags", timeout=1.0)
        if tags.ok:
            wanted = os.getenv("OLLAMA_MODEL", "llama3.2")
            # embedding-only models cannot write answers; older Ollama versions do not list capabilities, so keep those
            installed = [m.get("name", "") for m in tags.json().get("models", []) if "completion" in m.get("capabilities", ["completion"])]
            if not installed:  # Ollama is running but has no usable model pulled, so AI answers stay off
                return None
            # asking for a model that is not pulled gives a 404, so fall back to an installed one (Llama first, it follows the citation rules well)
            if wanted not in installed and f"{wanted}:latest" not in installed:
                wanted = next((m for m in installed if m.startswith("llama")), installed[0])
            return {"name": "Ollama (local)", "model": wanted}
    except (requests.RequestException, ValueError):
        pass
    return None


def build_messages(question: str, passages: list[dict]) -> list[dict]:
    """passages: dicts with speaker, party, date, text."""
    blocks = [f"[{i}] {p['speaker']} ({p['party']}), {p['date']}: {p['text']}" for i, p in enumerate(passages, 1)]
    user = f"Question: {question}\n\nPassages:\n" + "\n\n".join(blocks) + "\n\nAnswer using only these passages."
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _call(messages: list[dict], prov: dict) -> str:
    name = prov["name"]
    if name == "OpenAI":
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            timeout=60,
            headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"].strip()},
            json={"model": prov["model"], "messages": messages, "temperature": 0, "max_tokens": 300},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    if name == "Anthropic":
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            timeout=60,
            headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"].strip(), "anthropic-version": "2023-06-01"},
            json={"model": prov["model"], "system": messages[0]["content"], "messages": messages[1:], "temperature": 0, "max_tokens": 300},
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"]).strip()
    r = requests.post(
        os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/api/chat",
        timeout=180,
        json={"model": prov["model"], "messages": messages, "stream": False, "options": {"temperature": 0}},
    )
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


_NUM = re.compile(r"\d[\d,]*\.?\d*")
_CITE = re.compile(r"\[(\d+)\]")


def check_answer(answer: str, passages: list[dict]) -> list[str]:
    """Return warnings (empty = passed). Checks citations exist, every sentence is cited, numbers appear in the cited passages."""
    if answer.strip() == NO_ANSWER:
        return []
    warns, n = [], len(passages)
    cited = {int(m) for m in _CITE.findall(answer)}
    if not cited:
        warns.append("The answer has no citations, so it cannot be checked.")
    if any(c < 1 or c > n for c in cited):
        warns.append("The answer cites a passage that does not exist.")
    uncited = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if len(s.split()) > 4 and not _CITE.search(s)]
    if uncited:
        warns.append("Some sentences have no citation.")
    source = " ".join(passages[c - 1]["text"] for c in cited if 1 <= c <= n).replace(",", "")
    bad = [
        x for x in _NUM.findall(_CITE.sub("", answer)) if x.strip(".,").replace(",", "") and x.strip(".,").replace(",", "") not in source
    ]
    if bad:
        warns.append("These figures were not found in the cited passages: " + ", ".join(sorted(set(x.strip(".,") for x in bad))) + ".")
    return warns


def answer(question: str, passages: list[dict]) -> dict:
    """Return {'text', 'warnings', 'provider', 'error'}. Never raises."""
    prov = provider()
    if prov is None:
        return {"text": "", "warnings": [], "provider": None, "error": "No AI provider is configured."}
    try:
        text = _call(build_messages(question, passages), prov)
        return {"text": text, "warnings": check_answer(text, passages), "provider": f"{prov['name']} ({prov['model']})", "error": None}
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        try:
            detail = (e.response.json().get("error") or {}).get("code") or ""
        except (ValueError, AttributeError):
            detail = ""
        if detail in ("insufficient_quota", "credit_balance_exhausted"):
            hint = f"The {prov['name']} account has no credits left. Add credits in its billing settings, then try again."
        else:
            hint = {
                401: "The API key was rejected. Check the key in your .env file.",
                429: "The AI service is busy. Please try again in a minute.",
            }.get(code, f"The AI service returned error {code}.")
        return {"text": "", "warnings": [], "provider": prov["name"], "error": hint}
    except requests.RequestException:
        return {
            "text": "",
            "warnings": [],
            "provider": prov["name"],
            "error": "Could not reach the AI service. Check your internet connection.",
        }
    except (KeyError, ValueError):
        return {"text": "", "warnings": [], "provider": prov["name"], "error": "The AI service returned an unexpected reply."}
