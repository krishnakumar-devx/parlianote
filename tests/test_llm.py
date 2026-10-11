"""Offline tests for src/llm.py (no key or network needed): python tests/test_llm.py"""

import sys
import requests

sys.path.insert(0, ".")
from src import llm

P = [
    {
        "speaker": "Chalmers, Jim",
        "party": "ALP",
        "date": "20 Jun 2023",
        "text": "We have set aside $1.7 billion and 20,000 university places to help households.",
    },
    {
        "speaker": "Plibersek, Tanya",
        "party": "ALP",
        "date": "13 Feb 2008",
        "text": "Housing affordability is at its worst ever and the previous government were in denial.",
    },
]
fails = []


def check(name, cond):
    """Print and record one PASS or FAIL result."""
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


msgs = llm.build_messages("housing?", P)
check(
    "prompt numbers passages and names speakers",
    "[1] Chalmers, Jim (ALP), 20 Jun 2023" in msgs[1]["content"] and "[2] Plibersek" in msgs[1]["content"],
)
check("system prompt forbids outside facts", "ONLY" in msgs[0]["content"] and "do not answer" in msgs[0]["content"])

check(
    "good cited answer passes",
    llm.check_answer("Chalmers said $1.7 billion was set aside [1]. Plibersek said housing was at its worst [2].", P) == [],
)
check("abstention passes", llm.check_answer(llm.NO_ANSWER, P) == [])
check("invented number is flagged", any("figures" in w for w in llm.check_answer("Chalmers set aside $9 billion [1].", P)))
check("missing citations flagged", any("no citations" in w for w in llm.check_answer("Chalmers talked about households and funding.", P)))
check("non-existent citation flagged", any("does not exist" in w for w in llm.check_answer("Chalmers spoke about households [7].", P)))
check(
    "uncited sentence flagged",
    any(
        "no citation" in w
        for w in llm.check_answer("Chalmers set aside $1.7 billion [1]. Everyone agrees this was a great decision for the country.", P)
    ),
)
check("comma numbers match", llm.check_answer("About 20,000 places were funded [1].", P) == [])

# provider selection and error handling with the network call stubbed
for k in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
    llm.os.environ.pop(k, None)
llm.requests.get = lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError())
check("no key and no Ollama -> provider None", llm.provider() is None)
r = llm.answer("q", P)
check("answer() without provider returns an error, not an exception", r["error"] and not r["text"])


class FakeTags:
    """A stand-in for Ollama's /api/tags reply listing the given models."""

    ok = True

    def __init__(self, models):
        self.models = models

    def json(self):
        return {"models": self.models}


def ollama_with(models):
    """provider() as it would be with Ollama running and these models pulled (OLLAMA_MODEL left at its default)."""
    llm.os.environ.pop("OLLAMA_MODEL", None)
    llm.requests.get = lambda *a, **k: FakeTags(models)
    return llm.provider()


check("Ollama running but no model pulled -> AI off, not a 404", ollama_with([]) is None)
check(
    "Ollama with only an embedding model -> AI off",
    ollama_with([{"name": "nomic-embed-text:latest", "capabilities": ["embedding"]}]) is None,
)
check("configured Ollama model installed -> used as is", ollama_with([{"name": "llama3.2:latest"}])["model"] == "llama3.2")
check(
    "configured Ollama model missing -> falls back to an installed Llama",
    ollama_with([{"name": "gemma4:e4b"}, {"name": "llama3.1:8b"}])["model"] == "llama3.1:8b",
)
check("no Llama installed -> falls back to the first installed model", ollama_with([{"name": "mistral:7b"}])["model"] == "mistral:7b")
llm.requests.get = lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError())
llm.os.environ["OPENAI_API_KEY"] = "sk-test"
check("key present -> OpenAI selected", llm.provider()["name"] == "OpenAI")
llm.os.environ["AI_ANSWERS"] = "0"
check("AI_ANSWERS=0 switches it off", llm.provider() is None)
llm.os.environ["AI_ANSWERS"] = "1"

llm._call = lambda messages, prov: "Chalmers said $1.7 billion was set aside [1]."
r = llm.answer("q", P)
check("stubbed model reply is returned and checked", r["text"].startswith("Chalmers") and r["warnings"] == [] and "OpenAI" in r["provider"])
llm._call = lambda messages, prov: "Chalmers said $50 billion was set aside [1]."
check("fabricated figure is caught", llm.answer("q", P)["warnings"] != [])


def http_error(code):
    """A fake HTTP error with the given status code."""
    resp = requests.Response()
    resp.status_code = code
    return requests.HTTPError(response=resp)


def quota_error():
    """A fake 'account out of credit' error, as OpenAI returns it."""
    resp = requests.Response()
    resp.status_code = 429
    resp._content = b'{"error": {"type": "insufficient_quota", "code": "credit_balance_exhausted"}}'
    return requests.HTTPError(response=resp)


llm._call = lambda messages, prov: (_ for _ in ()).throw(quota_error())
check("out-of-credit error is explained clearly", "no credits left" in llm.answer("q", P)["error"])
for code, word in [(401, "key was rejected"), (429, "busy"), (500, "error 500")]:
    llm._call = lambda messages, prov, c=code: (_ for _ in ()).throw(http_error(c))
    check(f"HTTP {code} gives a friendly message", word in llm.answer("q", P)["error"])
llm._call = lambda messages, prov: (_ for _ in ()).throw(requests.ConnectionError())
check("no internet gives a friendly message", "internet" in llm.answer("q", P)["error"])
llm._call = lambda messages, prov: (_ for _ in ()).throw(KeyError("choices"))
check("unexpected reply format handled", "unexpected" in llm.answer("q", P)["error"])

print(f"\n{'ALL' if not fails else len(fails)} {'passed' if not fails else 'FAILED'}")
sys.exit(1 if fails else 0)
