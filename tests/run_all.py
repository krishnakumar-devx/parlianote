"""Run every test and print one summary line per suite:  python tests/run_all.py

The app test starts the real app with odd inputs, so it needs the data in app/data/ and takes a few minutes.
Use `python tests/run_all.py --fast` to run only the quick offline tests.
"""

import subprocess
import sys

SUITES = [
    ("core logic (offline)", "tests/test_core.py", True),
    ("AI answer module (offline)", "tests/test_llm.py", True),
    ("whole app with odd inputs", "tests/test_app_edge_cases.py", False),
]
fast_only = "--fast" in sys.argv
bad = 0
for name, path, is_fast in SUITES:
    if fast_only and not is_fast:
        print(f"SKIP  {name}")
        continue
    result = subprocess.run([sys.executable, path], capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = [line for line in result.stdout.splitlines() if "passed" in line or "FAILED" in line or line.startswith("FAIL")]
    print(("OK    " if result.returncode == 0 else "FAIL  ") + name + (f"   ({tail[-1].strip()})" if tail else ""))
    bad += result.returncode != 0
sys.exit(1 if bad else 0)
