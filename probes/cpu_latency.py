"""Client-side latency of jeff-serve per adapter: each adapter's bundled example.json, plus a Hermes-shaped
`tools` request (short summary: goal, last turn, Hermes tool names). Sequential, one request at a time."""
import json
import math
import statistics
import sys
import time
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8765/v1/systemone"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
ROOT = Path(__file__).resolve().parent.parent

HERMES_TOOLS = {
    "answer_directly": "No tool is needed: answer the user directly from the conversation",
    "ask_user": "A tool is needed but required information is missing: ask the user first",
    "read_file": "read_file(path): read a text file from disk",
    "search_files": "search_files(pattern, path): search file contents or names",
    "terminal": "terminal(command): run a shell command",
    "write_file": "write_file(path, content): write a file",
    "patch": "patch(path, old, new): edit part of a file",
    "web_search": "web_search(query): search the web",
    "memory": "memory(action, text): save or read persistent notes about the user",
}
HERMES_REQ = {
    "model": "tools",
    "state": {
        "agent": "A personal assistant with shell, file, web and memory tools on the user's Linux workstation.",
        "conversation": [{"role": "user", "text": "Is the arr-watchdog service healthy?"},
                         {"role": "assistant", "text": "Checking its status now."}],
        "user_message": "Also, what's in ~/.hermes/SEERR_HANDOFF.md?",
    },
    "questions": {"tool": {
        "type": "choice",
        "instructions": "Which tool should the agent call next to handle the user's latest message? Use the "
                        "conversation for context. If no tool is needed, choose answer directly. If a tool is "
                        "needed but information it requires is missing, choose ask the user.",
        "criteria": HERMES_TOOLS}},
}


def post(body: dict) -> tuple[float, dict]:
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        out = json.load(r)
    return (time.perf_counter() - t) * 1000, out


cases = {f"{p.parent.name} (example)": json.loads(p.read_text())
         for p in sorted((ROOT / "models/adapters").glob("*/example.json"))}
cases["tools (hermes-shaped)"] = HERMES_REQ
cases["base, no adapter (hermes-shaped)"] = {**HERMES_REQ, "model": "jeff"}

print("| case | input tokens | median ms | p95 ms | min ms | top choice |\n|---|---:|---:|---:|---:|---|")
for name, body in cases.items():
    for _ in range(3):
        post(body)  # warm-up
    times, out = [], None
    for _ in range(N):
        ms, out = post(body)
        times.append(ms)
    times.sort()
    first = next(iter(out["answers"].values()))
    top = f"{first.get('choice', first.get('value'))} p={max(first.get('probabilities', {0: 0}).values()):.2f}"
    print(f"| {name} | {out['usage']['input_tokens']} | {statistics.median(times):.0f} | "
          f"{times[math.ceil(0.95 * N) - 1]:.0f} | {times[0]:.0f} | {top} |")
