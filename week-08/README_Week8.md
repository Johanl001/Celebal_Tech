Single-Agent Smart Assistant — Conditional Tool Routing Pipeline

Week 8 Assignment | Data Science Internship

Author: Dhanraj Deshmukh

Overview
This project builds a single-agent pipeline that reads a natural language query, decides which tool (if any) it needs based on keyword-based intent detection, runs that tool, and returns a structured JSON response. The agent handles three categories of input — math expressions, keyword extraction requests, and everything else — and is built to fail safely rather than crash: malformed input, empty queries, and unmatched intents all resolve to a clean, predictable JSON shape instead of an unhandled exception.

The core question this assignment is really testing: can routing logic reliably tell the difference between "this looks like a math query," "this looks like a keyword query," and "this is neither, and possibly malformed," and can it do that without ever returning something that isn't valid, parseable JSON.

Project Structure
```
Week8_Single_Agent_Pipeline.ipynb   ← Main notebook (fully self-contained, executed with real output)
README.md                             ← This file
```

Problem Statement
Build a single-agent smart assistant that understands user queries, routes tasks based on intent, uses tools when required, and returns structured JSON output. The agent must handle:
- Math queries → Calculator Tool
- Keyword extraction requests → Keyword Tool
- General queries → Direct fallback response

Pipeline Stages

| # | Stage | Description |
|---|---|---|
| 1 | Tool Definitions | `calculator()` and `extract_keywords()` provided as-is, used unmodified by the agent |
| 2 | Expression Parsing Helper | `extract_expression()` pulls just the arithmetic substring out of a natural language query before handing it to `calculator()` |
| 3 | Agent Routing Logic | `agent()` — conditional routing on the lowercased query string, dispatches to the correct tool or the fallback handler |
| 4 | Bonus Tool | `count_words()` added as a third tool, triggered on "count words", to demonstrate the routing pattern extends cleanly |
| 5 | Structured Output | Every response follows `{"type": ..., "result": ...}`, printed via `json.dumps()` so it's valid JSON text, not a Python dict repr |
| 6 | Automated Test Cases | 6 queries covering all 5 response types (calculation, keywords, general, word_count, error) run through the agent |
| 7 | Validation Report | `agent_log` collects every call made; summarized as a DataFrame with a breakdown by response type |
| 8 | Interactive Mode | `while True` loop for typing queries in one at a time, wrapped to fail gracefully with no stdin available |

Routing Logic Design

| Trigger | Route | Behavior |
|---|---|---|
| Query contains "calculate" | Calculation | `extract_expression()` pulls out just the math substring first, since a raw query like "Calculate 20 + 5" isn't valid Python on its own — the word "Calculate" would break `eval()` |
| Query contains "keywords" | Keyword extraction | The full query is piped directly into `extract_keywords()` with no stripping, per the assignment's explicit instruction to pipe the text content directly |
| Query contains "count words" (bonus) | Word count | `count_words()` runs on the full query |
| No match | General | Returns a fallback message describing what the agent can actually do, instead of a generic error |
| Empty/whitespace-only query | Error | Caught before routing even starts |
| "calculate" present but no valid math found | Error | Distinguishes bad formatting from a working calculation |
| Unexpected exception anywhere in routing | Error | Caught by a top-level try/except so a genuine bug still returns clean JSON instead of crashing the agent |

Why parsing differs between routes: the assignment instructions explicitly say to "parse the mathematical expression" for the calculate route but to "pipe the text content directly" for the keyword route — that asymmetry is intentional, not an inconsistency. It also means `extract_expression()`'s regex (digits, decimal points, and math operators only) doubles as a safety filter before anything reaches `eval()`, since nothing outside that character set can survive the extraction step.

Sample Interactive Session
Actual output from running the notebook's interactive mode — one query per required route, plus a deliberately malformed one to confirm the error path holds up outside the built-in test cases:

```
Enter query (type 'exit' to stop): Calculate 45 * 3 - 10
Response: {
  "type": "calculation",
  "result": "125"
}

Enter query (type 'exit' to stop): Extract keywords from Climate change is accelerating faster than predicted
Response: {
  "type": "keywords",
  "result": [
    "keywords",
    "climate",
    "predicted",
    "accelerating",
    "change"
  ]
}

Enter query (type 'exit' to stop): Tell me a fun fact about space
Response: {
  "type": "general",
  "result": "I can help with calculations (say 'calculate ...'), keyword extraction (say 'extract keywords from ...'), or word counts (say 'count words in ...'). I don't have a specific tool for that yet."
}

Enter query (type 'exit' to stop): Calculate hello world
WARNING:agent:calculate route, no expression found in: Calculate hello world
Response: {
  "type": "error",
  "result": "Could not find a valid math expression in: 'Calculate hello world'"
}
```

One thing worth noting from this run: the keyword route pulled "keywords" itself into the result list (see the second response above). That's expected, not a bug — since the instructions call for piping the query directly into `extract_keywords()` without stripping trigger words first, any word over 4 characters in the original query is fair game, including "keywords" itself. It's a direct consequence of following the "pipe directly" instruction rather than an oversight in the extraction logic.

Automated Test Results
The notebook's built-in test cases exercise all 5 response types across 6 queries, logged via `agent_log` and confirmed in the Validation Report:

| Query | Type | Result |
|---|---|---|
| "Calculate 20 + 5" | calculation | 25 |
| "Extract keywords from Artificial Intelligence is transforming industries" | keywords | [extract, intelligence, industries, artificial, transforming] |
| "What is machine learning?" | general | fallback message |
| "Calculate banana + apple" | error | no valid expression found |
| "Count words in this sentence right here" | word_count | 7 |
| "" (empty string) | error | empty query received |

Validation report breakdown: 2 error, 1 calculation, 1 keywords, 1 general, 1 word_count — matching all 6 test queries with none dropped or misrouted.

Key Findings

| Area | Finding |
|---|---|
| Routing accuracy | All 6 automated test queries and all 4 interactive queries routed to the correct type, with zero unhandled exceptions |
| Error handling | Both required error cases (malformed math, empty query) return clean `{"type": "error", ...}` JSON rather than crashing or falling through to "general" |
| Keyword extraction quirk | Piping the query directly (per instructions) means trigger words like "keywords" or "extract" can appear in the result — expected behavior given how the tool is instructed to be called, not a defect |
| Safety of eval() | Restricting `extract_expression()`'s regex to digits/operators means nothing outside that character set reaches `eval()`, which is what makes using `eval()` here acceptable at all |
| Logging | Every one of the 10 total queries run across this notebook (6 automated + 4 interactive) is logged with both a `logging` module entry and an `agent_log` record |

Tech Stack

| Library | Purpose |
|---|---|
| re | Regex-based expression extraction for the calculate route |
| json | Structured JSON output via `json.dumps()` |
| logging | Per-call routing logs (INFO for successful routes, WARNING for handled errors) |
| pandas | Validation report — DataFrame summary of `agent_log` |

How to Run
```
# No external dependencies beyond the Python standard library and pandas.
pip install pandas

jupyter notebook Week8_Single_Agent_Pipeline.ipynb
```
Run all cells top to bottom. The automated test cases and validation report run immediately with no input needed. The final cell is interactive — it will prompt for queries in the notebook's own input box; type `exit` to stop. In a non-interactive environment (e.g. running via nbconvert with no stdin), that cell fails gracefully and prints a message instead of raising an error.

Evaluation Metrics

| Metric | Definition | Result |
|---|---|---|
| Routing accuracy | Correct route chosen for a given query intent | 10/10 across automated and interactive testing |
| JSON validity | Every response is valid, parseable JSON with both "type" and "result" keys | Confirmed via `json.dumps()` on every response, including error cases |
| Error safety | Malformed input and empty input handled without crashing | Both cases confirmed in automated tests |
| Log completeness | Every query processed has a corresponding log entry | 10/10 queries present in `agent_log` |

Week 8 Assignment — Data Science Internship
