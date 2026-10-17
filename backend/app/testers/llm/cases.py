"""Capability evaluation cases, sandboxed code execution, and LLM-as-a-Judge helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Callable


@dataclass
class EvaluationOutcome:
    """Result of evaluating a single capability case locally."""
    passed: bool
    message: str
    checks: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityCase:
    """Definition of a built-in LLM capability test case."""
    id: str
    suite: str
    name: str
    category: str
    description: str
    prompt: str
    system_prompt: str
    evaluator: Callable[..., EvaluationOutcome]
    sample_passing_response: str


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract and parse a JSON object from markdown code blocks or conversational text."""
    if not text:
        return None

    # 1. Try ```json ... ``` or ``` ... ``` fenced code block
    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    candidate = fenced_match.group(1).strip() if fenced_match else text.strip()

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 2. Fallback: find outermost '{' ... '}' in text
    for source in (candidate, text):
        start_idx = source.find("{")
        end_idx = source.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                parsed = json.loads(source[start_idx : end_idx + 1])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue

    return None


def extract_python_code(text: str) -> str:
    """Extract Python source code from markdown code fences or raw text."""
    if not text:
        return ""
    match = re.search(r"```(?:python|py)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def coerce_judge_score(value: Any) -> int | None:
    """Coerce an untrusted LLM judge score to a bounded integer in [1, 5], or None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        num = int(round(value))
    else:
        try:
            num = int(round(float(str(value).strip())))
        except (TypeError, ValueError):
            return None
    if 1 <= num <= 5:
        return num
    return None


# ===========================================================================
# 1. Coding Capability Evaluator (Isolated Subprocess Execution)
# ===========================================================================

_SANDBOX_RUNNER_SCRIPT = r"""
import builtins
import json
import sys

payload = json.loads(sys.stdin.read())
code = payload["code"]
test_cases = payload["test_cases"]

ALLOWED_IMPORTS = {
    "re", "math", "itertools", "functools", "collections",
    "typing", "string", "heapq", "bisect", "operator",
}
_real_import = builtins.__import__

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORTS:
        raise ImportError(f"Import of module '{name}' is not permitted in sandbox")
    return _real_import(name, globals, locals, fromlist, level)

safe_builtins = dict(builtins.__dict__)
for forbidden in ("open", "eval", "exec", "compile", "input", "breakpoint"):
    safe_builtins.pop(forbidden, None)
safe_builtins["__import__"] = _safe_import

namespace = {"__builtins__": safe_builtins, "__name__": "__sandbox__"}

try:
    exec(compile(code, "<llm_code>", "exec"), namespace, namespace)
except Exception as e:
    print(json.dumps({
        "passed": False,
        "error": f"Code compilation/execution error: {type(e).__name__}: {e}",
        "checks": [],
    }))
    sys.exit(0)

rle_encode = namespace.get("rle_encode")
rle_decode = namespace.get("rle_decode")

if not callable(rle_encode) or not callable(rle_decode):
    print(json.dumps({
        "passed": False,
        "error": "Both 'rle_encode(s)' and 'rle_decode(s)' functions must be defined.",
        "checks": [],
    }))
    sys.exit(0)

checks = []
all_passed = True

for orig, expected_enc, desc in test_cases:
    try:
        actual_enc = rle_encode(orig)
        enc_ok = (actual_enc == expected_enc)
        checks.append({
            "name": f"encode ({desc})",
            "passed": enc_ok,
            "detail": f"rle_encode({orig!r}) -> {actual_enc!r} (expected {expected_enc!r})",
        })
        if not enc_ok:
            all_passed = False

        actual_dec = rle_decode(expected_enc)
        dec_ok = (actual_dec == orig)
        checks.append({
            "name": f"decode ({desc})",
            "passed": dec_ok,
            "detail": f"rle_decode({expected_enc!r}) -> {actual_dec!r} (expected {orig!r})",
        })
        if not dec_ok:
            all_passed = False
    except Exception as exc:
        all_passed = False
        checks.append({
            "name": f"exception ({desc})",
            "passed": False,
            "detail": f"{type(exc).__name__}: {exc}",
        })

print(json.dumps({
    "passed": all_passed,
    "error": None,
    "checks": checks,
}))
"""


def evaluate_coding(response_text: str, timeout_seconds: float = 5.0) -> EvaluationOutcome:
    """Extract Python code and execute unit tests in an isolated subprocess."""
    code = extract_python_code(response_text)
    if not code:
        return EvaluationOutcome(
            passed=False,
            message="Failed: Response did not contain any code.",
            checks=[{"name": "code_present", "passed": False, "detail": "Empty code block"}],
        )

    if "def rle_encode" not in code or "def rle_decode" not in code:
        return EvaluationOutcome(
            passed=False,
            message="Failed: Functions 'rle_encode' and/or 'rle_decode' were not defined.",
            checks=[
                {
                    "name": "function_signatures",
                    "passed": False,
                    "detail": "Missing 'def rle_encode' or 'def rle_decode'",
                }
            ],
            extra={"extracted_code": code},
        )

    test_vectors = [
        ("AAAAABBBCC", "5A3B2C", "standard run-length sequence"),
        ("", "", "empty string"),
        ("A", "1A", "single character"),
        ("ABC", "1A1B1C", "non-repeating characters"),
        (
            "WWWWWWWWWWWWBWWWWWWWWWWWWBBBWWWWWWWWWWWWWWWWWWWWWWWWB",
            "12W1B12W3B24W1B",
            "multi-digit run counts",
        ),
    ]

    stdin_payload = json.dumps({"code": code, "test_cases": test_vectors})
    clean_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}

    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", _SANDBOX_RUNNER_SCRIPT],
            input=stdin_payload,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=tempfile.gettempdir(),
            env=clean_env,
        )
    except subprocess.TimeoutExpired:
        return EvaluationOutcome(
            passed=False,
            message=f"Failed: Code execution timed out after {timeout_seconds}s.",
            checks=[{"name": "execution_timeout", "passed": False, "detail": "Subprocess timed out"}],
            extra={"extracted_code": code},
        )
    except Exception as e:
        return EvaluationOutcome(
            passed=False,
            message=f"Failed: Subprocess execution error: {e}",
            checks=[{"name": "subprocess_exec", "passed": False, "detail": str(e)}],
            extra={"extracted_code": code},
        )

    if proc.returncode != 0 or not proc.stdout.strip():
        stderr_msg = (proc.stderr or "").strip()
        return EvaluationOutcome(
            passed=False,
            message=f"Failed: Sandbox exited with code {proc.returncode}: {stderr_msg}",
            checks=[{"name": "sandbox_exit", "passed": False, "detail": stderr_msg or "Non-zero exit"}],
            extra={"extracted_code": code},
        )

    try:
        sandbox_res = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as e:
        return EvaluationOutcome(
            passed=False,
            message=f"Failed to parse sandbox output: {e}",
            checks=[{"name": "sandbox_parse", "passed": False, "detail": proc.stdout[:200]}],
            extra={"extracted_code": code},
        )

    if sandbox_res.get("error"):
        return EvaluationOutcome(
            passed=False,
            message=f"Failed: {sandbox_res['error']}",
            checks=[{"name": "runtime_execution", "passed": False, "detail": sandbox_res["error"]}],
            extra={"extracted_code": code},
        )

    checks = sandbox_res.get("checks", [])
    passed_count = sum(1 for c in checks if c.get("passed"))
    total_count = len(checks)
    all_passed = bool(sandbox_res.get("passed")) and total_count > 0

    if all_passed:
        msg = f"Passed all {passed_count}/{total_count} executed unit test checks."
    else:
        msg = f"Failed coding checks: {passed_count}/{total_count} assertions passed."

    return EvaluationOutcome(
        passed=all_passed,
        message=msg,
        checks=checks,
        extra={
            "extracted_code": code,
            "passed_assertions": passed_count,
            "total_assertions": total_count,
        },
    )


# ===========================================================================
# 2. Tool-Call JSON Capability Evaluator
# ===========================================================================

def evaluate_tool_call_json(response_text: str, **_kwargs: Any) -> EvaluationOutcome:
    """Validate structured tool-call JSON syntax, prerequisite planning, and arguments."""
    data = extract_json_object(response_text)
    if data is None:
        return EvaluationOutcome(
            passed=False,
            message="Failed: Output is not valid JSON or could not be parsed as a JSON object.",
            checks=[{"name": "valid_json", "passed": False, "detail": "JSON parsing failed"}],
        )

    checks: list[dict[str, Any]] = [
        {"name": "valid_json", "passed": True, "detail": "Parsed JSON object successfully"}
    ]

    thought = data.get("thought") or data.get("reasoning")
    thought_ok = isinstance(thought, str) and len(thought.strip()) > 3
    checks.append({
        "name": "thought_present",
        "passed": thought_ok,
        "detail": f"thought={thought!r}" if thought_ok else "Missing or empty 'thought' string",
    })

    tool_name = data.get("tool_to_call") or data.get("tool") or data.get("name")
    tool_ok = tool_name == "get_weather"
    checks.append({
        "name": "correct_prerequisite_tool",
        "passed": tool_ok,
        "detail": (
            "Selected 'get_weather' as prerequisite step"
            if tool_ok
            else f"Expected tool 'get_weather', got {tool_name!r}"
        ),
    })

    arguments = data.get("arguments") if "arguments" in data else data.get("parameters")
    args_dict_ok = isinstance(arguments, dict)
    city_val = arguments.get("city") if isinstance(arguments, dict) else None
    city_ok = isinstance(city_val, str) and city_val.strip().lower() == "chicago"

    checks.append({
        "name": "arguments_city",
        "passed": args_dict_ok and city_ok,
        "detail": (
            f"arguments.city={city_val!r}"
            if (args_dict_ok and city_ok)
            else f"Expected arguments={{'city': 'Chicago'}}, got {arguments!r}"
        ),
    })

    all_passed = all(c["passed"] for c in checks)
    if all_passed:
        msg = "Passed tool-call JSON schema, prerequisite tool selection, and argument binding."
    else:
        failed_names = [c["name"] for c in checks if not c["passed"]]
        msg = f"Failed tool-call JSON validation ({', '.join(failed_names)})."

    return EvaluationOutcome(
        passed=all_passed,
        message=msg,
        checks=checks,
        extra={"parsed_json": data},
    )


# ===========================================================================
# 3. Constraint Following Capability Evaluator
# ===========================================================================

def evaluate_constraint_following(response_text: str, **_kwargs: Any) -> EvaluationOutcome:
    """Validate strict line count and starting-letter constraints (4-line A-poem)."""
    lines = [line.strip() for line in response_text.strip().splitlines() if line.strip()]
    checks: list[dict[str, Any]] = []

    line_count_ok = len(lines) == 4
    checks.append({
        "name": "exact_4_lines",
        "passed": line_count_ok,
        "detail": f"Found {len(lines)} non-empty lines (expected 4)",
    })

    all_start_with_a = len(lines) > 0
    for idx, line in enumerate(lines):
        cleaned = re.sub(r"^[^\w]+", "", line)
        starts_a = bool(cleaned) and cleaned[0].lower() == "a"
        if not starts_a:
            all_start_with_a = False
        checks.append({
            "name": f"line_{idx + 1}_starts_with_a",
            "passed": starts_a,
            "detail": f"Line {idx + 1}: {line[:40]!r}",
        })

    all_passed = line_count_ok and all_start_with_a
    if all_passed:
        msg = "Passed all constraint checks: exactly 4 lines, each starting with 'A'/'a'."
    else:
        failed = [c["detail"] for c in checks if not c["passed"]]
        msg = f"Failed constraint checks: {'; '.join(failed)}"

    return EvaluationOutcome(
        passed=all_passed,
        message=msg,
        checks=checks,
        extra={"line_count": len(lines), "lines": lines},
    )


# ===========================================================================
# 4. Reasoning Capability Evaluator
# ===========================================================================

def evaluate_reasoning(response_text: str, **_kwargs: Any) -> EvaluationOutcome:
    """Validate step-by-step river crossing logic puzzle and terminal 'DONE' marker."""
    cleaned = response_text.strip()
    lower = cleaned.lower()
    checks: list[dict[str, Any]] = []

    # Strip trailing punctuation when checking for DONE suffix
    trimmed_end = re.sub(r"[\s.!\"'`*]+$", "", cleaned)
    ends_with_done = trimmed_end.upper().endswith("DONE")
    checks.append({
        "name": "ends_with_done",
        "passed": ends_with_done,
        "detail": (
            "Response ends with 'DONE'"
            if ends_with_done
            else f"Response did not end with 'DONE' (tail: {cleaned[-25:]!r})"
        ),
    })

    has_chicken = any(w in lower for w in ("chicken", "hen", "goose", "duck"))
    has_fox = any(w in lower for w in ("fox", "wolf"))
    has_grain = any(w in lower for w in ("grain", "corn", "wheat", "sack", "seeds", "beans"))
    entities_ok = has_chicken and has_fox and has_grain
    checks.append({
        "name": "puzzle_entities_present",
        "passed": entities_ok,
        "detail": f"chicken={has_chicken}, fox={has_fox}, grain={has_grain}",
    })

    # Verify chicken is mentioned multiple times (initial crossing + return/final crossing)
    chicken_mentions = sum(len(re.findall(rf"\b{w}\b", lower)) for w in ("chicken", "hen", "goose", "duck"))
    multi_step_ok = chicken_mentions >= 2
    checks.append({
        "name": "multi_step_crossing_logic",
        "passed": multi_step_ok,
        "detail": f"Key item mentioned {chicken_mentions} times across crossing steps",
    })

    all_passed = all(c["passed"] for c in checks)
    if all_passed:
        msg = "Passed step-by-step reasoning checks and terminal 'DONE' constraint."
    else:
        failed_names = [c["name"] for c in checks if not c["passed"]]
        msg = f"Failed reasoning checks: {', '.join(failed_names)}."

    return EvaluationOutcome(
        passed=all_passed,
        message=msg,
        checks=checks,
    )


# ===========================================================================
# 5. Summarization Capability Evaluator
# ===========================================================================

def evaluate_summarization(response_text: str, **_kwargs: Any) -> EvaluationOutcome:
    """Validate single-sentence summary under 20 words with key concept retention."""
    cleaned = response_text.strip().strip("\"'")
    words = [w for w in cleaned.split() if w]
    word_count = len(words)

    checks: list[dict[str, Any]] = []

    length_ok = 3 <= word_count <= 20
    checks.append({
        "name": "word_count_limit",
        "passed": length_ok,
        "detail": f"Summary has {word_count} words (required: 3..20 words)",
    })

    # Count sentence-ending punctuation
    terminal_marks = cleaned.count(".") + cleaned.count("!") + cleaned.count("?")
    single_sentence_ok = terminal_marks <= 1 and "\n" not in cleaned
    checks.append({
        "name": "single_sentence",
        "passed": single_sentence_ok,
        "detail": f"Detected {terminal_marks} sentence terminal mark(s)",
    })

    key_terms = {
        "ai", "artificial", "intelligence", "machines", "machine",
        "agents", "agent", "environment", "goals", "goal", "cognitive", "learning",
    }
    lower_words = {re.sub(r"[^\w]", "", w.lower()) for w in words}
    matched_terms = sorted(lower_words.intersection(key_terms))
    concept_ok = len(matched_terms) >= 2
    checks.append({
        "name": "key_concepts_covered",
        "passed": concept_ok,
        "detail": f"Matched key concepts: {matched_terms}",
    })

    all_passed = all(c["passed"] for c in checks)
    if all_passed:
        msg = f"Passed summarization checks ({word_count} words, single sentence, concepts={matched_terms})."
    else:
        failed_names = [c["name"] for c in checks if not c["passed"]]
        msg = f"Failed summarization checks ({', '.join(failed_names)})."

    return EvaluationOutcome(
        passed=all_passed,
        message=msg,
        checks=checks,
        extra={"word_count": word_count, "matched_concepts": matched_terms},
    )


# ===========================================================================
# Built-in Capability Cases Catalog
# ===========================================================================

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise, helpful instruction-following AI assistant. "
    "Follow all formatting and output constraints strictly."
)

BUILTIN_CAPABILITY_CASES: list[CapabilityCase] = [
    CapabilityCase(
        id="coding",
        suite="coding",
        name="Coding & Execution (Run-Length Encoder/Decoder)",
        category="coding",
        description=(
            "Generate a working Python module with rle_encode(s) and rle_decode(s), "
            "verified by executing 5 algorithmic unit test vectors in a sandbox."
        ),
        prompt=(
            "Write a Python module containing two functions:\n"
            "1. `rle_encode(s: str) -> str`: takes a string and returns its run-length encoded "
            "version (e.g. 'AAAAABBBCC' -> '5A3B2C'). Return '' for an empty string.\n"
            "2. `rle_decode(s: str) -> str`: takes a run-length encoded string and returns the "
            "original decoded string (e.g. '5A3B2C' -> 'AAAAABBBCC').\n\n"
            "Return ONLY valid Python code inside a ```python ``` block."
        ),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        evaluator=evaluate_coding,
        sample_passing_response=(
            "```python\n"
            "def rle_encode(s: str) -> str:\n"
            "    if not s:\n"
            "        return ''\n"
            "    out = []\n"
            "    curr = s[0]\n"
            "    count = 1\n"
            "    for ch in s[1:]:\n"
            "        if ch == curr:\n"
            "            count += 1\n"
            "        else:\n"
            "            out.append(f'{count}{curr}')\n"
            "            curr = ch\n"
            "            count = 1\n"
            "    out.append(f'{count}{curr}')\n"
            "    return ''.join(out)\n\n"
            "def rle_decode(s: str) -> str:\n"
            "    if not s:\n"
            "        return ''\n"
            "    out = []\n"
            "    digits = []\n"
            "    for ch in s:\n"
            "        if ch.isdigit():\n"
            "            digits.append(ch)\n"
            "        else:\n"
            "            out.append(ch * int(''.join(digits)))\n"
            "            digits = []\n"
            "    return ''.join(out)\n"
            "```"
        ),
    ),
    CapabilityCase(
        id="tool_call_json",
        suite="tool_call",
        name="Tool-Call JSON (Conditional Prerequisite Planning)",
        category="tool_call",
        description=(
            "Select the prerequisite tool call and output a valid JSON object matching "
            "the tool schema and argument types."
        ),
        prompt=(
            "You are an autonomous smart-home agent with access to three tools:\n"
            "1. `get_weather(city: str) -> str`: Returns weather and temperature forecast.\n"
            "2. `set_thermostat(temp: int) -> str`: Sets the home thermostat temperature.\n"
            "3. `broadcast_alert(message: str) -> str`: Sends a home notification.\n\n"
            "User request: 'It is freezing in Chicago today! If the temperature forecast is "
            "below 32°F, set my thermostat to 72°F, otherwise set it to 68°F. Make sure to "
            "check the weather forecast first.'\n\n"
            "Output ONLY a JSON object representing the FIRST tool call to execute, using "
            "this schema:\n"
            "{\n"
            '  "thought": "brief reasoning for choosing this first step",\n'
            '  "tool_to_call": "tool_name",\n'
            '  "arguments": {"city": "city_name"}\n'
            "}"
        ),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        evaluator=evaluate_tool_call_json,
        sample_passing_response=(
            "```json\n"
            "{\n"
            '  "thought": "The thermostat target depends on the Chicago temperature forecast, so I must call get_weather for Chicago first.",\n'
            '  "tool_to_call": "get_weather",\n'
            '  "arguments": {\n'
            '    "city": "Chicago"\n'
            "  }\n"
            "}\n"
            "```"
        ),
    ),
    CapabilityCase(
        id="constraint_following",
        suite="constraints",
        name="Constraint Following (4-Line Alliterative A-Poem)",
        category="constraints",
        description=(
            "Write a 4-line poem about gravity where every single line starts with "
            "the letter 'A' or 'a' and contains no extra commentary."
        ),
        prompt=(
            "Write a short 4-line poem about gravity. Every line must start with the letter "
            "'A' or 'a'. Do not include any title, introduction, or concluding remarks—output "
            "only the 4 lines."
        ),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        evaluator=evaluate_constraint_following,
        sample_passing_response=(
            "All masses pull each other near,\n"
            "Across the vast and silent night,\n"
            "Attracting worlds from sphere to sphere,\n"
            "Anchoring stars in steady flight."
        ),
    ),
    CapabilityCase(
        id="reasoning",
        suite="reasoning",
        name="Reasoning & Logic (River Crossing Riddle)",
        category="reasoning",
        description=(
            "Solve the classic farmer, fox, chicken, and grain river-crossing riddle "
            "step-by-step and end with the exact word DONE."
        ),
        prompt=(
            "A farmer has a chicken, a fox, and a sack of grain. He must cross a river in a "
            "boat that can only hold himself and one item at a time. If left unattended together, "
            "the fox will eat the chicken, and the chicken will eat the grain. Explain step-by-step "
            "how the farmer can get everything across safely, and end your response with the exact "
            "word 'DONE'."
        ),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        evaluator=evaluate_reasoning,
        sample_passing_response=(
            "1. The farmer takes the chicken across the river first and leaves it on the far bank.\n"
            "2. The farmer returns alone to the starting bank.\n"
            "3. The farmer takes the fox across the river to the far bank.\n"
            "4. The farmer brings the chicken back to the starting bank so the fox does not eat it.\n"
            "5. The farmer leaves the chicken and takes the sack of grain across to the far bank with the fox.\n"
            "6. The farmer returns alone to the starting bank.\n"
            "7. Finally, the farmer takes the chicken across the river to the far bank.\n"
            "DONE"
        ),
    ),
    CapabilityCase(
        id="summarization",
        suite="summarization",
        name="Summarization (Single-Sentence <=20 Words)",
        category="summarization",
        description=(
            "Summarize a technical AI passage in a single sentence of at most 20 words "
            "while preserving key concepts."
        ),
        prompt=(
            "Summarize the following text in a single sentence of 20 words or fewer:\n"
            "'Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed "
            "to natural intelligence displayed by humans or animals. Leading AI textbooks define "
            "the field as the study of intelligent agents: any system that perceives its "
            "environment and takes actions that maximize its chance of achieving its goals.'\n"
            "Output ONLY the single summary sentence."
        ),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        evaluator=evaluate_summarization,
        sample_passing_response=(
            "Artificial intelligence studies machine agents that perceive their environment "
            "and act to achieve specific goals."
        ),
    ),
]


def get_builtin_cases(selected_suites: list[str] | None = None) -> list[CapabilityCase]:
    """Return built-in capability cases filtered by selected suites."""
    if not selected_suites:
        return list(BUILTIN_CAPABILITY_CASES)
    suite_set = set(selected_suites)
    return [c for c in BUILTIN_CAPABILITY_CASES if c.suite in suite_set or c.id in suite_set]


# ===========================================================================
# LLM-as-a-Judge Prompt & Parser
# ===========================================================================

def build_judge_messages(
    case: CapabilityCase,
    response_text: str,
    local_passed: bool,
    local_message: str,
    min_score: int = 3,
) -> list[dict[str, str]]:
    """Construct system and user messages for the secondary LLM-as-a-Judge target."""
    system_msg = (
        "You are an objective, analytical evaluation judge assessing LLM responses against "
        "task instructions, accuracy, and constraints. You always respond strictly with a "
        "valid JSON object."
    )
    user_msg = (
        f"Evaluate the following model response for test case '{case.name}' ({case.category}).\n\n"
        f"--- TASK PROMPT ---\n{case.prompt}\n\n"
        f"--- AUTOMATED CHECK RESULT ---\nPassed={local_passed}: {local_message}\n\n"
        f"--- MODEL RESPONSE ---\n{response_text}\n\n"
        "Rate the response on an integer scale from 1 (completely failed/inaccurate) to 5 "
        "(excellent adherence, accuracy, and clarity).\n"
        "Return ONLY a JSON object with this exact schema:\n"
        "{\n"
        '  "score": <integer 1 to 5>,\n'
        f'  "passed": <boolean, true if score >= {min_score} and task requirements are met>,\n'
        '  "justification": "<concise 1-3 sentence explanation of the rating>"\n'
        "}"
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def parse_judge_response(raw_text: str, min_score: int = 3) -> dict[str, Any]:
    """Parse and sanitize the JSON response from the LLM Judge."""
    data = extract_json_object(raw_text)
    if not data:
        return {
            "score": None,
            "passed": None,
            "justification": None,
            "error": "Could not parse valid JSON from judge response",
        }

    score = coerce_judge_score(data.get("score"))
    justification = str(data.get("justification") or "").strip()

    if score is None:
        return {
            "score": None,
            "passed": None,
            "justification": justification or None,
            "error": f"Invalid or out-of-range judge score: {data.get('score')!r}",
        }

    raw_passed = data.get("passed")
    if isinstance(raw_passed, bool):
        passed = raw_passed and (score >= min_score)
    else:
        passed = score >= min_score

    return {
        "score": score,
        "passed": passed,
        "justification": justification,
        "error": None,
    }
