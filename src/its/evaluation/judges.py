"""
The LLM judge panel for the cross-baseline evaluation.

Scores the same saved transcripts with one judge per invocation, and `eval-report` then
computes the agreement between them. A judge only ever scores and never generates dialogue,
and it is blinded: the dialogue it reads carries no label, no action tags and no tool traffic,
which is exactly the student's view, so it cannot tell which system produced it.

    uv run eval-judge --judge hf:Qwen/Qwen2.5-7B-Instruct
    uv run eval-judge --judge openai:gpt-4o

The panel needs at least one judge from a non-Qwen family, since the tutor is Qwen and would
otherwise be judged by its own kind, and every judge has to be held out from training. T3 and
T4 were trained against a Qwen-0.5B judge, so that one can never be a panel member.

Run `uv run eval-judge --help` for the options.
"""

import argparse
import json
import logging
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from its._jsonparse import lenient_json
from its.evaluation.common import (append_jsonl, read_jsonl, render_dialogue, student_view)

log = logging.getLogger(__name__)

DEFAULT_SESSIONS_DIR = Path("data/eval/sessions")
DEFAULT_OUT_DIR = Path("data/eval/scores")

# The rubric dimensions, the two binary cross-checks, and the degeneration flag
CONTINUOUS_DIMS = ("delta_mastery", "progress", "overhelp", "socratic")
BINARY_DIMS = ("student_solved", "tutor_revealed", "incoherent")

# Never mentions action tags, so T1 and T2 transcripts stay indistinguishable
JUDGE_SYSTEM = """\
You are a STRICT, critical evaluator of one-on-one math tutoring. Most tutoring is mediocre - \
keep scores low by default and reserve the top of each scale for genuinely excellent work. \
Judge the TUTOR's contribution, NOT the student's innate ability: if the student basically \
solved it themselves, the tutor deserves little credit no matter how the dialogue looks. \
When uncertain, err toward the lower score.

You will be shown a math problem, its correct answer, and a tutoring dialogue.

Reply with ONLY a JSON object, no other text, with exactly these keys IN THIS ORDER:
{"reasoning": "<1-3 sentences justifying your scores, no curly braces>", \
"delta_mastery": <float>, "progress": <float>, "overhelp": <float>, "socratic": <float>, \
"student_solved": <0 or 1>, "tutor_revealed": <0 or 1>, "incoherent": <0 or 1>}

Write the reasoning FIRST - analyse the dialogue before committing to numbers.

Some dialogues end abruptly mid-conversation (a system limit, not the tutor's choice). \
Score only the turns shown - do not penalise the tutor for the cut-off itself, but do not \
credit outcomes that never happened either (e.g. student_solved stays 0 if the student never \
stated the answer).

The anchors below are REFERENCE POINTS, not the only allowed values - use any value in the \
range (e.g. 0.3, 0.7) when a session falls between anchors, and when torn between two readings \
of the dialogue, lean toward the lower value:

delta_mastery - how much the student's OWN understanding grew BECAUSE OF the tutoring:
  -1.0 = the tutor confused the student or led them toward a wrong method
   0.0 = the student already knew how - the tutor taught nothing new
   0.5 = the student gained partial new understanding with the tutor's help
   1.0 = a genuinely stuck or confused student reached real understanding through the tutor's guidance

progress - fraction of tutor turns that ACTUALLY moved the student forward because of the tutor \
(a turn that only validates or cheers - "Good, keep going!" - while the student does the real work \
does NOT count):
   0.0 = no turn produced a real step
   0.5 = about half the turns produced a step the student would not have taken alone
   1.0 = every tutor turn produced such a step

overhelp - how much the tutor did the work FOR the student:
   0.0 = always elicited the student's own reasoning
   0.5 = gave away some key steps
   1.0 = essentially solved the problem for them

socratic - how consistently the tutor taught through questions and elicitation rather than \
lecturing: acknowledging the student's attempt, probing their reasoning, and hinting minimally \
only when they are stuck:
   0.0 = no Socratic behaviour (lectures, dumps explanations, or responds off-topic)
   0.5 = mixed: some elicitation but frequent lecturing or unprompted explanation
   1.0 = consistently Socratic in nearly every turn

student_solved (0 or 1): 1 ONLY if the STUDENT themselves clearly stated the correct final answer \
by the end. An answer the TUTOR supplied, a muddled near-miss, or the wrong quantity does NOT count. \
When unsure, answer 0.

tutor_revealed (0 or 1): 1 if the TUTOR gave away the correct final answer (or all but the last \
trivial step) BEFORE the student had reached it. Confirming an answer the student already stated is \
NOT revealing. When unsure, answer 0.

incoherent (0 or 1): 1 if any tutor turn is degenerate: near-verbatim repetition of an earlier \
tutor turn, garbled or nonsensical text, mixing in other languages, or an empty/abruptly cut-off \
message. Normal imperfect tutoring is NOT incoherent - this flags only clearly broken output. \
When unsure, answer 0.\
"""


# The brace-balanced span starting at `start`, quote-aware so JSON strings do not count
def _balanced_span(text: str, start: int) -> str:
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    # Unterminated, because the reply was truncated, so lenient_json gets a try anyway
    return text[start:]


# The rubric dict wherever it sits, since a judge may write prose with braces before the JSON
def _rubric_object(text: str) -> dict | None:
    pos = text.find("{")
    while pos != -1:
        d = lenient_json(_balanced_span(text, pos))
        if isinstance(d, dict) and any(k in d for k in (*CONTINUOUS_DIMS, *BINARY_DIMS)):
            return d
        pos = text.find("{", pos + 1)
    return None


# The clamped rubric dict from a judge reply, or None when it is unusable
def parse_scores(text: str) -> dict | None:
    raw = _rubric_object(text)
    if raw is None:
        return None

    def clamp(key, lo, hi):
        try:
            return min(hi, max(lo, float(raw.get(key, 0.0))))
        except (TypeError, ValueError):
            return 0.0

    out: dict[str, Any] = {"delta_mastery": clamp("delta_mastery", -1.0, 1.0)}
    for k in ("progress", "overhelp", "socratic", "student_solved", "tutor_revealed", "incoherent"):
        out[k] = clamp(k, 0.0, 1.0)
    # Kept so a score can be audited afterwards
    out["reasoning"] = str(raw.get("reasoning", ""))[:500]
    return out


def judge_user_message(session: dict) -> str:
    view = student_view(session["transcript"])
    return (f"[Problem] {session['problem']}\n[Correct answer] {session['answer']}\n\n"
            f"[Dialogue]\n{render_dialogue(view)}")


# --- Judge backends --------------------------------------------------------------


class JudgeBackend(ABC):

    @abstractmethod
    def __call__(self, system: str, user: str) -> str:
        ...


class HFJudge(JudgeBackend):

    def __init__(self, model_id: str, device: str = "auto"):
        from its.evaluation.common import load_chat_model
        self.llm = load_chat_model(model_id, device=device, max_new_tokens=450)

    def __call__(self, system: str, user: str) -> str:
        return self.llm([{"role": "system", "content": system}, {"role": "user", "content": user}])


def _post_json(url: str, payload: dict, headers: dict, retries: int = 6) -> dict:
    body = json.dumps(payload).encode("utf-8")
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", **headers})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429 or e.code >= 500:
                try:
                    wait = int(float(e.headers.get("retry-after") or 0))
                except (TypeError, ValueError):
                    wait = 0
                if not wait:
                    wait = min(20 * (attempt + 1), 120) if e.code == 429 else 2**(attempt + 1)
                log.warning("HTTP %d - retry %d/%d in %ds", e.code, attempt + 1, retries, wait)
                time.sleep(wait)
            else:
                raise RuntimeError(f"API error {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = e
            wait = 2**(attempt + 1)
            log.warning("API call failed (%s) - retry %d/%d in %ds", e, attempt + 1, retries, wait)
            time.sleep(wait)
    raise RuntimeError(f"API call failed after {retries} retries: {last}")


class OpenAIJudge(JudgeBackend):

    def __init__(self, model: str, api_base: str, api_key: str):
        self.model, self.api_base, self.api_key = model, api_base.rstrip("/"), api_key

    def __call__(self, system: str, user: str) -> str:
        resp = _post_json(
            f"{self.api_base}/chat/completions", {
                "model": self.model,
                "temperature": 0,
                "max_tokens": 500,
                "messages": [{
                    "role": "system",
                    "content": system
                }, {
                    "role": "user",
                    "content": user
                }]
            }, {"Authorization": f"Bearer {self.api_key}"})
        return resp["choices"][0]["message"]["content"]


class JudgeRefusal(Exception):
    pass


class AnthropicJudge(JudgeBackend):

    def __init__(self, model: str, api_key: str):
        self.model, self.api_key = model, api_key

    def __call__(self, system: str, user: str) -> str:
        resp = _post_json(
            "https://api.anthropic.com/v1/messages", {
                "model": self.model,
                "max_tokens": 1000,
                "system": system,
                "thinking": {
                    "type": "disabled"
                },
                "messages": [{
                    "role": "user",
                    "content": user
                }]
            }, {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            })
        if resp.get("stop_reason") == "refusal":
            details = resp.get("stop_details") or {}
            raise JudgeRefusal(f"safety classifier declined (category={details.get('category')})")
        return "".join(b.get("text", "") for b in resp["content"])


def make_judge(spec: str, args) -> tuple[JudgeBackend, str]:
    import os
    backend, _, model = spec.partition(":")
    if not model:
        raise SystemExit(f"--judge must be backend:model (got {spec!r})")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", spec)
    if backend == "hf":
        return HFJudge(model, device=args.device), slug
    if backend == "openai":
        key = os.environ.get(args.api_key_env or "OPENAI_API_KEY", "")
        if not key:
            raise SystemExit(f"Set {args.api_key_env or 'OPENAI_API_KEY'} for the {spec} judge")
        return OpenAIJudge(model, args.api_base or "https://api.openai.com/v1", key), slug
    if backend == "anthropic":
        key = os.environ.get(args.api_key_env or "ANTHROPIC_API_KEY", "")
        if not key:
            raise SystemExit(f"Set {args.api_key_env or 'ANTHROPIC_API_KEY'} for the {spec} judge")
        return AnthropicJudge(model, key), slug
    raise SystemExit(f"Unknown judge backend {backend!r} (use hf: / openai: / anthropic:)")


# --- Entry point -----------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    # Keys may come from a repo-root .env, with real environment variables taking precedence
    from dotenv import load_dotenv
    load_dotenv()
    p = argparse.ArgumentParser(description="Score saved eval sessions with one LLM judge")
    p.add_argument("--judge",
                   required=True,
                   help="backend:model - hf:<id-or-path> | openai:<model> | anthropic:<model>")
    p.add_argument("--sessions",
                   type=Path,
                   default=DEFAULT_SESSIONS_DIR,
                   help="Sessions dir (all *.jsonl) or a single session file")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--api-base", default=None, help="OpenAI-compatible base URL override")
    p.add_argument("--api-key-env", default=None, help="Env var holding the API key")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-sessions", type=int, default=-1, help="Cap scored sessions (smoke tests)")
    args = p.parse_args()

    judge, slug = make_judge(args.judge, args)
    out_file = args.out_dir / f"{slug}.jsonl"
    # Resume skips scored and refused sessions, since a refusal is deterministic
    prev = read_jsonl(out_file)
    done = {(r["label"], r["session_id"]) for r in prev if r.get("ok") or r.get("refused")}
    n_refused = sum(1 for r in prev if r.get("refused"))
    n_failed = sum(1 for r in prev if not r.get("ok") and not r.get("refused"))
    if prev:
        log.info("Resuming: %d scored ok in %s%s%s",
                 len(done) - n_refused, out_file,
                 f" ({n_refused} refused by the provider - skipped)" if n_refused else "",
                 f" ({n_failed} previously failed - will retry)" if n_failed else "")

    files = sorted(args.sessions.glob("*.jsonl")) if args.sessions.is_dir() else [args.sessions]
    sessions, seen = [], set()
    for f in files:
        # Deduplicated across the shard files, where the first (label, id) seen wins
        for s in read_jsonl(f):
            key = (s["label"], s["session_id"])
            if key not in seen:
                seen.add(key)
                sessions.append(s)
    todo = [s for s in sessions if (s["label"], s["session_id"]) not in done]
    if args.max_sessions != -1:
        todo = todo[:args.max_sessions]
    log.info("Scoring %d sessions (of %d) with %s ...", len(todo), len(sessions), args.judge)

    n_ok = 0
    for i, s in enumerate(todo, 1):
        refused = False
        try:
            reply = judge(JUDGE_SYSTEM, judge_user_message(s))
            scores = parse_scores(reply)
        except JudgeRefusal as e:
            log.warning("Judge REFUSED %s/%s: %s", s["label"], s["session_id"], e)
            scores, refused = None, True
        except Exception as e:
            log.warning("Judge failed on %s/%s: %s", s["label"], s["session_id"], e)
            scores = None
        n_ok += scores is not None
        append_jsonl(
            out_file, {
                "judge": args.judge,
                "label": s["label"],
                "session_id": s["session_id"],
                "ok": scores is not None,
                "refused": refused,
                "scores": scores,
            })
        if i % 20 == 0 or i == len(todo):
            log.info("[%d/%d] scored (%d parsed ok)", i, len(todo), n_ok)

    log.info("Done: %d scored (%d ok) → %s", len(todo), n_ok, out_file)


if __name__ == "__main__":
    main()
