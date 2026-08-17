"""
An interactive chat with one tutor, for trying a checkpoint out before running the eval.

Loads a single tutor and lets you play the student in the terminal, through the same session
machinery the harness uses, so the prompts, action tags, tool calls and greedy decoding all
match. It prints the tutor's action tag and any tool calls, so you can see what the model is
actually doing.

    uv run eval-chat --model Paulo-Oliva/its-t2-sft --mode plain
    uv run eval-chat --model Paulo-Oliva/its-t4-grpo --mode tools

Run `uv run eval-chat --help` for the options.
"""

import argparse
import logging
import random
import re
import sys
from dataclasses import dataclass, fields
from pathlib import Path

from its.config import TUTOR_ACTIONS
from its.evaluation.common import (UNIFIED_TUTOR_SYSTEM, load_chat_model, read_jsonl, student_view)
from its.protocol import ALL_TUTOR_ACTIONS, detect_leakage, verify_answer
from its.evaluation.harness import (DEFAULT_BANK, DEFAULT_KB, build_eval_tools, generate_tutor_turn,
                                    tutor_system_prompt)

log = logging.getLogger(__name__)


# Colours for the terminal
@dataclass(frozen=True)
class Palette:
    TUTOR: str = "\033[36m"
    STUDENT: str = "\033[32m"
    ACTION: str = "\033[35m"
    TOOL: str = "\033[33m"
    DIM: str = "\033[2m"
    BOLD: str = "\033[1m"
    OFF: str = "\033[0m"


def _paint(enabled: bool) -> Palette:
    return Palette() if enabled else Palette(*[""] * len(fields(Palette)))


def pick_problem(args, rng: random.Random) -> tuple[str, str]:
    if args.problem:
        return args.problem, args.answer or ""
    if args.eval_set and Path(args.eval_set).exists():
        rows = read_jsonl(Path(args.eval_set))
        row = rows[args.index] if args.index is not None else rng.choice(rows)
        return row["problem"], str(row["answer"])
    if Path(args.bank).exists():
        rows = read_jsonl(Path(args.bank))
        row = rng.choice(rows)
        return row["problem"], str(row["answer"])
    raise SystemExit("No problem source: pass --problem/--answer or --eval-set or a valid --bank")


# Display only, dropping a line that is nothing but an invented bracket tag
_JUNK_TAG_LINE = re.compile(r"^\s*\[\s*[A-Za-z][\w ]{0,30}\]?\s*(?:\(.{0,40}\))?\s*$")


def _drop_junk_tag_lines(text: str) -> str:
    lines = [l for l in text.splitlines() if not _JUNK_TAG_LINE.match(l)]
    return "\n".join(lines).strip()


# Prints the tutor and tool entries added this turn, with `raw` dumping the output verbatim
def _print_tutor_turn(transcript: list[dict], since: int, paint, raw: bool = False) -> None:
    from its.protocol import strip_leading_tag, strip_tool_calls
    for t in transcript[since:]:
        if t["role"] == "tool":
            mark = "ok" if t.get("ok", True) else "FAILED"
            print(f"{paint.TOOL}   ⚙ {t['name']} [{mark}] → {t['content']}{paint.OFF}")
        elif t["role"] == "tutor":
            if raw:
                print(f"{paint.TUTOR}{paint.BOLD}Tutor (raw):{paint.OFF} "
                      f"{paint.TUTOR}{t['content']}{paint.OFF}")
                continue
            action = t.get("action")
            body = _drop_junk_tag_lines(strip_tool_calls(strip_leading_tag(t["content"])))
            has_call = "<tool_call>" in t["content"]
            tag = f"{paint.ACTION}[{action}]{paint.OFF} " if action else ""
            if has_call and not body:
                # A pure tool-call segment (no student-facing prose) - already shown as the tool line
                continue
            print(f"{paint.TUTOR}{paint.BOLD}Tutor:{paint.OFF} {tag}{paint.TUTOR}{body}{paint.OFF}")


def chat_loop(
    tutor,
    mode: str,
    tools,
    problem: str,
    answer: str,
    max_turns: int,
    max_tool_calls: int,
    paint,
    free: bool = False,
    all_scenarios: bool = False,
) -> str:
    if all_scenarios:
        # The full menu, including [explain] and [present_problem]
        actions = ALL_TUTOR_ACTIONS
        # One prompt covering solve, check, request and explain
        base = UNIFIED_TUTOR_SYSTEM
        # The gold answer is withheld, so the tutor has to actually use the tools
        tutor_answer = ""
    else:
        actions = TUTOR_ACTIONS if mode in ("actions", "tools") else None
        base = None
        tutor_answer = answer
    schemas = tools.schemas if (mode == "tools" and tools) else None

    tutor_msgs = [{"role": "system", "content": tutor_system_prompt(tutor_answer, actions, schemas, base=base)}]
    transcript: list[dict] = []
    first = True
    turns = 0
    pending_force: str | None = None

    print(f"\n{paint.BOLD}{'═' * 70}{paint.OFF}")
    print(f"{paint.BOLD}PROBLEM:{paint.OFF} {problem}")
    print(f"{paint.DIM}(gold answer hidden from you; the tutor has it. /answer to reveal){paint.OFF}")
    print(f"{paint.BOLD}{'═' * 70}{paint.OFF}")
    print(f"{paint.DIM}You are the student. Type your message, or /help for commands.{paint.OFF}\n")

    while True:
        try:
            user = input(f"{paint.STUDENT}{paint.BOLD}You:{paint.OFF} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return "quit"
        if not user:
            continue
        low = user.lower()
        if low in ("/quit", "/q", "/exit"):
            return "quit"
        if low in ("/new", "/n"):
            return "new"
        if low in ("/reset", "/r"):
            print(f"{paint.DIM}- restarting this problem -{paint.OFF}")
            return "reset"
        if low == "/problem":
            print(f"{paint.BOLD}PROBLEM:{paint.OFF} {problem}")
            continue
        if low == "/answer":
            print(f"{paint.BOLD}GOLD ANSWER:{paint.OFF} {answer}")
            continue
        if low == "/help":
            print(f"{paint.DIM}/new new problem · /reset restart · /problem show · "
                  f"/answer reveal · /force <verifier|planner|retriever> seed a tool call "
                  f"on the next turn · /quit exit{paint.OFF}")
            continue
        if low.startswith("/force"):
            name = low.removeprefix("/force").strip().lstrip("_")
            tool_name = {
                "verifier": "call_verifier",
                "planner": "call_planner",
                "retriever": "call_retriever"
            }.get(name)
            if not (tool_name and schemas):
                print(f"{paint.DIM}usage: /force verifier|planner|retriever (needs --mode tools){paint.OFF}")
                continue
            pending_force = tool_name
            print(f"{paint.DIM}- next tutor turn will open a {tool_name} call "
                  f"(training-style scaffold); type your message -{paint.OFF}")
            continue

        transcript.append({"role": "student", "content": user})
        if first:
            tutor_msgs.append({"role": "user", "content": f"Problem: {problem}\n\nStudent: {user}"})
            first = False
        else:
            tutor_msgs.append({"role": "user", "content": user})

        if verify_answer(user, answer):
            print(f"{paint.DIM}✓ (your message states the correct answer){paint.OFF}")

        since = len(transcript)
        print(f"{paint.DIM}… thinking …{paint.OFF}", end="\r")
        generate_tutor_turn(tutor,
                            tutor_msgs,
                            transcript,
                            actions,
                            schemas,
                            tools,
                            problem,
                            answer,
                            "medium",
                            max_tool_calls,
                            constrain=not free,
                            force_tool=pending_force)
        pending_force = None
        print(" " * 20, end="\r")
        _print_tutor_turn(transcript, since, paint, raw=free)

        # Non-intrusive leakage flag over what you (the student) have seen so far
        if detect_leakage(student_view(transcript), answer):
            print(f"{paint.TOOL}⚠ the tutor appears to have revealed the answer{paint.OFF}")

        turns += 1
        if turns >= max_turns:
            print(f"{paint.DIM}- reached max {max_turns} turns; /new or /reset -{paint.OFF}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s  %(levelname)-8s  %(message)s")
    p = argparse.ArgumentParser(description="Interactive chat with a tutor baseline")
    p.add_argument("--model", required=True, help="Tutor: HF repo / local dir / LoRA adapter")
    p.add_argument("--mode",
                   required=True,
                   choices=["plain", "actions", "tools"],
                   help="plain (T1/T2), actions (T3), tools (T4)")
    p.add_argument("--free",
                   action="store_true",
                   help="Don't force/strip the action tag: keep the action menu in the prompt "
                   "but let the model generate the whole turn freely, and print it raw "
                   "(diagnostic - see how the trained model behaves unconstrained). "
                   "No effect in plain mode, which is already free-form.")
    p.add_argument("--all-scenarios",
                   action="store_true",
                   help="Use ONE unified system prompt + the full action menu so the tutor can "
                   "handle any request (solve / check work / request problem / explain concept) "
                   "and call the matching tool. Needs --mode tools. Withholds the gold answer "
                   "from the tutor so it must actually use the tools. NOTE: T4 was trained on "
                   "the scenarios separately, so this combined prompt is out-of-distribution.")
    p.add_argument("--base-model", default=None, help="Base for a LoRA (default: adapter config)")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--max-tool-calls", type=int, default=2)
    p.add_argument("--temperature",
                   type=float,
                   default=0.7,
                   help="Sampling temperature (0 = greedy). Default 0.7 adds variety so the "
                   "tutor doesn't repeat the same turn verbatim.")
    p.add_argument("--top-p",
                   type=float,
                   default=0.9,
                   help="Nucleus sampling: trims the junk-token tail (default 0.9; 1.0 = off, "
                   "which makes Qwen-Math degenerate into multilingual gibberish).")
    p.add_argument("--repetition-penalty",
                   type=float,
                   default=1.05,
                   help="Penalise tokens already generated (>1). Keep mild (default 1.05) - "
                   "high values suppress normal math tokens and cause degeneration.")
    p.add_argument("--no-repeat-ngram",
                   type=int,
                   default=4,
                   help="Hard-block repeating any n-gram of this size (0 = off). Default 4 "
                   "kills verbatim-repeated turns without banning common math trigrams.")
    # Problem source
    p.add_argument("--problem", default=None, help="Problem text (with --answer)")
    p.add_argument("--answer", default=None, help="Gold answer for --problem")
    p.add_argument("--eval-set",
                   default="data/eval/problems_eval.jsonl",
                   help="Draw problems from this JSONL (random unless --index)")
    p.add_argument("--index", type=int, default=None, help="Pick eval-set row by index")
    p.add_argument("--bank", type=Path, default=DEFAULT_BANK, help="Fallback problem source")
    p.add_argument("--seed", type=int, default=None, help="Problem-pick RNG seed")
    # tools-mode helpers
    p.add_argument("--verifier-model",
                   default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="Small model backing the Verifier tool (tools mode)")
    p.add_argument("--verifier-device", default="auto")
    p.add_argument("--kb", type=Path, default=DEFAULT_KB, help="Retriever KB JSONL (tools mode)")
    p.add_argument("--no-color", action="store_true")
    args = p.parse_args()

    paint = _paint(not args.no_color and sys.stdout.isatty())
    rng = random.Random(args.seed)

    if args.all_scenarios and args.mode != "tools":
        log.warning("--all-scenarios needs --mode tools for check/request/explain to call tools; "
                    "without it the tutor can only route in prose.")

    print(f"Loading tutor {args.model} [{args.mode}] …")
    tutor = load_chat_model(args.model,
                            base_model=args.base_model,
                            device=args.device,
                            max_new_tokens=args.max_new_tokens)
    # Sampling with mild penalties, since pure greedy loops on a collapsed policy
    tutor.do_sample = args.temperature > 0
    tutor.temperature = args.temperature if args.temperature > 0 else 1.0
    tutor.top_p = args.top_p
    tutor.repetition_penalty = args.repetition_penalty
    tutor.no_repeat_ngram_size = args.no_repeat_ngram

    tools = None
    if args.mode == "tools":
        print(f"Loading Verifier backing model {args.verifier_model} …")
        verifier = load_chat_model(args.verifier_model, device=args.verifier_device, max_new_tokens=args.max_new_tokens)
        tools = build_eval_tools(verifier, args.bank, args.kb, seed=0)

    control = "new"
    problem = answer = ""
    while control != "quit":
        if control in ("new", None):
            problem, answer = pick_problem(args, rng)
        control = chat_loop(tutor,
                            args.mode,
                            tools,
                            problem,
                            answer,
                            args.max_turns,
                            args.max_tool_calls,
                            paint,
                            free=args.free,
                            all_scenarios=args.all_scenarios)
    print("bye.")


if __name__ == "__main__":
    main()
