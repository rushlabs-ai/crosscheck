"""Describe a photo with several vision models and find where they agree.

1. Each model lists what it sees as short, separate facts (structured JSON output).
2. Facts that say the same thing are merged; facts that differ in a number or in
   quoted text are kept apart, because that is exactly where errors hide.
3. Every model judges every fact it did not state: yes, no, or unsure.
4. A fact is "agreed" when every model says yes, "disputed" when any says no,
   and "unconfirmed" otherwise.

Agreement is not proof: models can share a mistake. The wording never says
"verified" or "correct".
"""

from __future__ import annotations

import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from .ollama import Ollama, OllamaError

DESCRIBE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "facts": {"type": "array", "items": {"type": "string"}, "maxItems": 15},
    },
    "required": ["summary", "facts"],
}

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"n": {"type": "integer"}, "answer": {"type": "string", "enum": ["yes", "no", "unsure"]}},
                "required": ["n", "answer"],
            },
        }
    },
    "required": ["answers"],
}

DESCRIBE_PROMPT = """You are helping a blind person understand a photo they took.{question}
List what you can see as short, separate facts, one fact per item. Start with the most important:
what the main object or scene is, and any text you can read, quoted exactly. Include colours,
positions, quantities and anything that matters for safety. Only state what is visible. If
something is unclear, blurry or cut off, say so in the fact instead of guessing.
Also write a one-sentence summary."""

VERIFY_PROMPT = """Another assistant described this photo with the numbered statements below. Look at the
photo yourself and judge each statement. Answer "yes" if the photo clearly shows it, "no" if the
photo shows something different, and "unsure" if you cannot tell from the photo.

{statements}"""

_WORD = re.compile(r"\w+", re.UNICODE)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_QUOTED = re.compile(r"[\"“”'‘’«»]([^\"“”'‘’«»]{2,})[\"“”'‘’«»]")


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _f1(a: str, b: str) -> float:
    x, y = _WORD.findall(_norm(a)), _WORD.findall(_norm(b))
    if not x or not y:
        return 0.0
    common = sum((Counter(x) & Counter(y)).values())
    if not common:
        return 0.0
    p, r = common / len(y), common / len(x)
    return 2 * p * r / (p + r)


def _anchors(text: str) -> tuple[frozenset[str], frozenset[str]]:
    t = _norm(text)
    return frozenset(_NUMBER.findall(t)), frozenset(q.strip() for q in _QUOTED.findall(t))


def same_fact(a: str, b: str, threshold: float = 0.6) -> bool:
    """Near-duplicates only if numbers and quoted text match too."""
    return _anchors(a) == _anchors(b) and _f1(a, b) >= threshold


@dataclass
class Fact:
    text: str
    sources: list[str]  # models that stated it
    votes: dict[str, str] = field(default_factory=dict)  # model -> yes/no/unsure

    def status(self, models: list[str]) -> str:
        answers = [("yes" if m in self.sources else self.votes.get(m, "unsure")) for m in models]
        if "no" in answers:
            return "disputed"
        if all(a == "yes" for a in answers):
            return "agreed"
        return "unconfirmed"


@dataclass
class Result:
    models: list[str]
    summary: str
    spoken: str
    facts: list[Fact]
    descriptions: dict[str, dict[str, Any]]
    failed: dict[str, str]
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = {"agreed": [], "disputed": [], "unconfirmed": []}
        for f in self.facts:
            groups[f.status(self.models)].append({
                "text": f.text,
                "stated_by": f.sources,
                "judgements": {m: ("stated it" if m in f.sources else f.votes.get(m, "no answer")) for m in self.models},
            })
        return {"models": self.models, "summary": self.summary, "spoken": self.spoken, **groups,
                "descriptions": self.descriptions, "failed": self.failed, "seconds": self.seconds}


Progress = Callable[[str], None]


def _short(text: str, limit: int = 90) -> str:
    text = text.strip().rstrip(".")
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def spoken_summary(summary: str, facts: list[Fact], models: list[str], failed: dict[str, str]) -> str:
    agreed = [f for f in facts if f.status(models) == "agreed"]
    disputed = [f for f in facts if f.status(models) == "disputed"]
    parts = [summary.strip().rstrip(".") + "."] if summary.strip() else []
    if len(models) < 2:
        parts.append("Only one model answered, so nothing could be cross-checked.")
    else:
        parts.append(f"The {len(models)} models agree on {len(agreed)} of {len(facts)} details.")
        if disputed:
            listed = "; ".join(_short(f.text) for f in disputed[:3])
            more = f", and {len(disputed) - 3} more" if len(disputed) > 3 else ""
            parts.append(f"They disagree about: {listed}{more}.")
        else:
            parts.append("No model contradicted another.")
    if failed:
        parts.append(f"{len(failed)} model{'s' if len(failed) > 1 else ''} did not answer.")
    return " ".join(parts)


def run(client: Ollama, image: bytes, models: list[str], question: str = "",
        progress: Progress | None = None) -> Result:
    started = time.monotonic()
    say = progress or (lambda _msg: None)
    q = f"\nThe person asks: {question.strip()}\nAnswer that first, then describe the rest." if question.strip() else ""
    descriptions: dict[str, dict[str, Any]] = {}
    failed: dict[str, str] = {}

    for i, model in enumerate(models, start=1):
        say(f"Model {i} of {len(models)} is describing the photo.")
        try:
            out, timing = client.chat_json(model, DESCRIBE_PROMPT.format(question=q), image, DESCRIBE_SCHEMA)
        except OllamaError as e:
            failed[model] = str(e)
            continue
        facts = [str(x).strip() for x in (out.get("facts") or []) if str(x).strip()][:15]
        descriptions[model] = {"summary": str(out.get("summary") or "").strip(), "facts": facts, **timing}

    answered = [m for m in models if m in descriptions]
    merged: list[Fact] = []
    for model in answered:
        for text in descriptions[model]["facts"]:
            match = next((f for f in merged if same_fact(f.text, text)), None)
            if match:
                if model not in match.sources:
                    match.sources.append(model)
            else:
                merged.append(Fact(text=text, sources=[model]))

    if len(answered) > 1:
        # Judge in reverse order so the model loaded last goes first (fewer model swaps).
        for i, model in enumerate(reversed(answered), start=1):
            todo = [f for f in merged if model not in f.sources]
            if not todo:
                continue
            say(f"Model {i} of {len(answered)} is checking the other models' details.")
            statements = "\n".join(f"{n}. {f.text}" for n, f in enumerate(todo, start=1))
            try:
                out, timing = client.chat_json(model, VERIFY_PROMPT.format(statements=statements), image, VERIFY_SCHEMA)
            except OllamaError as e:
                failed[model] = f"while checking: {e}"
                continue
            descriptions[model]["check_seconds"] = timing["seconds"]
            for a in out.get("answers") or []:
                try:
                    n = int(a.get("n"))
                except (TypeError, ValueError):
                    continue
                if 1 <= n <= len(todo) and a.get("answer") in ("yes", "no", "unsure"):
                    todo[n - 1].votes[model] = a["answer"]

    order = {"disputed": 0, "agreed": 1, "unconfirmed": 2}
    merged.sort(key=lambda f: order[f.status(answered)])
    summary = _best_summary(descriptions, merged, answered)
    return Result(
        models=answered,
        summary=summary,
        spoken=spoken_summary(summary, merged, answered, failed) if answered else
        "No model could describe the photo. " + "; ".join(f"{m}: {e}" for m, e in failed.items()),
        facts=merged,
        descriptions=descriptions,
        failed=failed,
        seconds=round(time.monotonic() - started, 1),
    )


def _best_summary(descriptions: dict[str, dict[str, Any]], facts: list[Fact], models: list[str]) -> str:
    """The summary from the model whose own facts the others agreed with most."""
    best, best_score = "", -1.0
    for m in models:
        own = [f for f in facts if m in f.sources]
        score = sum(1 for f in own if f.status(models) == "agreed") / max(1, len(own))
        if score > best_score and descriptions[m]["summary"]:
            best, best_score = descriptions[m]["summary"], score
    return best
