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

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "summary": {"type": "string"},
        "facts": {"type": "array", "items": {"type": "string"}, "maxItems": 15},
    },
    "required": ["answer", "summary", "facts"],
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
    answer: str | None
    answer_agreement: str | None  # "all", "most" or "none" (None without a question)
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
        return {"models": self.models, "summary": self.summary, "answer": self.answer,
                "answers": {m: d.get("answer") for m, d in self.descriptions.items() if "answer" in d},
                "answer_agreement": self.answer_agreement, "spoken": self.spoken, **groups,
                "descriptions": self.descriptions, "failed": self.failed, "seconds": self.seconds}


Progress = Callable[[str], None]


def _short(text: str, limit: int = 90) -> str:
    text = text.strip().rstrip(".")
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


_STOP = {"a", "an", "the", "it", "is", "its", "it's", "this", "that", "of", "and", "there", "be", "are", "looks",
         "like", "appears", "to", "seems", "in", "on", "photo", "image", "picture", "shows"}


def _stem(w: str) -> str:
    """Very light stemming so "canned" matches "can" and "tomatoes" matches "tomato"."""
    for suffix in ("ing", "ed", "es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            w = w[: -len(suffix)]
            break
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiou":
        w = w[:-1]
    return w


def same_answer(a: str, b: str) -> bool:
    """Short answers match on their content words, and on any numbers or quoted text."""
    if _norm(a).strip(" .") == _norm(b).strip(" ."):
        return True
    x = [_stem(w) for w in _WORD.findall(_norm(a)) if w not in _STOP]
    y = [_stem(w) for w in _WORD.findall(_norm(b)) if w not in _STOP]
    if not x or not y or _anchors(a) != _anchors(b):
        return False
    common = sum((Counter(x) & Counter(y)).values())
    return common > 0 and 2 * common / (len(x) + len(y)) >= 0.6


_ABSTAIN = ("cannot tell", "can't tell", "can not tell", "unknown", "not sure", "unanswerable", "unclear",
            "can't provide", "cannot provide", "cannot see", "can't see", "not visible", "unable to", "too blurry",
            "too small and blurry", "i'm sorry")


def is_abstention(answer: str) -> bool:
    """'Cannot tell' and refusals: the model did not answer (which is not the same as disagreeing)."""
    a = (answer or "").lower().strip()
    return not a or any(p in a for p in _ABSTAIN)


def answer_statement(question: str, answer: str) -> str:
    return f'For the question "{question.strip()}", the answer is: {answer.strip()}'


def answer_agreement(answers: dict[str, str], votes: dict[str, dict[str, str]]) -> tuple[str | None, str | None]:
    """Consensus over the models that answered. Model O backs model M's answer if O gave the same
    answer, or judged M's answer "yes" after looking at the photo itself.

    Returns (answer, agreement) with agreement "all", "most", "none", "single" (one model answered)
    or "abstain" (none did)."""
    if not answers:
        return None, None
    answering = {m: a for m, a in answers.items() if not is_abstention(a)}
    if not answering:
        return "cannot tell", "abstain"
    if len(answering) == 1:
        return next(iter(answering.values())), "single"
    best, best_key = None, None
    for m, a in answering.items():
        backers = {m} | {o for o, b in answering.items() if o != m and (same_answer(a, b) or votes.get(m, {}).get(o) == "yes")}
        against = sum(1 for o, b in answering.items() if o != m and not same_answer(a, b) and votes.get(m, {}).get(o) == "no")
        key = (len(backers), -against)
        if best_key is None or key > best_key:
            best, best_key = m, key
    n_backers, neg_against = best_key
    if n_backers == len(answering) and neg_against == 0:
        level = "all"
    elif n_backers > len(answering) / 2:
        level = "most"
    else:
        level = "none"
    return answering[best], level


def answer_consensus(answers: dict[str, str]) -> tuple[str | None, str | None]:
    """The answer most models give, and whether all, most or none of them give it."""
    given = {m: a for m, a in answers.items() if a}
    if not given:
        return None, None
    groups: list[list[str]] = []
    for m, a in given.items():
        for g in groups:
            if same_answer(given[g[0]], a):
                g.append(m)
                break
        else:
            groups.append([m])
    best = max(groups, key=len)
    if len(given) == 1:
        return given[best[0]], None
    if len(best) == len(given):
        return given[best[0]], "all"
    return given[best[0]], ("most" if len(best) > len(given) / 2 else "none")


def spoken_summary(summary: str, facts: list[Fact], models: list[str], failed: dict[str, str],
                   answer: str | None = None, agreement: str | None = None,
                   answers: dict[str, str] | None = None) -> str:
    agreed = [f for f in facts if f.status(models) == "agreed"]
    disputed = [f for f in facts if f.status(models) == "disputed"]
    parts = []
    if answer is not None:
        if agreement == "all":
            parts.append(f"Answer: {answer.rstrip('.')}. All {len(models)} models give this answer.")
        elif agreement == "most":
            parts.append(f"Probably: {answer.rstrip('.')}. Most models give this answer, but not all.")
        elif agreement == "abstain":
            parts.append("The models cannot tell that from this photo.")
        elif agreement == "single":
            parts.append(f"Only one model could answer: {answer.rstrip('.')}. Treat this as unconfirmed.")
        elif agreement == "none":
            listed = "; ".join(f"{m} says {a.rstrip('.')}" for m, a in (answers or {}).items() if a and not is_abstention(a))
            parts.append(f"The models give different answers: {listed}. Treat this as uncertain.")
        else:
            parts.append(f"Answer: {answer.rstrip('.')}.")
    if summary.strip():
        parts.append(summary.strip().rstrip(".") + ".")
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
    q = (f"\nThe person asks: {question.strip()}\nPut a short, direct answer in 'answer' (a few words; say 'cannot tell' "
         "if the photo does not show it), then describe the rest." if question.strip() else "")
    schema = ANSWER_SCHEMA if question.strip() else DESCRIBE_SCHEMA
    descriptions: dict[str, dict[str, Any]] = {}
    failed: dict[str, str] = {}

    for i, model in enumerate(models, start=1):
        say(f"Model {i} of {len(models)} is describing the photo.")
        try:
            out, timing = client.chat_json(model, DESCRIBE_PROMPT.format(question=q), image, schema)
        except OllamaError as e:
            failed[model] = str(e)
            continue
        facts = [str(x).strip() for x in (out.get("facts") or []) if str(x).strip()][:15]
        descriptions[model] = {"summary": str(out.get("summary") or "").strip(), "facts": facts, **timing}
        if question.strip():
            descriptions[model]["answer"] = str(out.get("answer") or "").strip()

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

    answers = {m: descriptions[m].get("answer", "") for m in answered} if question.strip() else {}
    answer_votes: dict[str, dict[str, str]] = {m: {} for m in answers}  # owner -> judge -> vote
    if len(answered) > 1:
        # Judge in reverse order so the model loaded last goes first (fewer model swaps).
        for i, model in enumerate(reversed(answered), start=1):
            todo = [f for f in merged if model not in f.sources]
            others = [o for o, a in answers.items() if o != model and not is_abstention(a)]
            if not todo and not others:
                continue
            say(f"Model {i} of {len(answered)} is checking the other models' details.")
            texts = [f.text for f in todo] + [answer_statement(question, answers[o]) for o in others]
            statements = "\n".join(f"{n}. {t}" for n, t in enumerate(texts, start=1))
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
                if a.get("answer") not in ("yes", "no", "unsure"):
                    continue
                if 1 <= n <= len(todo):
                    todo[n - 1].votes[model] = a["answer"]
                elif len(todo) < n <= len(todo) + len(others):
                    answer_votes[others[n - len(todo) - 1]][model] = a["answer"]

    order = {"disputed": 0, "agreed": 1, "unconfirmed": 2}
    merged.sort(key=lambda f: order[f.status(answered)])
    summary = _best_summary(descriptions, merged, answered)
    answer, agreement = answer_agreement(answers, answer_votes) if answers else (None, None)
    for m in answers:
        descriptions[m]["answer_judged_by"] = answer_votes.get(m, {})
    return Result(
        models=answered,
        summary=summary,
        answer=answer,
        answer_agreement=agreement,
        spoken=spoken_summary(summary, merged, answered, failed, answer, agreement, answers) if answered else
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
