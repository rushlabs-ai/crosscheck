"""Re-grade a VizWiz run with a text-only judge, and test an alternative agreement rule.

    modal run eval/rescore.py --results eval/results/vizwiz.json

1. Grading: matching free-form answers to crowd answers by words is crude ("Maxell CD-R disc"
   vs "maxell"). A text model (Qwen3-8B, not one of the vision models being graded) decides
   whether an answer matches what the crowd said: correct, partly, wrong, or abstain.
2. Agreement: instead of vision models judging each other's answers (they say "yes" too
   easily), the same text judge decides whether the models' *independent* answers mean the same.
Both run on answers already collected; no vision model is called again.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from collections import Counter
from pathlib import Path

import modal

HERE = Path(__file__).parent
OLLAMA = ("v0.40.13", "c9490e0f520be21271d969e6703f012268b1d39f8b70dd58854e5bfef7a75bb6")
JUDGE = "qwen3:8b"

volume = modal.Volume.from_name("crosscheck-eval", create_if_missing=True)
app = modal.App("crosscheck-rescore")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "zstd", "ca-certificates")
    .run_commands(
        f"curl -fsSL -o /tmp/ollama.tar.zst https://github.com/ollama/ollama/releases/download/{OLLAMA[0]}/ollama-linux-amd64.tar.zst",
        f"echo '{OLLAMA[1]}  /tmp/ollama.tar.zst' | sha256sum -c -",
        "mkdir -p /opt/ollama && tar --zstd -xf /tmp/ollama.tar.zst -C /opt/ollama && rm /tmp/ollama.tar.zst",
    )
    .env({"OLLAMA_MODELS": "/vol/ollama", "OLLAMA_HOST": "127.0.0.1:11434"})
)

GRADE_SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["correct", "partly", "wrong", "abstain"]}},
                "required": ["verdict"]}
SAME_SCHEMA = {"type": "object", "properties": {"same": {"type": "string", "enum": ["yes", "no"]}}, "required": ["same"]}


def chat(prompt: str, schema: dict) -> dict:
    body = {"model": JUDGE, "messages": [{"role": "user", "content": prompt}], "format": schema, "stream": False,
            "think": False, "options": {"temperature": 0}}
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(json.loads(r.read())["message"]["content"])


def grade(question: str, crowd: list[str], answer: str) -> str:
    counts = Counter(c.strip().lower() for c in crowd)
    listed = "; ".join(f"{a} ({n})" for a, n in counts.most_common())
    return chat(
        "A blind person took a photo and asked a question. Ten sighted people answered it (counts in brackets).\n"
        f"Question: {question}\nPeople's answers: {listed}\nAnswer to grade: {answer}\n\n"
        "Grade the answer against what most people said. 'correct' if it gives the same information as the most "
        "common meaningful answer (extra detail is fine), 'partly' if it is partly right, 'wrong' if it contradicts "
        "them, 'abstain' if it only says it cannot tell or cannot see. If most people said 'unanswerable', an "
        "answer that guesses specific content is 'wrong'.", GRADE_SCHEMA)["verdict"]


def same(question: str, a: str, b: str) -> bool:
    return chat(f"Two answers to the question \"{question}\" about the same photo.\nA: {a}\nB: {b}\n"
                "Do A and B give the same answer (ignore wording, detail and spelling)?", SAME_SCHEMA)["same"] == "yes"


@app.function(image=image, gpu=["L4", "A10G"], volumes={"/vol": volume}, timeout=2 * 3600)
def rescore(rows: list[dict]) -> list[dict]:
    from itertools import combinations

    proc = subprocess.Popen(["/opt/ollama/bin/ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=2)
            break
        except Exception:
            time.sleep(0.5)
    subprocess.run(["/opt/ollama/bin/ollama", "pull", JUDGE], check=True, stdout=subprocess.DEVNULL)
    volume.commit()
    out = []
    for i, r in enumerate(rows, 1):
        answers = {m: (x.get("answer") or "") for m, x in r["models"].items()}
        verdicts = {m: grade(r["question"], r["crowd"], a) for m, a in answers.items()}
        cons = grade(r["question"], r["crowd"], r["consensus"]) if r["consensus"] else "abstain"
        given = {m: a for m, a in answers.items() if verdicts[m] != "abstain"}
        pairs = {f"{x}|{y}": same(r["question"], given[x], given[y]) for x, y in combinations(given, 2)}
        out.append({**r, "judge": {"models": verdicts, "consensus": cons, "same_pairs": pairs}})
        if i % 10 == 0:
            print(f"{i}/{len(rows)}", flush=True)
    proc.terminate()
    return out


def summarize(rows: list[dict]) -> dict:
    def rate(xs, pred):
        return round(sum(1 for x in xs if pred(x)) / len(xs), 3) if xs else None

    models = sorted({m for r in rows for m in r["models"]})
    ok = lambda v: v == "correct"  # noqa: E731
    answered = [r for r in rows if r["judge"]["consensus"] != "abstain"]
    # (a) the run's own agreement (vision models judging each other)
    agree_a = [r for r in answered if r["agreement"] == "all"]
    split_a = [r for r in answered if r["agreement"] != "all"]
    # (b) independent answers judged equivalent by the text model: all pairs the same, at least 2 answers
    def indep(r):
        p = r["judge"]["same_pairs"]
        return bool(p) and all(p.values())
    agree_b = [r for r in answered if indep(r)]
    split_b = [r for r in answered if not indep(r)]
    wrong = [r for r in answered if not ok(r["judge"]["consensus"])]
    return {
        "questions": len(rows),
        "per_model_correct": {m: rate(rows, lambda r, m=m: ok(r["judge"]["models"].get(m))) for m in models},
        "per_model_abstain": {m: rate(rows, lambda r, m=m: r["judge"]["models"].get(m, "abstain") == "abstain") for m in models},
        "consensus_correct_when_answering": rate(answered, lambda r: ok(r["judge"]["consensus"])),
        "consensus_abstained": rate(rows, lambda r: r["judge"]["consensus"] == "abstain"),
        "a_vision_cross_judging": {"agree_share": rate(answered, lambda r: r in agree_a),
                                   "wrong_when_agree": rate(agree_a, lambda r: not ok(r["judge"]["consensus"])),
                                   "wrong_when_disagree": rate(split_a, lambda r: not ok(r["judge"]["consensus"])),
                                   "wrong_answers_flagged": rate(wrong, lambda r: r in split_a)},
        "b_independent_answers_equivalent": {"agree_share": rate(answered, lambda r: r in agree_b),
                                             "wrong_when_agree": rate(agree_b, lambda r: not ok(r["judge"]["consensus"])),
                                             "wrong_when_disagree": rate(split_b, lambda r: not ok(r["judge"]["consensus"])),
                                             "wrong_answers_flagged": rate(wrong, lambda r: r in split_b)},
    }


@app.local_entrypoint()
def main(results: str = "eval/results/vizwiz.json") -> None:
    path = Path(results)
    data = json.loads(path.read_text())
    rows = rescore.remote(data["rows"])
    data["rows"] = rows
    data["judge"] = JUDGE
    out = path.with_name(path.stem + "-judged.json")
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))  # save before summarising
    data["judged_summary"] = summarize(rows)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(json.dumps(data["judged_summary"], indent=2))
    print(f"Saved {out}")
