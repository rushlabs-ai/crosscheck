"""Does disagreement between models predict wrong answers? An evaluation on VizWiz, on Modal.

    modal run eval/modal_eval.py                 # 100 questions
    modal run eval/modal_eval.py --limit 3       # smoke test

VizWiz-VQA (CC BY 4.0, vizwiz.org): photos taken by blind people, each with the question
they asked and ten crowd answers. For a sample of answerable questions, Crosscheck asks every
model, finds the consensus answer and whether all models agree, and the crowd answers say
whether the consensus was right. If Crosscheck is useful, wrong answers should be much more
common when the models disagree than when they all agree.

Runs on one L4 (or A10G, both 24 GB) GPU with the official Ollama 0.40.13 release (checksum-pinned) and this repo's
Crosscheck source. Results (per-question answers and the summary) go to eval/results/.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import modal

HERE = Path(__file__).parent
OLLAMA = ("v0.40.13", "c9490e0f520be21271d969e6703f012268b1d39f8b70dd58854e5bfef7a75bb6")
MODELS = ["qwen3-vl:4b-instruct", "gemma4:e4b", "ministral-3:3b"]
VIZWIZ = "https://vizwiz.cs.colorado.edu/VizWiz_final"

volume = modal.Volume.from_name("crosscheck-eval", create_if_missing=True)
app = modal.App("crosscheck-eval")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "zstd", "ca-certificates")
    .run_commands(
        f"curl -fsSL -o /tmp/ollama.tar.zst https://github.com/ollama/ollama/releases/download/{OLLAMA[0]}/ollama-linux-amd64.tar.zst",
        f"echo '{OLLAMA[1]}  /tmp/ollama.tar.zst' | sha256sum -c -",
        "mkdir -p /opt/ollama && tar --zstd -xf /tmp/ollama.tar.zst -C /opt/ollama && rm /tmp/ollama.tar.zst",
    )
    .env({"OLLAMA_MODELS": "/vol/ollama", "OLLAMA_HOST": "127.0.0.1:11434", "OLLAMA_MAX_LOADED_MODELS": "3",
          "PYTHONPATH": "/root/src"})
    .add_local_dir(HERE.parent / "src", "/root/src")
)

# VQA answer normalisation, simplified from the official VQA evaluation script.
_ARTICLES = {"a", "an", "the"}
_PUNCT = ";/[]\"{}()=+\\_-><@`,?!*#%^&$|~'"


def norm_answer(text: str) -> str:
    t = text.lower().strip().rstrip(".")
    for ch in _PUNCT:
        t = t.replace(ch, " ")
    return " ".join(w for w in t.split() if w not in _ARTICLES)


def vqa_accuracy(answer: str, crowd: list[str]) -> float:
    """Official VQA metric: an answer is fully right if at least 3 of 10 people gave it."""
    a = norm_answer(answer)
    return min(1.0, sum(1 for c in crowd if norm_answer(c) == a) / 3)


def lenient_correct(answer: str, crowd: list[str]) -> bool:
    """Right if it matches, or contains the words of, an answer at least 2 people gave."""
    from crosscheck.pipeline import same_answer

    a = norm_answer(answer)
    if not a or a in ("cannot tell", "unanswerable", "unknown"):
        return False
    counts = Counter(norm_answer(c) for c in crowd)
    for c, n in counts.items():
        if n < 2 or not c or c == "unanswerable":
            continue
        if c == a or same_answer(a, c) or set(c.split()) <= set(a.split()):
            return True
    return False


_ABSTAIN = ("cannot tell", "can't tell", "can not tell", "unknown", "not sure", "unanswerable", "unclear",
            "can't provide", "cannot provide", "cannot see", "can't see", "not visible", "unable to")


def abstained(answer: str) -> bool:
    from crosscheck.pipeline import is_abstention

    return is_abstention(answer)


def _get(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-fsSL", "--retry", "5", "-o", f"{dest}.part", url], check=True)
    os.replace(f"{dest}.part", dest)
    volume.commit()


def sample_questions(limit: int) -> tuple[list[dict], dict]:
    data = Path("/vol/vizwiz")
    _get(f"{VIZWIZ}/vqa_data/Annotations.zip", data / "Annotations.zip")
    _get(f"{VIZWIZ}/images/val.zip", data / "val.zip")
    with zipfile.ZipFile(data / "Annotations.zip") as z:
        name = next(n for n in z.namelist() if n.endswith("val.json"))
        items = json.loads(z.read(name))
    usable = [x for x in items if x.get("answerable") == 1 and x.get("answer_type") != "unanswerable"]
    usable.sort(key=lambda x: x["image"])
    random.Random(2025).shuffle(usable)
    chosen = usable[:limit]
    zip_sha = hashlib.sha256()
    with open(data / "val.zip", "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            zip_sha.update(chunk)
    with zipfile.ZipFile(data / "val.zip") as z:
        names = {Path(n).name: n for n in z.namelist()}
        for x in chosen:
            x["bytes"] = z.read(names[x["image"]])
    meta = {"val_questions": len(items), "answerable": len(usable), "val_zip_sha256": zip_sha.hexdigest()}
    return chosen, meta


@app.function(image=image, gpu=["L4", "A10G"], volumes={"/vol": volume}, timeout=3 * 3600)
def evaluate(limit: int = 100) -> dict:
    from crosscheck.ollama import Ollama
    from crosscheck.pipeline import run

    ollama = subprocess.Popen(["/opt/ollama/bin/ollama", "serve"], stdout=open("/tmp/ollama.log", "wb"),
                              stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=2)
            break
        except Exception:
            time.sleep(0.5)
    for m in MODELS:
        subprocess.run(["/opt/ollama/bin/ollama", "pull", m], check=True, stdout=subprocess.DEVNULL)
    volume.commit()

    questions, meta = sample_questions(limit)
    client = Ollama()
    rows = []
    for i, q in enumerate(questions, start=1):
        t0 = time.time()
        res = run(client, q["bytes"], MODELS, q["question"]).to_dict()
        seconds = time.time() - t0
        crowd = [a["answer"] for a in q["answers"]]
        per_model = {m: {"answer": a, "vqa": vqa_accuracy(a or "", crowd), "correct": lenient_correct(a or "", crowd)}
                     for m, a in res["answers"].items()}
        consensus = res["answer"] or ""
        if res["answer_agreement"] == "abstain":
            consensus = "cannot tell"
        rows.append({
            "image": q["image"], "question": q["question"], "answer_type": q.get("answer_type"),
            "crowd": crowd, "models": per_model, "consensus": consensus,
            "agreement": res["answer_agreement"], "consensus_correct": lenient_correct(consensus, crowd),
            "consensus_vqa": vqa_accuracy(consensus, crowd), "abstained": abstained(consensus),
            "disputed_facts": len(res["disputed"]),
            "failed": res["failed"], "seconds": round(seconds, 1),
        })
        print(f"{i}/{len(questions)} {seconds:.0f}s agreement={res['answer_agreement']} "
              f"correct={rows[-1]['consensus_correct']} q={q['question'][:60]!r}", flush=True)
    ollama.terminate()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
    return {"rows": rows, "meta": meta, "models": MODELS, "ollama": OLLAMA[0], "gpu": gpu.stdout.strip()}


def summarize(res: dict) -> dict:
    """Wrong answers should cluster where the models disagree. "Cannot tell" is counted as an
    abstention, not as a wrong answer: for a blind user it is the safe outcome."""
    rows = [r for r in res["rows"] if r["agreement"] is not None]
    n = len(rows)
    answered = [r for r in rows if not r["abstained"]]
    agree = [r for r in answered if r["agreement"] == "all"]
    split = [r for r in answered if r["agreement"] != "all"]
    wrong = [r for r in answered if not r["consensus_correct"]]

    def rate(xs, pred):
        return round(sum(1 for x in xs if pred(x)) / len(xs), 3) if xs else None

    secs = sorted(r["seconds"] for r in res["rows"][1:]) or [0]  # the first photo includes model loading
    return {
        "questions": n,
        "abstained": rate(rows, lambda r: r["abstained"]),
        "per_model_accuracy": {m: rate(rows, lambda r, m=m: (r["models"].get(m) or {}).get("correct"))
                               for m in res["models"]},
        "per_model_vqa_accuracy": {m: round(sum((r["models"].get(m) or {}).get("vqa", 0) for r in rows) / n, 3)
                                   for m in res["models"]} if n else {},
        "consensus_accuracy_when_answering": rate(answered, lambda r: r["consensus_correct"]),
        "all_agree_share_of_answers": rate(answered, lambda r: r["agreement"] == "all"),
        "wrong_rate_when_all_agree": rate(agree, lambda r: not r["consensus_correct"]),
        "wrong_rate_when_models_disagree": rate(split, lambda r: not r["consensus_correct"]),
        "wrong_answers_that_were_flagged": rate(wrong, lambda r: r["agreement"] != "all"),
        "counts": {"answered": len(answered), "all_agree": len(agree), "disagree": len(split), "wrong": len(wrong)},
        "seconds_per_photo_median": secs[len(secs) // 2],
        "seconds_per_photo_p90": secs[int(len(secs) * 0.9)],
    }


@app.local_entrypoint()
def main(limit: int = 100) -> None:
    res = evaluate.remote(limit)
    summary = summarize(res)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    name = "smoke" if limit < 20 else "vizwiz"
    (out / f"{name}.json").write_text(json.dumps({"summary": summary, **{k: v for k, v in res.items()}}, indent=2,
                                                 ensure_ascii=False))
    print(json.dumps(summary, indent=2))
    print(f"Saved {out / (name + '.json')}")
