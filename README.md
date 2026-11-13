# Crosscheck

[![CI](https://github.com/rushlabs-ai/crosscheck/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rushlabs-ai/crosscheck/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/crosscheck-vision)](https://pypi.org/project/crosscheck-vision/) [![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/rushlabs-ai/crosscheck/blob/main/LICENSE)

**A second opinion for AI photo descriptions.** Several vision models, running on your own
computer, describe the same photo. Crosscheck tells you what they all agree on, and, first and
most clearly, where they disagree.

Built for blind and low-vision people, who use AI descriptions for labels, medicine, mail and
screens but have no easy way to tell when a description is wrong.

## Why

A single AI description sounds equally confident when it is right and when it is wrong. A study
at ASSETS 2025, [*Surfacing Variations to Calibrate Perceived Reliability of MLLM-generated Image
Descriptions*](https://arxiv.org/abs/2507.15692) (Meng Chen, Akhil Iyer, Amy Pavel), found that
when blind and low-vision participants were shown where several models' descriptions varied,
they spotted **4.9 times as many unreliable claims** as with a single description (2.62 against
0.53 per image). 11 of 15 participants ranked that presentation as their favourite, and all of
them wanted to use it in the future.

Crosscheck brings that idea to models that run locally through [Ollama](https://ollama.com):
free, offline, and the photo never leaves your computer.

## How it works

1. Each model lists what it sees as short, separate facts, and quotes any text it can read.
2. Facts that say the same thing are merged. Facts that differ in a **number or quoted text**
   ("Ibuprofen 200 mg" against "400 mg") are kept apart, because that is where errors hide.
3. Every model looks at the photo again and judges each fact the others stated: yes, no, or unsure.
4. If you asked a question, each model also answers it in a few words, and the models judge each
   other's answers the same way. "Cannot tell" counts as not answering, not as disagreeing.
5. You hear a short summary first, for example:

   > A pill bottle on a wooden table. The 3 models agree on 6 of 9 details. They disagree about:
   > The label says "Ibuprofen 200 mg"; The label says "Ibuprofen 400 mg".

   Then, on demand: what every model agrees on, where they disagree (and which model said what),
   and details not every model could judge.

**Agreement is not proof.** Models can make the same mistake. For medicine, money or safety,
confirm with a person or another trusted source. Crosscheck never says "verified".

## Install

1. Install [Ollama](https://ollama.com) and pull three vision models from different families
   (different families make different mistakes):

   ```sh
   ollama pull qwen3-vl:4b-instruct
   ollama pull gemma4:e4b
   ollama pull ministral-3:3b
   ```

2. Install and start Crosscheck (Python 3.10+, no other dependencies):

   ```sh
   uv tool install crosscheck-vision      # or: pipx install crosscheck-vision
   crosscheck serve
   ```

   It opens http://127.0.0.1:8765 in your browser.

Choose other models with `--models a,b,c` (or `CROSSCHECK_MODELS`). With a 12–16 GB GPU, expect
about 30–90 seconds per photo while Ollama swaps the models in and out. A CPU works, slowly.

## Using it

- **Web app**: choose a photo, optionally ask a question ("What does the label say?"), press
  *Check photo*. Progress is announced, and when the result is ready, focus moves to it. Everything
  works with the keyboard and a screen reader. A checkbox can also read the result aloud with
  the browser's own voice.
- **Terminal**: `crosscheck describe photo.jpg --question "What does the label say?"`, or `--json`
  for the full result.

The web app shrinks the photo in your browser before sending it, which also removes metadata
such as GPS location.

### From your phone

```sh
crosscheck serve --lan
```

Crosscheck then also listens on your home network and prints an address such as
`http://192.168.1.20:8765/` and a 6-digit pairing code. Open the address on your phone (same Wi-Fi),
enter the code once, and use **Take a photo** to open the camera. A link with the code,
`http://192.168.1.20:8765/?code=123456`, pairs in one step.

- Every device except the computer running Crosscheck must enter the code. After five wrong
  codes a new one is shown and pairing pauses for a minute, so guessing would take months.
  Paired devices get a session cookie for 30 days.
- The page uses the phone's normal camera picker, which works without HTTPS. Photos then travel
  over your Wi-Fi unencrypted, so use `--lan` only on a network you trust, or add
  `--tls-cert cert.pem --tls-key key.pem` for HTTPS (a self-signed certificate works after your
  phone's warning).

## Evaluation

`eval/modal_eval.py` runs Crosscheck on 100 photos taken by blind people, from the VizWiz-VQA
validation set (CC BY 4.0), each with the question its photographer asked and ten crowd answers.
`eval/rescore.py` then grades every answer against the crowd answers with a separate text model
(Qwen3-8B), because word matching misjudges free-form answers. Results on one NVIDIA A10, all three
default models loaded (`eval/results/vizwiz-judged.json`):

| | Result |
|---|---|
| Correct, single model (Qwen3-VL 4B / Gemma 4 E4B / Ministral 3 3B) | 56% / 55% / 45% |
| **Correct, Crosscheck's combined answer** | **63%** |
| Wrong when the models disagree | 45% |
| Wrong when all models agree | 33% |
| Wrong answers that were flagged as a disagreement | 38% |
| Time per photo (median / 90th percentile) | 13 s / 42 s |

What this means:

- Combining three small models gives a better answer than any one of them.
- Disagreement is a real warning sign (errors are more common), but **agreement does not mean
  the answer is right**: a third of agreed answers were still wrong. VizWiz photos are often
  blurry, dark or badly framed, which is exactly when blind users most need help and when small
  models are weakest.
- An earlier version compared answers by their words and used Granite 3.2 Vision, which refused
  38% of photos. It flagged almost everything and told users nothing. Its results are kept in
  `eval/results/vizwiz-v1-granite-lexical.json`.

The ASSETS '25 study used large cloud models. We have not yet measured how much bigger local
models (for example on a 24 GB GPU) improve these numbers.

## Privacy and security

- Runs on 127.0.0.1 only unless you pass `--lan`. Photos stay in memory and go only to your
  Ollama. Nothing is logged.
- With `--lan`, other devices must pair with a code shown on the computer. The exemption for the
  computer itself is decided by the connection's address, never by a header a client controls.
- Requests must come from the app itself: a `Host` that is not localhost or an IP address (DNS
  rebinding) or a foreign `Origin` (another website) is refused. A strict Content-Security-Policy
  blocks outside content.
- Model output is shown as text, never as HTML.

## Status and limits

- Phones work over your home Wi-Fi with `--lan` (see above). Tested with Chrome's phone
  emulation. Reports from real iPhones and Android phones with VoiceOver or TalkBack are welcome.
- Merging facts across models uses word overlap, so two phrasings of one fact are sometimes
  listed separately, but each still gets judged.
- Small local models are weaker than the cloud models in the study, and their agreement is a
  weaker signal (see Evaluation). Treat "the models agree" as encouraging, never as proof.

Feedback from blind and low-vision users matters most: please open an issue or email
admin@rushlabs.ai.

## Development

```sh
python -m unittest discover -s tests -v
```

Tests run against a fake Ollama: merging and judging, a failing model, the web app's security
checks, the job queue and the CLI.

## Licence

Apache-2.0. Built by [Rush Labs](https://rushlabs.ai), an applied AI lab. The models keep their
own licences (all three defaults are Apache-2.0).
