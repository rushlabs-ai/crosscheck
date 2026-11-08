# Crosscheck

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
4. You hear a short summary first, for example:

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
   ollama pull granite3.2-vision:2b
   ```

2. Install and start Crosscheck (Python 3.10+, no other dependencies):

   ```sh
   uv tool install crosscheck-vision      # or: pipx install crosscheck-vision
   crosscheck serve
   ```

   It opens http://127.0.0.1:8765 in your browser.

Choose other models with `--models a,b,c` (or `CROSSCHECK_MODELS`). With a 12–16 GB GPU, expect
about 30–90 seconds per photo while Ollama swaps the models in and out; a CPU works, slowly.

## Using it

- **Web app**: choose a photo, optionally ask a question ("What does the label say?"), press
  *Check photo*. Progress is announced; when the result is ready, focus moves to it. Everything
  works with the keyboard and a screen reader. A checkbox can also read the result aloud with
  the browser's own voice.
- **Terminal**: `crosscheck describe photo.jpg --question "What does the label say?"`, or `--json`
  for the full result.

The web app shrinks the photo in your browser before sending it, which also removes metadata
such as GPS location.

## Privacy and security

- Runs on 127.0.0.1 only. Photos stay in memory and go only to your Ollama. Nothing is logged.
- Requests must come from the app itself: a non-local `Host` (DNS rebinding) or a foreign
  `Origin` (another website) is refused. A strict Content-Security-Policy blocks outside content.
- Model output is shown as text, never as HTML.

## Status and limits

- v0.1, for desktop browsers on the same computer. Using a phone camera over Wi-Fi needs HTTPS
  and is planned for v0.2.
- Merging facts across models uses word overlap, so two phrasings of one fact are sometimes
  listed separately; each still gets judged.
- Small local models are weaker than the cloud models in the study. We are measuring how well
  their disagreement predicts real errors on photos taken by blind people (VizWiz), and will
  publish the results and timings here.

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
