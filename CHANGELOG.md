# Changelog

## 0.3.0

- `crosscheck serve --lan`: use Crosscheck from a phone on your home Wi-Fi. Devices pair once with a
  6-digit code shown on the computer (after five wrong tries: a new code and a one-minute pause);
  the computer itself needs no code. **Take a photo** opens the phone's camera. Optional HTTPS with `--tls-cert` / `--tls-key`.
- Host checks now allow IP addresses on the network but still refuse domain names (DNS rebinding);
  POSTs must be same-origin.

## 0.2.0

- Questions get a short, direct answer first, with whether the models agree on it. The models
  judge each other's answers by looking at the photo; "cannot tell" counts as not answering.
- Default models: Ministral 3 3B replaces Granite 3.2 Vision, which often refused to describe photos.
- Evaluation on 100 VizWiz photos (`eval/`): the combined answer is right more often than any single
  model (63% vs at most 56%); agreement is a warning signal, not proof.

## 0.1.0 (unreleased)

- `crosscheck serve`: a local, keyboard-only web app with a spoken summary first and details on demand.
- `crosscheck describe PHOTO`: the same in the terminal, or `--json`.
- Each model lists facts as JSON; facts are merged across models unless they differ in a number
  or quoted text; every model judges the others' facts (yes / no / unsure).
