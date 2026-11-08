# Changelog

## Unreleased

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
