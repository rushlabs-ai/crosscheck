# Changelog

## 0.1.0 (unreleased)

- `crosscheck serve`: a local, keyboard-only web app with a spoken summary first and details on demand.
- `crosscheck describe PHOTO`: the same in the terminal, or `--json`.
- Each model lists facts as JSON; facts are merged across models unless they differ in a number
  or quoted text; every model judges the others' facts (yes / no / unsure).
