# Security

Report vulnerabilities privately through GitHub's "Report a vulnerability" button on this
repository, or by email to vayun@rushlabs.ai.

Crosscheck listens on 127.0.0.1 only, refuses requests with a non-local Host header (DNS
rebinding) or a foreign Origin (cross-site requests), keeps photos in memory, and sends them
only to the Ollama you point it at. It does not log photos or questions.
