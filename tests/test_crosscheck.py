"""Run: python -m unittest discover -s tests -v"""

from __future__ import annotations

import base64
import http.client
import io
import json
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fake_ollama import PNG, FakeOllama  # noqa: E402

from crosscheck import cli, server  # noqa: E402
from crosscheck.images import ImageError, check_image  # noqa: E402
from crosscheck.ollama import Ollama  # noqa: E402
from crosscheck.pipeline import answer_agreement, answer_consensus, is_abstention, run, same_fact  # noqa: E402
import fake_ollama  # noqa: E402


class SameFactTests(unittest.TestCase):
    def test_paraphrase_merges(self):
        self.assertTrue(same_fact("A red mug on a wooden table.", "A red mug sits on a wooden table."))

    def test_numbers_and_quotes_keep_facts_apart(self):
        self.assertFalse(same_fact('The label says "Ibuprofen 200 mg".', 'The label says "Ibuprofen 400 mg".'))
        self.assertFalse(same_fact('The sign reads "EXIT".', 'The sign reads "ENTRY".'))
        self.assertFalse(same_fact("There are 2 spoons.", "There are 3 spoons."))

    def test_unrelated(self):
        self.assertFalse(same_fact("A dog on grass.", "A red mug on a table."))


class PipelineTests(unittest.TestCase):
    def test_agreement_and_disagreement(self):
        with FakeOllama() as fake:
            messages = []
            res = run(Ollama(fake.url), PNG, ["a", "b"], progress=messages.append)
            d = res.to_dict()
            agreed = [f["text"] for f in d["agreed"]]
            disputed = [f["text"] for f in d["disputed"]]
            unconfirmed = [f["text"] for f in d["unconfirmed"]]
            self.assertIn("A red mug on a wooden table.", agreed)  # merged across models
            self.assertIn("There are 2 spoons.", agreed)
            self.assertEqual(len(disputed), 2)  # 200 mg vs 400 mg, both contradicted
            self.assertEqual(unconfirmed, ["A window is behind the mug."])
            self.assertIn("agree on 2 of 5 details", d["spoken"])
            self.assertIn("disagree about", d["spoken"])
            self.assertIn("Ibuprofen", d["spoken"])
            self.assertNotIn("verified", d["spoken"].lower())
            self.assertEqual(d["disputed"][0]["judgements"]["b"], "no")
            self.assertTrue(any("describing" in m for m in messages))
            self.assertTrue(any("checking" in m for m in messages))
            # Every request asked for structured output, without thinking, at temperature 0.
            for call in fake.state.calls:
                self.assertIs(call["think"], False)
                self.assertEqual(call["options"]["temperature"], 0)

    def test_a_failing_model_does_not_stop_the_others(self):
        with FakeOllama() as fake:
            d = run(Ollama(fake.url), PNG, ["a", "bad", "b"]).to_dict()
            self.assertEqual(d["models"], ["a", "b"])
            self.assertIn("bad", d["failed"])
            self.assertIn("1 model did not answer", d["spoken"])
            bad_calls = [c for c in fake.state.calls if c["model"] == "bad"]
            self.assertEqual(len(bad_calls), 2)  # one retry

    def test_one_model_cannot_crosscheck(self):
        with FakeOllama() as fake:
            d = run(Ollama(fake.url), PNG, ["a"]).to_dict()
            self.assertIn("Only one model answered", d["spoken"])
            self.assertEqual(len(d["agreed"]), 3)

    def test_question_goes_into_the_prompt(self):
        with FakeOllama() as fake:
            run(Ollama(fake.url), PNG, ["a"], question="What does the label say?")
            self.assertIn("What does the label say?", fake.state.calls[0]["messages"][0]["content"])


class ImageTests(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(check_image(PNG), PNG)
        self.assertEqual(check_image(b"\xff\xd8\xff\xe0rest"), b"\xff\xd8\xff\xe0rest")
        for bad in (b"", b"GIF89a....", b"<svg>"):
            with self.assertRaises(ImageError):
                check_image(bad)


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeOllama()
        self.httpd = server.serve(Ollama(self.fake.url), ["a", "b"], port=0)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.fake.__exit__()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        h = {"Host": f"127.0.0.1:{self.port}"}
        h.update(headers or {})
        data = json.dumps(body).encode() if isinstance(body, dict) else body
        if data is not None:
            h.setdefault("Content-Type", "application/json")
        conn.request(method, path, body=data, headers=h)
        resp = conn.getresponse()
        return resp.status, dict(resp.getheaders()), resp.read()

    def test_page_and_headers(self):
        status, headers, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b'role="status"', body)
        self.assertIn("script-src 'self'", headers["Content-Security-Policy"])
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        for path in ("/app.js", "/app.css"):
            self.assertEqual(self.request("GET", path)[0], 200)

    def test_dns_rebinding_and_cross_origin_are_refused(self):
        self.assertEqual(self.request("GET", "/", headers={"Host": f"evil.example:{self.port}"})[0], 403)
        img = "data:image/png;base64," + base64.b64encode(PNG).decode()
        status, _, _ = self.request("POST", "/api/check", {"image": img}, headers={"Origin": "https://evil.example"})
        self.assertEqual(status, 403)

    def test_bad_uploads(self):
        self.assertEqual(self.request("POST", "/api/check", {"image": "!!notbase64"})[0], 400)
        gif = base64.b64encode(b"GIF89a" + b"\x00" * 10).decode()
        status, _, body = self.request("POST", "/api/check", {"image": gif})
        self.assertEqual(status, 400)
        self.assertIn(b"JPEG or PNG", body)
        self.assertEqual(self.request("POST", "/api/check", b"x", headers={"Content-Type": "text/plain"})[0], 415)

    def test_full_check(self):
        img = "data:image/png;base64," + base64.b64encode(PNG).decode()
        status, _, body = self.request("POST", "/api/check", {"image": img, "question": "dose?"},
                                       headers={"Origin": f"http://127.0.0.1:{self.port}"})
        self.assertEqual(status, 202, body)
        job = json.loads(body)["job"]
        deadline = time.time() + 10
        while time.time() < deadline:
            status, _, body = self.request("GET", f"/api/jobs/{job}")
            data = json.loads(body)
            if data["state"] == "done":
                break
            time.sleep(0.05)
        self.assertEqual(data["state"], "done")
        self.assertEqual(len(data["result"]["disputed"]), 2)
        self.assertNotIn("image", data)
        self.assertEqual(self.request("GET", "/api/jobs/nope")[0], 404)


class CliTests(unittest.TestCase):
    def test_describe_json_and_missing_models(self):
        with FakeOllama() as fake, tempfile.TemporaryDirectory() as tmp:
            photo = Path(tmp) / "p.png"
            photo.write_bytes(PNG)
            out = io.StringIO()
            with redirect_stdout(out):
                code = cli.main(["describe", str(photo), "--ollama", fake.url, "--models", "a,b", "--json"])
            self.assertEqual(code, 0)
            self.assertEqual(len(json.loads(out.getvalue())["disputed"]), 2)

            err = io.StringIO()
            with redirect_stderr(err):
                code = cli.main(["describe", str(photo), "--ollama", fake.url, "--models", "a,qwen3-vl:4b"])
            self.assertEqual(code, 2)
            self.assertIn("ollama pull qwen3-vl:4b", err.getvalue())

    def test_text_output(self):
        with FakeOllama() as fake, tempfile.TemporaryDirectory() as tmp:
            photo = Path(tmp) / "p.png"
            photo.write_bytes(PNG)
            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(io.StringIO()):
                cli.main(["describe", str(photo), "--ollama", fake.url, "--models", "a,b"])
            text = out.getvalue()
            self.assertIn("Where the models disagree", text)
            self.assertIn("a: stated it; b: no", text)


class AnswerTests(unittest.TestCase):
    def test_consensus(self):
        self.assertEqual(answer_consensus({"a": "Red", "b": "red.", "c": "red"}), ("Red", "all"))
        self.assertEqual(answer_consensus({"a": "200 mg", "b": "400 mg", "c": "200 mg"}), ("200 mg", "most"))
        self.assertEqual(answer_consensus({"a": "a cat", "b": "a dog"})[1], "none")
        self.assertEqual(answer_consensus({"a": "It is a can of tomatoes", "b": "tomatoes, canned"})[1], "all")
        self.assertEqual(answer_consensus({"a": "Expires 2026", "b": "Expires 2027"})[1], "none")
        self.assertEqual(answer_consensus({"a": "", "b": ""}), (None, None))
        self.assertEqual(answer_consensus({"a": "blue"}), ("blue", None))

    def test_disagreeing_answers_lead_the_summary(self):
        with FakeOllama() as fake:
            d = run(Ollama(fake.url), PNG, ["a", "b"], question="What is the dose?").to_dict()
            self.assertEqual(d["answer_agreement"], "none")
            self.assertTrue(d["spoken"].startswith("The models give different answers: a says Ibuprofen 200 mg; "
                                                   "b says Ibuprofen 400 mg."), d["spoken"])
            self.assertEqual(d["answers"], {"a": "Ibuprofen 200 mg", "b": "Ibuprofen 400 mg"})

    def test_majority_answer(self):
        with FakeOllama() as fake:
            d = run(Ollama(fake.url), PNG, ["a", "b", "c"], question="What is the dose?").to_dict()
            self.assertEqual(d["answer_agreement"], "most")
            self.assertTrue(d["spoken"].startswith("Probably: Ibuprofen 200 mg."), d["spoken"])

    def test_no_question_no_answer(self):
        with FakeOllama() as fake:
            d = run(Ollama(fake.url), PNG, ["a", "b"]).to_dict()
            self.assertIsNone(d["answer"])
            self.assertEqual(d["answers"], {})


class SemanticAgreementTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(fake_ollama.ANSWERS)

    def tearDown(self):
        fake_ollama.ANSWERS.clear()
        fake_ollama.ANSWERS.update(self.saved)

    def test_rules(self):
        self.assertEqual(answer_agreement({"a": "x", "b": "cannot tell"}, {}), ("x", "single"))
        self.assertEqual(answer_agreement({"a": "Cannot tell", "b": "I'm sorry, I can't see it"}, {}), ("cannot tell", "abstain"))
        votes = {"a": {"b": "yes"}, "b": {"a": "yes"}}
        self.assertEqual(answer_agreement({"a": "Blue and grey", "b": "a blue-gray backpack"}, votes)[1], "all")
        votes = {"a": {"b": "no"}, "b": {"a": "no"}}
        self.assertEqual(answer_agreement({"a": "cat", "b": "dog"}, votes)[1], "none")
        self.assertTrue(is_abstention("The image is too small and blurry to provide a description."))
        self.assertFalse(is_abstention("Doritos"))

    def test_different_words_same_answer_counts_as_agreement(self):
        fake_ollama.ANSWERS.update({"a": "Blue and grey", "b": "a blue-gray backpack"})
        with FakeOllama() as fake:
            d = run(Ollama(fake.url), PNG, ["a", "b"], question="What colour is my bag?").to_dict()
            self.assertEqual(d["answer_agreement"], "all", d["spoken"])
            self.assertTrue(d["spoken"].startswith("Answer: Blue and grey. All 2 models give this answer."))
            judged = [c for c in fake.state.calls if "the answer is:" in c["messages"][0]["content"]]
            self.assertEqual(len(judged), 2)  # each model judged the other's answer

    def test_one_model_abstaining_is_not_disagreement(self):
        fake_ollama.ANSWERS.update({"a": "cannot tell", "b": "Ibuprofen 400 mg"})
        with FakeOllama() as fake:
            d = run(Ollama(fake.url), PNG, ["a", "b"], question="What is the dose?").to_dict()
            self.assertEqual(d["answer_agreement"], "single")
            self.assertIn("Only one model could answer", d["spoken"])


if __name__ == "__main__":
    unittest.main()
