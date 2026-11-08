"use strict";

// Everything from the server is inserted with textContent, never as HTML.
const $ = (id) => document.getElementById(id);
const MAX_SIDE = 1600;

function setStatus(text) {
  const el = $("status");
  if (el.textContent !== text) el.textContent = text; // repeat nothing to screen readers
}

// Downscale and re-encode as JPEG in the browser: smaller uploads, and the photo's
// metadata (such as GPS location) is dropped.
async function preparePhoto(file) {
  let bitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch {
    throw new Error("This browser cannot open that photo. Try a JPEG or PNG.");
  }
  const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
  return await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Could not read the photo."));
    reader.readAsDataURL(blob);
  });
}

function fillList(id, items, models) {
  const list = $(id);
  list.replaceChildren();
  const heading = $(id + "-heading");
  if (!items.length) {
    const li = document.createElement("li");
    li.textContent = "None.";
    list.append(li);
    return;
  }
  for (const item of items) {
    const li = document.createElement("li");
    const text = document.createElement("span");
    text.className = "fact";
    text.textContent = item.text;
    li.append(text);
    if (models.length > 1) {
      const who = document.createElement("span");
      who.className = "who";
      who.textContent = " (" + models.map((m) => m + ": " + item.judgements[m]).join("; ") + ")";
      li.append(who);
    }
    list.append(li);
  }
  heading.textContent = heading.textContent.replace(/ \(\d+\)$/, "") + " (" + items.length + ")";
}

function showResult(result) {
  $("spoken").textContent = result.spoken;
  fillList("disputed", result.disputed, result.models);
  fillList("agreed", result.agreed, result.models);
  fillList("unconfirmed", result.unconfirmed, result.models);
  const box = $("descriptions");
  box.replaceChildren();
  for (const [model, d] of Object.entries(result.descriptions)) {
    const h = document.createElement("h4");
    h.textContent = model;
    if (d.answer) {
      const a = document.createElement("p");
      a.textContent = "Answer: " + d.answer;
      box.append(h, a);
    }
    const p = document.createElement("p");
    p.textContent = d.summary;
    const ul = document.createElement("ul");
    for (const f of d.facts) {
      const li = document.createElement("li");
      li.textContent = f;
      ul.append(li);
    }
    if (!d.answer) box.append(h);
    box.append(p, ul);
  }
  for (const [model, err] of Object.entries(result.failed)) {
    const p = document.createElement("p");
    p.textContent = model + " did not answer: " + err;
    box.append(p);
  }
  $("timing").textContent = result.seconds < 1 ? "Took under a second." : "Took " + Math.round(result.seconds) + " seconds.";
  const section = $("result");
  section.hidden = false;
  section.focus(); // screen readers start at the result
  if ($("speak").checked && "speechSynthesis" in window) {
    const utterance = new SpeechSynthesisUtterance(result.spoken);
    const local = speechSynthesis.getVoices().find((v) => v.localService && v.lang.startsWith(document.documentElement.lang));
    if (local) utterance.voice = local; // prefer a voice that runs on this device
    speechSynthesis.cancel();
    speechSynthesis.speak(utterance);
  }
}

async function poll(job) {
  for (;;) {
    await new Promise((r) => setTimeout(r, 1000));
    const resp = await fetch("/api/jobs/" + encodeURIComponent(job));
    const data = await resp.json();
    if (data.state === "done") return data.result;
    if (data.state === "error" || !resp.ok) throw new Error(data.message || data.error || "Something went wrong.");
    setStatus(data.state === "queued" && data.ahead ? `Waiting: ${data.ahead} photo(s) ahead.` : data.message);
  }
}

async function onSubmit(event) {
  event.preventDefault();
  const file = $("photo").files[0];
  if (!file) {
    setStatus("Choose a photo first.");
    $("photo").focus();
    return;
  }
  const button = $("submit");
  button.disabled = true;
  $("result").hidden = true;
  try {
    setStatus("Preparing the photo.");
    const image = await preparePhoto(file);
    setStatus("Sending the photo to the models. This can take a minute.");
    const resp = await fetch("/api/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image, question: $("question").value }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "The photo was not accepted.");
    const result = await poll(data.job);
    setStatus("Result ready.");
    showResult(result);
  } catch (err) {
    setStatus(err.message);
  } finally {
    button.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  $("form").addEventListener("submit", onSubmit);
});
