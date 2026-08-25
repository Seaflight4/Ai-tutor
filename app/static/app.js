"use strict";

// AI Tutor frontend — vanilla JS, no dependencies.
// Consumes the existing FastAPI surface under /api.

const state = {
  sessionId: null,
  studentId: null,
  loopCount: 0,
  busy: false,
  resolved: false,
};

// DOM -----------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const fileInput = $("file-input");
const fileLabel = $("file-label");
const fileDrop = document.querySelector(".file-drop");
const preview = $("preview");
const externalRef = $("external-ref");
const startBtn = $("start-btn");
const problemMeta = $("problem-meta");
const problemText = $("problem-text");
const conceptsList = $("concepts");
const statusEl = $("status");
const messages = $("messages");
const chatForm = $("chat-form");
const chatInput = $("chat-input");
const sendBtn = $("send-btn");

// Upload --------------------------------------------------------------------
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  fileLabel.textContent = file.name;
  fileDrop.classList.add("has-file");
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.hidden = false;
  startBtn.disabled = false;
});

startBtn.addEventListener("click", startSession);

async function startSession() {
  const file = fileInput.files[0];
  if (!file || state.busy) return;
  setBusy(true);
  startBtn.disabled = true;
  startBtn.textContent = "Starting…";

  const form = new FormData();
  form.append("file", file, file.name);
  if (externalRef.value.trim()) {
    form.append("external_ref", externalRef.value.trim());
  }

  try {
    const res = await fetch("/api/sessions", { method: "POST", body: form });
    if (!res.ok) throw new Error(await res.text());
    const session = await res.json();
    state.sessionId = session.id;
    state.studentId = session.student_id;
    state.loopCount = session.loop_count;
    state.resolved = session.resolved;

    showProblemMeta(session);
    showStatus(session.id, state.loopCount);

    // The create endpoint strips the opening message, so fetch the transcript
    // to get the opening tutor turn.
    const tRes = await fetch(`/api/sessions/${state.sessionId}`);
    const tBody = await tRes.json();
    for (const turn of tBody.turns) {
      addBubble(turn.role, turn.content);
    }

    enableChat();
  } catch (err) {
    showError(err.message || "Failed to start session");
    startBtn.disabled = false;
    startBtn.textContent = "Start session";
  } finally {
    setBusy(false);
  }
}

function showProblemMeta(session) {
  problemText.textContent = session.problem_text || "(no text extracted)";
  conceptsList.innerHTML = "";
  for (const c of session.concepts || []) {
    const li = document.createElement("li");
    li.textContent = c;
    conceptsList.appendChild(li);
  }
  problemMeta.hidden = false;
}

function showStatus(id, loop) {
  const short = String(id).slice(0, 8);
  statusEl.textContent = `session ${short}… · loop ${loop}`;
  statusEl.hidden = false;
}

// Chat ---------------------------------------------------------------------
chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const msg = chatInput.value.trim();
  if (!msg || state.busy || state.resolved) return;
  sendReply(msg);
});

async function sendReply(message) {
  if (state.busy || state.resolved || !state.sessionId) return;
  setBusy(true);
  addBubble("student", message);
  chatInput.value = "";
  autoSize();

  try {
    const res = await fetch(`/api/sessions/${state.sessionId}/reply`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    if (!res.ok) throw new Error(await res.text());
    const reply = await res.json();

    state.loopCount = reply.loop_index;

    if (reply.hint) {
      addHintBubble(reply.hint, reply.classification, reply.content);
    } else {
      addBubble("tutor", reply.content);
    }
    showStatus(state.sessionId, state.loopCount);

    if (reply.resolved) {
      state.resolved = true;
      addBubble("system", "Session resolved.");
      disableChat();
    }
  } catch (err) {
    showError(err.message || "Reply failed");
  } finally {
    setBusy(false);
  }
}

// Helpers -------------------------------------------------------------------
function setBusy(v) {
  state.busy = v;
  sendBtn.disabled = v || !state.sessionId || state.resolved;
  chatInput.disabled = v || !state.sessionId || state.resolved;
}

function enableChat() {
  chatInput.disabled = false;
  sendBtn.disabled = false;
  chatInput.focus();
}

function disableChat() {
  chatInput.disabled = true;
  sendBtn.disabled = true;
}

function addBubble(role, content, modifier = null) {
  const div = document.createElement("div");
  div.className = "bubble " + (modifier || (role === "student" ? "student" : "tutor"));
  div.textContent = content;
  messages.appendChild(div);
  renderMath(div);
  messages.scrollTop = messages.scrollHeight;
}

function showError(msg) {
  const div = document.createElement("div");
  div.className = "bubble error";
  div.textContent = msg;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
}

// Structured hint card ------------------------------------------------------
function addHintBubble(hint, classification, content) {
  const div = document.createElement("div");
  div.className = "bubble tutor hint-card";

  const cls = classification ? classification : "";
  if (cls === "knowledge_gap") {
    if (hint.explanation) addField(div, "Concept", hint.explanation);
    if (hint.formula) addField(div, "Formula", hint.formula);
    if (hint.example) addField(div, "Example", hint.example);
  } else if (cls === "misapplication") {
    if (hint.mistake) addField(div, "Mistake", hint.mistake);
    if (hint.reason) addField(div, "Reason", hint.reason);
    if (hint.application_hint) addField(div, "Hint", hint.application_hint);
    if (hint.formula) addField(div, "Formula", hint.formula);
  } else if (cls === "on_track") {
    if (hint.confirmation) addField(div, "Confirmation", hint.confirmation);
    if (hint.next_step_hint) addField(div, "Next step", hint.next_step_hint);
  } else if (cls === "answer_check") {
    const status = hint.answer_status || "partial";
    const label = status === "correct" ? "Correct"
                : status === "incorrect" ? "Incorrect"
                : "Partial";
    addField(div, label, hint.answer_value ? `Your answer: ${hint.answer_value}` : "");
    if (hint.method_feedback) addField(div, "Method", hint.method_feedback);
    if (hint.mistake) addField(div, "Mistake", hint.mistake);
    if (hint.reason) addField(div, "Reason", hint.reason);
    if (hint.application_hint) addField(div, "Hint", hint.application_hint);
    if (status === "correct") div.classList.add("correct");
  } else if (cls === "incorrect_answer") {
    if (hint.mistake) addField(div, "Mistake", hint.mistake);
    if (hint.reason) addField(div, "Reason", hint.reason);
    if (hint.application_hint) addField(div, "Hint", hint.application_hint);
  } else if (cls === "solved") {
    if (hint.confirmation) addField(div, "", hint.confirmation);
    div.classList.add("correct");
  } else if (cls === "meta") {
    if (hint.meta_response) addField(div, "", hint.meta_response);
  } else {
    // unknown — just show the explanation
    if (hint.explanation) addField(div, "", hint.explanation);
  }

  // Source citation (when the hint was grounded in a reference chunk).
  if (hint.source_title) addSource(div, hint.source_title, hint.source_url);

  // If nothing was filled, fall back to a plain tutor bubble using the
  // backend-supplied `content` (which summarize_hint always makes non-empty).
  if (!div.children.length) {
    addBubble("tutor", content || hint.explanation || "");
    return;
  }

  messages.appendChild(div);
  renderMath(div);
  messages.scrollTop = messages.scrollHeight;
}

function addField(parent, label, value) {
  const field = document.createElement("div");
  field.className = "hint-field";
  if (label) {
    const lab = document.createElement("div");
    lab.className = "hint-label";
    lab.textContent = label;
    field.appendChild(lab);
  }
  const val = document.createElement("div");
  val.className = "hint-value";
  val.textContent = value;
  field.appendChild(val);
  parent.appendChild(field);
}

function addSource(parent, title, url) {
  const src = document.createElement("div");
  src.className = "hint-source";
  const label = document.createElement("span");
  label.textContent = "Source: ";
  src.appendChild(label);
  if (url) {
    const a = document.createElement("a");
    a.href = url;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.textContent = title;
    src.appendChild(a);
  } else {
    const t = document.createElement("span");
    t.textContent = title;
    src.appendChild(t);
  }
  parent.appendChild(src);
}

function renderMath(el) {
  if (typeof window.renderMathInElement === "function") {
    window.renderMathInElement(el, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false },
        { left: "\\[", right: "\\]", display: true },
      ],
      throwOnError: false,
    });
  }
}

function autoSize() {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
}

chatInput.addEventListener("input", autoSize);
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});
