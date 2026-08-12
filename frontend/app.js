const API = "";

const els = {
  topics: document.getElementById("topics"),
  pickPanel: document.getElementById("pickPanel"),
  talkPanel: document.getElementById("talkPanel"),
  stage: document.getElementById("stage"),
  keywords: document.getElementById("keywords"),
  partial: document.getElementById("partial"),
  latency: document.getElementById("latency"),
  callBtn: document.getElementById("callBtn"),
  callLabel: document.getElementById("callLabel"),
  callState: document.getElementById("callState"),
  callPulse: document.getElementById("callPulse"),
  status: document.getElementById("status"),
  topicLabel: document.getElementById("topicLabel"),
  sessionMeta: document.getElementById("sessionMeta"),
  backBtn: document.getElementById("backBtn"),
  player: document.getElementById("player"),
};

let sessionId = null;
let ws = null;
let mediaStream = null;
let audioCtx = null;
let processor = null;
let inCall = false;
let coachPlaying = false;
let playToken = 0;
let objectUrl = null;
let wantCall = false; // user wants call to stay up
let pingTimer = null;
let reconnectTimer = null;

function showLatency(lat) {
  if (!els.latency || !lat) return;
  const wait = lat.wait_ms ?? lat.total_ms ?? "—";
  const llm = lat.llm_ms ?? "—";
  const tts = lat.tts_ms ?? "—";
  const ttfb = lat.ttfb_ms ?? "—";
  const mode = lat.mode || (lat.speculative ? "spec" : "fresh");
  els.latency.textContent =
    `Latency  hear@${wait}ms · llm ${llm}ms · ttfb ${ttfb}ms · tts ${tts}ms · ${mode}`;
}

let pcmPlayer = null;

function stopPcmStream() {
  if (pcmPlayer) {
    try { pcmPlayer.stop(); } catch {}
    pcmPlayer = null;
  }
}

function startPcmStream(sampleRate = 16000) {
  stopPcmStream();
  playToken += 1;
  try {
    els.player.pause();
    els.player.removeAttribute("src");
    els.player.load();
  } catch {}
  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
    objectUrl = null;
  }
  const ctx = new AudioContext({ sampleRate });
  let nextTime = 0;
  let stopped = false;
  pcmPlayer = {
    push(arrayBuffer) {
      if (stopped || !arrayBuffer || arrayBuffer.byteLength < 2) return;
      const len = arrayBuffer.byteLength - (arrayBuffer.byteLength % 2);
      const i16 = new Int16Array(arrayBuffer.slice(0, len));
      const f32 = new Float32Array(i16.length);
      for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 0x8000;
      const buf = ctx.createBuffer(1, f32.length, sampleRate);
      buf.copyToChannel(f32, 0);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(ctx.destination);
      const now = ctx.currentTime;
      if (nextTime < now + 0.02) nextTime = now + 0.02;
      src.start(nextTime);
      nextTime += buf.duration;
    },
    stop() {
      stopped = true;
      try { ctx.close(); } catch {}
    },
  };
  if (ctx.state === "suspended") ctx.resume().catch(() => {});
  coachPlaying = true;
  setCallUi("Coach speaking… (you can interrupt)");
  setStatus("Coach speaking — streaming…");
  return pcmPlayer;
}

function setStatus(text) {
  els.status.textContent = text;
}

function setCallUi(state) {
  els.callState.textContent = state;
  els.callPulse.classList.toggle("on", inCall);
  els.callBtn.classList.toggle("recording", inCall);
  els.callLabel.textContent = inCall ? "End Call" : "Join Call";
}

function addBubble(role, text) {
  const empty = els.stage.querySelector(".hint");
  if (empty) empty.remove();
  const div = document.createElement("div");
  div.className = `bubble ${role === "assistant" ? "coach" : "you"}`;
  div.innerHTML = `<small>${role === "assistant" ? "Coach" : "You"}</small>${escapeHtml(text || "…")}`;
  els.stage.appendChild(div);
  els.stage.scrollTop = els.stage.scrollHeight;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function showKeywords(list) {
  els.keywords.textContent = list?.length ? `TF-IDF focus: ${list.join(", ")}` : "";
}

function stopCoachAudio() {
  playToken += 1;
  coachPlaying = false;
  stopPcmStream();
  try {
    els.player.pause();
    els.player.removeAttribute("src");
    els.player.load();
  } catch {}
  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
    objectUrl = null;
  }
  if (inCall) {
    setCallUi("Listening — talk naturally");
    setStatus("Listening… interrupt anytime.");
  }
}

function b64ToBlob(b64, mime = "audio/wav") {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}

async function playCoachAudio({ b64, url, blob }) {
  // Prepare next source BEFORE stopping old audio (avoid revoking the new URL)
  let nextUrl = url || "";
  let nextObject = null;
  if (blob) {
    nextObject = URL.createObjectURL(blob);
    nextUrl = nextObject;
  } else if (b64) {
    nextObject = URL.createObjectURL(b64ToBlob(b64));
    nextUrl = nextObject;
  }

  playToken += 1;
  const token = playToken;
  coachPlaying = true;
  try {
    els.player.pause();
  } catch {}
  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
    objectUrl = null;
  }
  objectUrl = nextObject;

  if (!nextUrl) {
    coachPlaying = false;
    return;
  }

  setCallUi("Coach speaking… (you can interrupt)");
  setStatus("Coach speaking — bol ke interrupt kar sakte ho.");
  els.player.src = nextUrl;
  try {
    await els.player.play();
    await new Promise((resolve) => {
      const done = () => resolve();
      els.player.onended = done;
      els.player.onerror = done;
      els.player.onpause = () => {
        if (token !== playToken) resolve();
      };
    });
  } catch {
    setStatus("Tap once if audio is blocked.");
  } finally {
    if (token === playToken) {
      coachPlaying = false;
      if (inCall) {
        setCallUi("Listening — talk naturally");
        setStatus("Live call on. Speak anytime.");
      }
    }
  }
}

async function loadTopics() {
  setStatus("Loading topics…");
  // Instant paint from cache if available
  try {
    const cached = sessionStorage.getItem("ai_talk_topics");
    if (cached) {
      renderTopics(JSON.parse(cached));
      setStatus("Pick a topic to begin.");
    }
  } catch {}

  const res = await fetch(`${API}/api/topics`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to load topics");
  const data = await res.json();
  try {
    sessionStorage.setItem("ai_talk_topics", JSON.stringify(data.topics));
  } catch {}
  renderTopics(data.topics);
  setStatus("Pick a topic to begin.");
}

function renderTopics(topics) {
  els.topics.innerHTML = "";
  for (const topic of topics) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "topic-btn";
    btn.innerHTML = `<strong>${escapeHtml(topic.title)}</strong><span>${escapeHtml(topic.prompt)}</span>`;
    btn.addEventListener("click", () => startSession(topic.id));
    els.topics.appendChild(btn);
  }
}

async function startSession(topicId) {
  await endCall(false);
  setStatus("Starting session…");
  const res = await fetch(`${API}/api/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ topic_id: topicId }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    setStatus(err.detail || "Could not start session");
    return;
  }
  const data = await res.json();
  sessionId = data.session_id;
  els.pickPanel.classList.add("hidden");
  els.talkPanel.classList.remove("hidden");
  els.topicLabel.textContent = data.topic.title;
  els.sessionMeta.textContent = `Session ${data.session_id}`;
  els.stage.innerHTML = "";
  els.partial.textContent = "";
  addBubble("assistant", data.coach_text);
  showKeywords(data.keywords);
  els.callBtn.disabled = false;
  setCallUi("Coach intro…");
  await playCoachAudio({ b64: data.coach_audio_b64 });
  await joinCall();
}

function wsUrl(id) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/ws/call/${id}`;
}

function floatTo16BitPCM(float32) {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out.buffer;
}

function downsample(buffer, inRate, outRate) {
  if (inRate === outRate) return buffer;
  const ratio = inRate / outRate;
  const newLen = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLen);
  for (let i = 0; i < newLen; i++) {
    result[i] = buffer[Math.floor(i * ratio)] || 0;
  }
  return result;
}

async function startMicStream() {
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      channelCount: 1,
    },
  });
  audioCtx = new AudioContext();
  const source = audioCtx.createMediaStreamSource(mediaStream);
  // Smaller buffer = lower mic latency to Deepgram
  processor = audioCtx.createScriptProcessor(2048, 1, 1);
  processor.onaudioprocess = (event) => {
    if (!inCall || !ws || ws.readyState !== WebSocket.OPEN) return;
    // Mute uplink while coach speaks — stops echo from cancelling TTS
    if (coachPlaying) return;
    const input = event.inputBuffer.getChannelData(0);
    const down = downsample(input, audioCtx.sampleRate, 16000);
    ws.send(floatTo16BitPCM(down));
  };
  source.connect(processor);
  const mute = audioCtx.createGain();
  mute.gain.value = 0;
  processor.connect(mute);
  mute.connect(audioCtx.destination);
}

let pendingBinaryAudio = null;

async function joinCall() {
  if (!sessionId) return;
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return;
  }
  wantCall = true;
  setStatus("Connecting live call…");
  try {
    if (!mediaStream) await startMicStream();
    if (audioCtx?.state === "suspended") {
      try { await audioCtx.resume(); } catch {}
    }
    ws = new WebSocket(wsUrl(sessionId));
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      inCall = true;
      setCallUi("Listening — talk naturally");
      setStatus("Call connected — line open. Speak anytime.");
      if (pingTimer) clearInterval(pingTimer);
      pingTimer = setInterval(() => {
        if (ws?.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "ping" }));
        }
      }, 5000);
    };

    ws.onmessage = async (event) => {
      // Binary: either one-shot WAV after audio_binary_next, or PCM stream chunks
      if (event.data instanceof ArrayBuffer) {
        if (pcmPlayer) {
          pcmPlayer.push(event.data);
          return;
        }
        if (pendingBinaryAudio) {
          const meta = pendingBinaryAudio;
          pendingBinaryAudio = null;
          const blob = new Blob([event.data], { type: "audio/wav" });
          addBubble("assistant", meta.coach_text);
          showKeywords(meta.keywords);
          if (meta.latency) showLatency(meta.latency);
          await playCoachAudio({ blob });
        }
        return;
      }

      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }

      if (msg.type === "barge_in") {
        stopCoachAudio();
        setStatus("Interrupted — listening to you…");
      }

      if (msg.type === "partial" && msg.text) {
        // Do NOT stop coach audio on partials — speaker echo was killing TTS
        els.partial.textContent = msg.is_final ? "" : `Hearing: ${msg.text}`;
      }

      if (msg.type === "thinking") {
        setCallUi("Thinking…");
        setStatus("Coach preparing a short reply…");
      }

      if (msg.type === "prep") {
        setStatus("Prefetching reply while you speak…");
      }

      if (msg.type === "prep_ready" && msg.latency) {
        showLatency({ ...msg.latency, wait_ms: "…", speculative: true, mode: "spec" });
        setStatus("Reply ready — waiting for you to finish…");
      }

      if (msg.type === "user_final") {
        els.partial.textContent = "";
        addBubble("user", msg.text);
      }

      if (msg.type === "coach_turn") {
        if (msg.latency) showLatency(msg.latency);
        if (msg.stream && msg.audio_format === "pcm_s16le") {
          addBubble("assistant", msg.coach_text);
          showKeywords(msg.keywords);
          startPcmStream(msg.sample_rate || 16000);
          return;
        }
        if (msg.audio_binary_next) {
          pendingBinaryAudio = msg;
          return;
        }
        addBubble("assistant", msg.coach_text);
        showKeywords(msg.keywords);
        await playCoachAudio({ b64: msg.coach_audio_b64 });
      }

      if (msg.type === "coach_audio_end") {
        if (msg.latency) showLatency(msg.latency);
        // Keep mic muted briefly so tail audio isn't echoed into STT
        setTimeout(() => {
          coachPlaying = false;
          pcmPlayer = null;
          if (inCall) {
            setCallUi("Listening — talk naturally");
            setStatus("Live call on. Speak anytime.");
          }
        }, 1200);
      }

      if (msg.type === "error") {
        setStatus(msg.detail || "Call error");
        setCallUi(inCall ? "Listening — talk naturally" : "Call error");
      }
      if (msg.type === "info") setStatus(msg.detail || "Call info");
      if (msg.type === "call_ready") setStatus(msg.message || "Call ready");
    };

    ws.onclose = () => {
      inCall = false;
      if (pingTimer) {
        clearInterval(pingTimer);
        pingTimer = null;
      }
      if (wantCall && sessionId) {
        setCallUi("Reconnecting…");
        setStatus("Line dropped — reconnecting…");
        if (reconnectTimer) clearTimeout(reconnectTimer);
        reconnectTimer = setTimeout(() => {
          if (wantCall) joinCall();
        }, 1000);
      } else {
        setCallUi("Call ended — tap Join Call to reconnect");
        setStatus("Call ended.");
      }
    };

    ws.onerror = () => setStatus("WebSocket error — retrying…");
  } catch (err) {
    setStatus(err.message || "Mic / call failed");
    if (wantCall) {
      if (reconnectTimer) clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(() => joinCall(), 1000);
    }
  }
}

async function endCall(notify = true) {
  wantCall = false;
  inCall = false;
  stopCoachAudio();
  if (pingTimer) {
    clearInterval(pingTimer);
    pingTimer = null;
  }
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  if (ws && ws.readyState === WebSocket.OPEN) {
    if (notify) ws.send(JSON.stringify({ type: "end_call" }));
    ws.close();
  }
  ws = null;
  if (processor) {
    try { processor.disconnect(); } catch {}
    processor = null;
  }
  if (audioCtx) {
    try { await audioCtx.close(); } catch {}
    audioCtx = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  setCallUi("Ready to join call");
}

async function resetToTopics() {
  await endCall(true);
  sessionId = null;
  els.talkPanel.classList.add("hidden");
  els.pickPanel.classList.remove("hidden");
  els.callBtn.disabled = true;
  els.partial.textContent = "";
  setStatus("Pick a topic to begin.");
}

els.callBtn.addEventListener("click", async () => {
  if (inCall) {
    await endCall(true);
    setStatus("Call ended. Tap Join Call to continue this topic.");
    return;
  }
  await joinCall();
});

els.backBtn.addEventListener("click", resetToTopics);

loadTopics().catch((err) => setStatus(err.message || "API offline — start uvicorn"));
