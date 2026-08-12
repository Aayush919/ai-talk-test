# AI Talk — Realtime English Coach

Live voice English practice: pick a topic, join a call, speak naturally.  
**Deepgram** (STT + TTS) + **Groq** (coach replies) + **MongoDB** (transcripts).

---

## How it works

```text
Browser mic (PCM chunks ~every 100ms)
    → WebSocket /ws/call/{session_id}
    → Deepgram live STT  (partial → final transcript)
    → Groq LLM           (short coach reply text)
    → Deepgram TTS       (stream PCM audio chunks)
    → Browser speaker
```

| Piece | Job |
|--------|-----|
| **Frontend** (`frontend/`) | Topics UI, Join/End Call, mic stream, play coach audio, latency line |
| **API** (`api/app.py`) | FastAPI: topics, create session, static UI, WebSocket entry |
| **Live bridge** (`api/call_ws.py`) | Mic ↔ Deepgram STT, speculative prep, Groq+TTS turn, barge-in rules |
| **Coach** (`core/coach_service.py`) | Draft/commit replies, TF-IDF keywords, async Mongo writes |
| **Groq** (`wrappers/groq_llm.py`) | Short English coach text |
| **Deepgram TTS** (`wrappers/deepgram_tts.py`) | Streaming speak (Aura), `DEEPGRAM_TTS_MODEL` / `SPEED` |
| **Topics** (`data/topics.json`) | Practice scenarios + starter lines |
| **Mongo** | Session + message text (no local audio required on live path) |

**Latency line in UI:** `hear@` ≈ time until you hear voice; `llm` / `ttfb` / `tts` break down Groq vs Deepgram Speak.

While the coach is speaking, mic uplink is muted briefly so speaker echo does not cancel TTS.

---

## Quick start

### 1. Clone

```bash
git clone <your-repo-url>
cd ai-talk
```

### 2. Create virtualenv & install

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Secrets — files in `.gitignore` (how users get them)

GitHub pe **secrets nahi jaate**. User khud banata hai:

| Ignored (not on GitHub) | How user gets it |
|-------------------------|------------------|
| `.env` | Copy from **`.env.example`** (this *is* in the repo) |
| `.venv/` | `python -m venv .venv` + `pip install -r requirements.txt` |
| `sessions/` | App create karti hai runtime pe (optional/local) |
| `*.wav` / audio dumps | Generated locally; live path doesn’t need them |

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Phir `.env` mein apni keys bharo:

- **GROQ_API_KEYS** — [console.groq.com](https://console.groq.com)
- **DEEPGRAM_API_KEYS** — [console.deepgram.com](https://console.deepgram.com)
- **MONGODB_URI** — Atlas connection string

Optional TTS tweaks:

```env
DEEPGRAM_TTS_MODEL=aura-helios-en
DEEPGRAM_TTS_SPEED=1.25
```

### 4. Run server

```bash
# Windows
.\.venv\Scripts\python run_server.py

# or
python run_server.py
```

Open: **http://127.0.0.1:8000**

- Pick a topic → coach intro → **Join Call** → speak  
- Keep that terminal open (no `--reload` for call testing)  
- If port 8000 busy: find PID with `netstat -ano | findstr :8000`, then `taskkill /PID <pid> /F`

Or double-click `start_server.bat` on Windows.

---

## Project layout

```text
ai-talk/
├── api/           # FastAPI + live WebSocket call
├── core/          # config, session, coach, prompts, TF-IDF, topics
├── wrappers/      # Groq, Deepgram STT/TTS, Mongo
├── frontend/      # HTML / CSS / JS call UI
├── data/          # topics.json
├── run_server.py  # uvicorn entry (reload OFF)
├── .env.example   # template for secrets
└── requirements.txt
```

---

## GitHub checklist

1. Confirm `.env` is **not** committed (listed in `.gitignore`)
2. Push code + `.env.example` + this README
3. Contributors: clone → copy `.env.example` → `.env` → fill keys → `pip install` → `run_server.py`

**Never** commit real API keys, Mongo passwords, or Cloudinary secrets.

---

## Notes

- Live path is **realtime WebSocket** — not hold-to-talk upload.
- Deepgram **Listen** (STT) is fast; **Speak** (TTS) is the usual latency bottleneck (especially from India).
- Fastest Aura-1 voice in our local benches: `aura-helios-en`.
