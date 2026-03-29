# 🎬 AI Film Engine V4 — Series Edition

An end-to-end AI-powered TV series production pipeline built with **Google Gemini** and **Gradio**.

From a one-line concept to a fully shot-decomposed, cinematography-annotated, image-prompt-ready storyboard — across an entire multi-episode series.

---

## 🗺️ Pipeline

```
Series Concept
    └─▶ Series Bible (12 episodes outlined)
            └─▶ Character Bible (visual DNA per character)
                    └─▶ Episode Script (one episode at a time)
                            └─▶ Screenplay (scene-by-scene)
                                    └─▶ Storyboard (genre-aware shot decomposition)
                                            └─▶ Shot Editor (cine params + image prompts)
```

---

## 🚀 Setup

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/ai-film-engine.git
cd ai-film-engine
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
```

Open `.env` and fill in your key:

```env
MODEL_PROVIDER=gemini
GOOGLE_API_KEY=your_actual_key_here
GEMINI_MODEL=gemini-2.5-flash
```

`.env` is in `.gitignore` — it will never be committed. `.env.example` is safe and committed as a template.

### 3. Run

```bash
python app.py
```

---

## 🔄 Switching Providers Later

All model/provider config is in `.env` only — no code changes needed for supported providers.

**Switch to OpenAI:**
```env
MODEL_PROVIDER=openai
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o
```
Then uncomment `openai>=1.0.0` in `requirements.txt` and run `pip install -r requirements.txt`.

**Switch to DeepSeek:**
```env
MODEL_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_MODEL=deepseek-chat
```
Same — uncomment `openai>=1.0.0` (DeepSeek uses the OpenAI-compatible SDK).

---

## 🎭 Genre Shot Budgets

| Genre | Shots/Scene | Avg Shot | Style |
|---|---|---|---|
| Romance | 8–30 | 5s | Slow, deliberate, close on faces |
| Drama | 15–50 | 4s | Naturalistic, reactive, medium coverage |
| Thriller | 30–80 | 3s | Tight, tense, rapid cutting |
| Sci-Fi | 20–60 | 4s | Wide establishing, detail inserts |
| Horror | 25–70 | 3s | Slow dread building to fast chaos |
| Comedy | 10–40 | 5s | Wide reaction shots, timing-driven |
| Action | 60–150 | 2s | Kinetic, overlapping, rapid intercutting |
| Mystery | 20–55 | 4s | Observational, detail-focused |
| Animation | 15–60 | 3s | Expressive, exaggerated, pose-to-pose |

---

## 📁 Project Structure

```
ai-film-engine/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── .env.example        # Config template — copy to .env and fill in keys
├── .gitignore          # .env and user data excluded
└── README.md
```

---

## 🛠️ Tech Stack

- **[Google Gemini 2.5 Flash](https://aistudio.google.com)** — default LLM
- **[Gradio](https://gradio.app)** — UI framework
- **[python-dotenv](https://pypi.org/project/python-dotenv/)** — env config

---

## 📄 License

MIT