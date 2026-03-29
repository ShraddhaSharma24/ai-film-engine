# ============================================================
# 🎬 AI FILM ENGINE V4
# Series → Episode → Scene → Shot
# Character Bible + Genre-Aware Shot Budgets + Consistency Engine
# Google Colab + Gradio
# ============================================================

# ── Install ──────────────────────────────────────────────────
# !pip install --quiet gradio google-genai

import os, json, time
import gradio as gr
from datetime import datetime

from google import genai
from google.genai import types

# ============================================================
# 🔑 API SETUP
# ============================================================
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
MODEL = "gemini-2.5-flash"

# ============================================================
# 🎭 GENRE SHOT BUDGETS
# Controls decompose behaviour per genre
# ============================================================
GENRE_SHOT_BUDGETS = {
    "Romance":    {"min_shots": 8,  "max_shots": 30,  "avg_shot_sec": 5, "style": "slow, deliberate, close on faces"},
    "Drama":      {"min_shots": 15, "max_shots": 50,  "avg_shot_sec": 4, "style": "naturalistic, reactive, medium coverage"},
    "Thriller":   {"min_shots": 30, "max_shots": 80,  "avg_shot_sec": 3, "style": "tight, tense, rapid cutting, motivated movement"},
    "Sci-Fi":     {"min_shots": 20, "max_shots": 60,  "avg_shot_sec": 4, "style": "wide establishing, detail inserts, awe-inspiring"},
    "Horror":     {"min_shots": 25, "max_shots": 70,  "avg_shot_sec": 3, "style": "slow dread building to fast chaos, POV heavy"},
    "Comedy":     {"min_shots": 10, "max_shots": 40,  "avg_shot_sec": 5, "style": "wide reaction shots, timing-driven, fluid"},
    "Action":     {"min_shots": 60, "max_shots": 150, "avg_shot_sec": 2, "style": "kinetic, overlapping action, rapid intercutting"},
    "Mystery":    {"min_shots": 20, "max_shots": 55,  "avg_shot_sec": 4, "style": "observational, detail-focused, withheld information"},
    "Animation":  {"min_shots": 15, "max_shots": 60,  "avg_shot_sec": 3, "style": "expressive, exaggerated, pose-to-pose"},
}

# ============================================================
# 💾 PROJECT HISTORY — undo per stage
# ============================================================
class History:
    def __init__(self):
        self.data = {
            "series_bible": [], "script": [], "screenplay": [],
            "storyboard": [], "character_bible": []
        }
        self.MAX = 20

    def save(self, stage, content):
        self.data[stage].append(content)
        if len(self.data[stage]) > self.MAX:
            self.data[stage].pop(0)

    def undo(self, stage):
        if len(self.data[stage]) > 1:
            self.data[stage].pop()
            return self.data[stage][-1]
        return None

history = History()

# ============================================================
# 🤖 GEMINI CALL
# ============================================================
def gemini(system_prompt, payload, retries=3):
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=system_prompt + "\n\nINPUT:\n" + json.dumps(payload, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    temperature=0.6,
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text)
        except Exception as e:
            if attempt == retries - 1:
                raise gr.Error(f"Gemini call failed after {retries} attempts: {e}")
            time.sleep(2 ** attempt)

# ============================================================
# 🧩 JSON HELPERS
# ============================================================
def to_json(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False)

def from_json(text, label="Input"):
    if not text or not text.strip():
        raise gr.Error(f"{label} is empty.")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise gr.Error(f"{label} — invalid JSON: {e.msg}")

def require(data, keys, label=""):
    missing = [k for k in keys if k not in data]
    if missing:
        raise gr.Error(f"{label} missing fields: {', '.join(missing)}")

# ============================================================
# 📜 PROMPTS
# ============================================================

SERIES_BIBLE_PROMPT = """
You are a TV series creator and showrunner.
Generate a complete SERIES BIBLE for a multi-episode series in strict JSON.

Schema:
{
  "series_title": "",
  "logline": "",
  "genre": "",
  "tone": "",
  "target_episodes": 12,
  "episode_runtime_minutes": 22,
  "world_setting": "",
  "time_period": "",
  "central_conflict": "",
  "themes": [],
  "visual_style": "",
  "color_palette_description": "",
  "series_arc": {
    "act_1_episodes": "1-4",
    "act_2_episodes": "5-9",
    "act_3_episodes": "10-12",
    "summary": ""
  },
  "episode_outlines": [
    {
      "episode_number": 1,
      "title": "",
      "logline": "",
      "cold_open": "",
      "act_breaks": ["", "", ""],
      "key_scenes": [],
      "character_focus": [],
      "emotional_arc": "",
      "runtime_minutes": 22
    }
  ]
}
Generate ALL 12 episode outlines.
STRICT JSON ONLY. No markdown, no explanation.
"""

CHARACTER_BIBLE_PROMPT = """
You are a character designer and casting director for a TV series.
Given the series bible, generate a comprehensive CHARACTER BIBLE with detailed visual DNA for every character.
This will be injected into every image generation prompt to ensure visual consistency across all 12 episodes.

Schema:
{
  "characters": [
    {
      "name": "",
      "role": "protagonist|antagonist|supporting|recurring|guest",
      "age": 0,
      "gender": "",
      "ethnicity": "",
      "build": "",
      "height": "",
      "hair": {
        "color": "",
        "style": "",
        "length": ""
      },
      "eyes": {
        "color": "",
        "shape": ""
      },
      "face": {
        "shape": "",
        "distinctive_features": ""
      },
      "skin_tone": "",
      "signature_wardrobe": {
        "casual": "",
        "formal": "",
        "signature_item": ""
      },
      "color_palette": [],
      "body_language": "",
      "personality": "",
      "character_arc": "",
      "episode_first_appearance": 1,
      "reference_prompt_seed": "",
      "consistency_notes": ""
    }
  ]
}

The 'reference_prompt_seed' should be a dense visual description (40-60 words) of the character that will be prepended to every image prompt featuring this character. It must be specific enough that an image generation model would produce consistent results across many prompts.

STRICT JSON ONLY. No markdown.
"""

EPISODE_SCRIPT_PROMPT = """
You are a professional TV screenwriter writing one episode of a series.
Using the series bible AND character bible for full context and consistency, write a detailed episode script in JSON.

You MUST maintain perfect character consistency with the character bible visual descriptions.
The episode runtime is approximately {runtime} minutes, which means approximately {scene_count} scenes.

Schema:
{{
  "episode_number": 0,
  "title": "",
  "cold_open": "",
  "previously_on": "",
  "acts": [
    {{
      "act_number": 1,
      "scenes": [
        {{
          "scene_number": 1,
          "heading": "INT./EXT. LOCATION - DAY/NIGHT",
          "location": "",
          "time_of_day": "",
          "estimated_duration_seconds": 120,
          "characters_present": [],
          "action": "",
          "dialogue": [{{"character": "", "line": ""}}],
          "emotional_beat": "",
          "scene_type": "dialogue|action|montage|transition|cold_open|climax",
          "pacing": "slow|medium|fast"
        }}
      ]
    }}
  ],
  "tag_scene": ""
}}
STRICT JSON ONLY. No markdown.
"""

SCREENPLAY_PROMPT = """
You are a professional TV screenplay writer.
Convert the episode script JSON into a full detailed SCREENPLAY JSON.
Maintain strict character consistency with the provided character bible.

Schema:
{
  "scenes": [
    {
      "scene_number": 1,
      "heading": "INT./EXT. LOCATION - DAY/NIGHT",
      "location": "",
      "time_of_day": "",
      "estimated_duration_seconds": 120,
      "characters_present": [],
      "action": "",
      "dialogue": [{"character": "", "line": ""}],
      "emotional_beat": "",
      "scene_type": "dialogue|action|montage|transition|cold_open|climax",
      "pacing": "slow|medium|fast"
    }
  ]
}
STRICT JSON ONLY. No markdown.
"""

DECOMPOSE_PROMPT_TEMPLATE = """
You are a film director breaking a TV scene into individual cinematic shots.
Genre: {genre} — Shot style: {style}
Target shots for this scene: {min_shots}–{max_shots} shots (avg {avg_sec}s each).
Scene type: {scene_type} | Pacing: {pacing} | Duration: {duration}s

Rules:
- Romance/Drama: fewer, longer shots — favor close-ups and silences
- Action/Thriller: many short shots — rapid coverage, overlapping motion
- Each shot must capture every micro-expression, gesture, body language, and dialogue beat
- Cover every line of dialogue across multiple reaction shots

Schema:
{{
  "shots": [
    {{
      "shot_id": 1,
      "timecode_start": "00:00",
      "timecode_end": "00:04",
      "duration_seconds": 4,
      "action": "",
      "gesture": "",
      "facial_expression": "",
      "body_language": "",
      "dialogue": "",
      "emotion": "",
      "energy": "low|medium|high"
    }}
  ]
}}
STRICT JSON ONLY. No markdown.
"""

CINE_SUGGEST_PROMPT = """
You are a cinematographer for a TV series.
Given a shot's narrative content — action, emotion, gesture, body language, dialogue, genre and scene type —
suggest the most cinematically appropriate technical parameters.
Your suggestions must SERVE the emotion, story, and genre conventions.

Return ONLY this JSON:
{
  "shot_size": "Extreme Wide Shot|Wide Shot|Full Shot|Medium Shot|Medium Close-Up|Close-Up|Extreme Close-Up",
  "vertical_angle": "Eye-Level|Low Angle|High Angle|Dutch Angle",
  "horizontal_angle": "Frontal|3/4 View|Profile|3/4 Rear|Rear View",
  "camera_movement": "Static|Panning|Tilting|Tracking|Dolly In|Dolly Out|Crane|Handheld",
  "stability": "Tripod|Steadicam|Handheld|Shaky Cam",
  "focal_length": "Wide Angle <35mm|Normal 50mm|Telephoto 85mm|Telephoto 135mm+",
  "depth_of_field": "Deep Focus|Shallow|Razor-Thin",
  "bokeh": "Creamy|Busy|None",
  "lens_artifact": "None|Lens Flare|Chromatic Aberration|Vignette",
  "key_light_direction": "Frontal|Rembrandt 3/4|Side|Backlight|Top|Under",
  "light_quality": "Hard|Soft",
  "contrast": "High-Key|Low-Key|Chiaroscuro",
  "atmosphere": "Clean|Haze|Fog|Volumetric",
  "color_temp": "Warm Tungsten|Neutral Daylight|Cool|Mixed",
  "color_harmony": "Monochromatic|Analogous|Complementary Teal-Orange|Desaturated",
  "film_emulation": "Kodak Portra|Kodak Vision3|Fujifilm|Cinestill 800T|Bleach Bypass|Digital Clean",
  "aspect_ratio": "1.33:1|1.85:1|2.39:1 Anamorphic",
  "framing": "Rule of Thirds|Center|Frame-within-Frame|Negative Space",
  "reasoning": "one sentence why these choices serve this shot"
}
STRICT JSON ONLY.
"""

SYNTHESIZE_PROMPT = """
You are a prompt engineer for cinematic image generation of a TV series.
Given a shot's full data — narrative, cinematography, AND character visual DNA from the character bible —
write a single coherent vivid image generation prompt that guarantees character visual consistency.

CRITICAL: For each character present in the shot, you MUST prepend their full reference_prompt_seed
from the character bible at the start of the character description in the prompt. This ensures
visual consistency across all episodes.

DO NOT concatenate sections. Write ONE flowing description.
Start with character visuals (using their seeds), then composition, weave in lighting and mood,
end with technical specs. 60–120 words.

Return ONLY:
{ "image_prompt": "..." }
STRICT JSON ONLY.
"""

REWRITE_PROMPT = """
You are a professional TV editor.
Rewrite the given JSON according to the instruction.
Preserve the EXACT same JSON structure and character consistency.
Modify only what the instruction asks for.
STRICT JSON ONLY. No markdown.
"""

# ============================================================
# 🎬 STAGE 0: Series Bible + Character Bible
# ============================================================
def generate_series_bible(concept, genre, tone, num_episodes):
    if not concept.strip():
        raise gr.Error("Enter a series concept.")
    payload = {
        "concept": concept, "genre": genre, "tone": tone,
        "target_episodes": int(num_episodes), "episode_runtime_minutes": 22
    }
    result = to_json(gemini(SERIES_BIBLE_PROMPT, payload))
    history.save("series_bible", result)
    return result, f"✅ Series bible generated for {num_episodes} episodes."

def generate_character_bible(series_bible_text):
    series_bible = from_json(series_bible_text, "Series Bible")
    require(series_bible, ["series_title", "episode_outlines"], "Series Bible")
    result = to_json(gemini(CHARACTER_BIBLE_PROMPT, series_bible))
    history.save("character_bible", result)
    char_data = json.loads(result)
    chars = char_data.get("characters", [])
    char_names = [c["name"] for c in chars]
    return result, f"✅ Character bible created: {len(chars)} characters — {', '.join(char_names)}"

def rewrite_series_bible(text, instruction):
    if not instruction.strip(): raise gr.Error("Enter a rewrite instruction.")
    result = to_json(gemini(REWRITE_PROMPT + f"\n\nInstruction: {instruction}", from_json(text, "Series Bible")))
    history.save("series_bible", result)
    return result

def rewrite_character_bible(text, instruction):
    if not instruction.strip(): raise gr.Error("Enter a rewrite instruction.")
    result = to_json(gemini(REWRITE_PROMPT + f"\n\nInstruction: {instruction}", from_json(text, "Character Bible")))
    history.save("character_bible", result)
    return result

def undo_series_bible():
    r = history.undo("series_bible")
    if not r: raise gr.Error("Nothing to undo.")
    return r

def undo_character_bible():
    r = history.undo("character_bible")
    if not r: raise gr.Error("Nothing to undo.")
    return r

def get_episode_choices(series_bible_text):
    """Return list of episode numbers from the series bible."""
    try:
        bible = from_json(series_bible_text, "Series Bible")
        outlines = bible.get("episode_outlines", [])
        return gr.update(choices=[e["episode_number"] for e in outlines],
                         value=1 if outlines else None)
    except Exception:
        return gr.update(choices=[], value=None)

# ============================================================
# 🎬 STAGE 1: Episode Script
# ============================================================
def generate_episode_script(series_bible_text, character_bible_text, episode_number):
    if episode_number is None:
        raise gr.Error("Select an episode number.")
    series_bible = from_json(series_bible_text, "Series Bible")
    char_bible = from_json(character_bible_text, "Character Bible")
    genre = series_bible.get("genre", "Drama")
    runtime = series_bible.get("episode_runtime_minutes", 22)
    avg_scene_min = 3 if genre in ["Action", "Thriller"] else 5
    scene_count = max(4, runtime // avg_scene_min)

    ep_outline = next(
        (e for e in series_bible.get("episode_outlines", []) if e["episode_number"] == int(episode_number)),
        None
    )
    if not ep_outline:
        raise gr.Error(f"Episode {episode_number} not found in series bible.")

    prompt = EPISODE_SCRIPT_PROMPT.format(
        runtime=runtime, scene_count=scene_count
    )
    payload = {
        "series_bible": series_bible,
        "character_bible": char_bible,
        "episode_outline": ep_outline,
        "episode_number": int(episode_number)
    }
    result = to_json(gemini(prompt, payload))
    history.save("script", result)
    return result, f"✅ Episode {episode_number} script generated."

def rewrite_script(script_text, instruction):
    if not instruction.strip(): raise gr.Error("Enter a rewrite instruction.")
    result = to_json(gemini(REWRITE_PROMPT + f"\n\nInstruction: {instruction}", from_json(script_text, "Script")))
    history.save("script", result)
    return result

def undo_script():
    r = history.undo("script")
    if not r: raise gr.Error("Nothing to undo.")
    return r

# ============================================================
# 🎬 STAGE 2: Screenplay
# ============================================================
def generate_screenplay(script_text, character_bible_text):
    script = from_json(script_text, "Script")
    char_bible = from_json(character_bible_text, "Character Bible")
    payload = {"script": script, "character_bible": char_bible}
    result = to_json(gemini(SCREENPLAY_PROMPT, payload))
    history.save("screenplay", result)
    return result, "✅ Screenplay generated."

def rewrite_screenplay(screenplay_text, instruction):
    if not instruction.strip(): raise gr.Error("Enter a rewrite instruction.")
    result = to_json(gemini(REWRITE_PROMPT + f"\n\nInstruction: {instruction}", from_json(screenplay_text, "Screenplay")))
    history.save("screenplay", result)
    return result

def undo_screenplay():
    r = history.undo("screenplay")
    if not r: raise gr.Error("Nothing to undo.")
    return r

# ============================================================
# 🎬 STAGE 3: Storyboard — Genre-Aware Decomposition
# ============================================================
def _decompose_one_scene(scene, genre, char_bible_data):
    budget = GENRE_SHOT_BUDGETS.get(genre, GENRE_SHOT_BUDGETS["Drama"])
    scene_type = scene.get("scene_type", "dialogue")
    pacing = scene.get("pacing", "medium")
    duration = scene.get("estimated_duration_seconds", 120)

    base_min = budget["min_shots"]
    base_max = budget["max_shots"]
    if scene_type in ["action", "climax"]:
        base_min = int(base_min * 1.4)
        base_max = min(150, int(base_max * 1.4))
    elif scene_type in ["dialogue", "transition"]:
        base_min = max(3, int(base_min * 0.6))
        base_max = max(10, int(base_max * 0.6))

    prompt = DECOMPOSE_PROMPT_TEMPLATE.format(
        genre=genre,
        style=budget["style"],
        min_shots=base_min,
        max_shots=base_max,
        avg_sec=budget["avg_shot_sec"],
        scene_type=scene_type,
        pacing=pacing,
        duration=duration
    )

    decomposed = gemini(prompt, scene)
    shots = []
    for s in decomposed.get("shots", []):
        shots.append({
            "shot_id": s.get("shot_id", 1),
            "timecode_start": s.get("timecode_start", "00:00"),
            "timecode_end": s.get("timecode_end", "00:04"),
            "duration_seconds": s.get("duration_seconds", 4),
            "action": s.get("action", ""),
            "gesture": s.get("gesture", ""),
            "facial_expression": s.get("facial_expression", ""),
            "body_language": s.get("body_language", ""),
            "dialogue": s.get("dialogue", ""),
            "emotion": s.get("emotion", ""),
            "energy": s.get("energy", "medium"),
            "characters": scene.get("characters_present", []),
            "cine": {
                "shot_size": "", "vertical_angle": "", "horizontal_angle": "",
                "camera_movement": "", "stability": "", "focal_length": "",
                "depth_of_field": "", "bokeh": "", "lens_artifact": "",
                "key_light_direction": "", "light_quality": "", "contrast": "",
                "atmosphere": "", "color_temp": "", "color_harmony": "",
                "film_emulation": "", "aspect_ratio": "", "framing": "",
                "reasoning": "", "suggestion_source": "none"
            },
            "image_prompt": "",
            "prompt_locked": False
        })
    return shots

def generate_storyboard(screenplay_text, series_bible_text, character_bible_text, progress=gr.Progress()):
    screenplay = from_json(screenplay_text, "Screenplay")
    require(screenplay, ["scenes"], "Screenplay")

    series_bible = from_json(series_bible_text, "Series Bible")
    char_bible = from_json(character_bible_text, "Character Bible")
    genre = series_bible.get("genre", "Drama")

    scenes = screenplay["scenes"]
    final_scenes = []
    errors = []

    for i, scene in enumerate(scenes):
        progress((i / len(scenes)), desc=f"Decomposing scene {i+1}/{len(scenes)}…")
        try:
            shots = _decompose_one_scene(scene, genre, char_bible)
        except Exception as e:
            errors.append(f"Scene {scene.get('scene_number', i+1)}: {e}")
            shots = []

        final_scenes.append({
            "scene_number": scene.get("scene_number", i+1),
            "heading": scene.get("heading", ""),
            "location": scene.get("location", ""),
            "time_of_day": scene.get("time_of_day", ""),
            "scene_type": scene.get("scene_type", "dialogue"),
            "characters_present": scene.get("characters_present", []),
            "emotional_beat": scene.get("emotional_beat", ""),
            "estimated_duration_seconds": scene.get("estimated_duration_seconds", 120),
            "shots": shots,
        })

    storyboard = {"scenes": final_scenes, "genre": genre}
    if errors:
        storyboard["_errors"] = errors

    result = to_json(storyboard)
    history.save("storyboard", result)
    scene_nums = [s["scene_number"] for s in final_scenes]

    total_shots = sum(len(s["shots"]) for s in final_scenes)
    status = f"✅ {len(final_scenes)} scenes · {total_shots} total shots · genre: {genre}"
    if errors:
        status += f" ⚠️ {len(errors)} scene(s) failed."
    return result, gr.update(choices=scene_nums, value=scene_nums[0] if scene_nums else None), status, gr.update(choices=scene_nums, value=scene_nums[0] if scene_nums else None)

def rewrite_storyboard(storyboard_text, instruction):
    if not instruction.strip(): raise gr.Error("Enter a rewrite instruction.")
    data = from_json(storyboard_text, "Storyboard")
    result = to_json(gemini(REWRITE_PROMPT + f"\n\nInstruction: {instruction}", data))
    history.save("storyboard", result)
    storyboard = json.loads(result)
    scene_nums = [s["scene_number"] for s in storyboard.get("scenes", [])]
    return result, gr.update(choices=scene_nums, value=scene_nums[0] if scene_nums else None)

def undo_storyboard():
    r = history.undo("storyboard")
    if not r: raise gr.Error("Nothing to undo.")
    storyboard = json.loads(r)
    scene_nums = [s["scene_number"] for s in storyboard.get("scenes", [])]
    return r, gr.update(choices=scene_nums, value=scene_nums[0] if scene_nums else None)

# ============================================================
# 🎥 SHOT EDITOR
# ============================================================
def _get_shot(storyboard, scene_id, shot_id):
    for scene in storyboard.get("scenes", []):
        if scene["scene_number"] == int(scene_id):
            for shot in scene.get("shots", []):
                if shot["shot_id"] == int(shot_id):
                    return shot
    return None

def load_shots_for_scene(storyboard_text, scene_id):
    if scene_id is None:
        return gr.update(choices=[], value=None)
    try:
        storyboard = from_json(storyboard_text, "Storyboard")
    except Exception:
        return gr.update(choices=[], value=None)
    for scene in storyboard.get("scenes", []):
        if scene["scene_number"] == int(scene_id):
            shot_ids = [s["shot_id"] for s in scene.get("shots", [])]
            return gr.update(choices=shot_ids, value=shot_ids[0] if shot_ids else None)
    return gr.update(choices=[], value=None)

def load_shot(storyboard_text, scene_id, shot_id):
    if scene_id is None or shot_id is None:
        raise gr.Error("Select a scene and shot.")
    storyboard = from_json(storyboard_text, "Storyboard")
    shot = _get_shot(storyboard, scene_id, shot_id)
    if shot is None:
        raise gr.Error(f"Shot {shot_id} not found in scene {scene_id}.")
    c = shot.get("cine", {})
    return (
        shot.get("timecode_start", ""), shot.get("timecode_end", ""),
        int(shot.get("duration_seconds", 4)),
        shot.get("action", ""), shot.get("gesture", ""),
        shot.get("facial_expression", ""), shot.get("body_language", ""),
        shot.get("dialogue", ""), shot.get("emotion", ""),
        shot.get("energy", "medium"),
        ", ".join(shot.get("characters", [])),
        c.get("shot_size", ""), c.get("vertical_angle", ""),
        c.get("horizontal_angle", ""), c.get("camera_movement", ""),
        c.get("stability", ""), c.get("focal_length", ""),
        c.get("depth_of_field", ""), c.get("bokeh", ""),
        c.get("lens_artifact", ""), c.get("key_light_direction", ""),
        c.get("light_quality", ""), c.get("contrast", ""),
        c.get("atmosphere", ""), c.get("color_temp", ""),
        c.get("color_harmony", ""), c.get("film_emulation", ""),
        c.get("aspect_ratio", ""), c.get("framing", ""),
        c.get("reasoning", ""),
        shot.get("image_prompt", ""),
        shot.get("prompt_locked", False),
        f"✅ Loaded scene {scene_id} / shot {shot_id}. Cine source: {c.get('suggestion_source', 'none')}"
    )

def save_shot(
    storyboard_text, scene_id, shot_id,
    tc_start, tc_end, duration,
    action, gesture, expression, body_language, dialogue, emotion, energy, characters,
    shot_size, v_angle, h_angle, movement, stability, focal, dof, bokeh, lens_artifact,
    key_light, light_quality, contrast, atmosphere, color_temp, color_harmony,
    film_emulation, aspect_ratio, framing, reasoning,
    image_prompt, prompt_locked,
    suggestion_source="user"
):
    if scene_id is None or shot_id is None:
        raise gr.Error("No shot selected.")
    storyboard = from_json(storyboard_text, "Storyboard")
    shot = _get_shot(storyboard, scene_id, shot_id)
    if shot is None:
        raise gr.Error(f"Shot {shot_id} not found.")
    shot.update({
        "timecode_start": tc_start, "timecode_end": tc_end,
        "duration_seconds": duration,
        "action": action, "gesture": gesture,
        "facial_expression": expression, "body_language": body_language,
        "dialogue": dialogue, "emotion": emotion, "energy": energy,
        "characters": [c.strip() for c in characters.split(",") if c.strip()],
        "cine": {
            "shot_size": shot_size, "vertical_angle": v_angle,
            "horizontal_angle": h_angle, "camera_movement": movement,
            "stability": stability, "focal_length": focal,
            "depth_of_field": dof, "bokeh": bokeh,
            "lens_artifact": lens_artifact, "key_light_direction": key_light,
            "light_quality": light_quality, "contrast": contrast,
            "atmosphere": atmosphere, "color_temp": color_temp,
            "color_harmony": color_harmony, "film_emulation": film_emulation,
            "aspect_ratio": aspect_ratio, "framing": framing,
            "reasoning": reasoning, "suggestion_source": suggestion_source
        },
        "image_prompt": image_prompt,
        "prompt_locked": prompt_locked
    })
    result = to_json(storyboard)
    history.save("storyboard", result)
    return result, f"✅ Scene {scene_id} / Shot {shot_id} saved."

def add_shot(storyboard_text, scene_id):
    if scene_id is None: raise gr.Error("Select a scene first.")
    storyboard = from_json(storyboard_text, "Storyboard")
    for scene in storyboard.get("scenes", []):
        if scene["scene_number"] == int(scene_id):
            shots = scene.get("shots", [])
            new_id = max([s["shot_id"] for s in shots], default=0) + 1
            shots.append({
                "shot_id": new_id, "timecode_start": "00:00", "timecode_end": "00:04",
                "duration_seconds": 4, "action": "", "gesture": "",
                "facial_expression": "", "body_language": "", "dialogue": "",
                "emotion": "", "energy": "medium", "characters": scene.get("characters_present", []),
                "cine": {
                    "shot_size": "", "vertical_angle": "", "horizontal_angle": "",
                    "camera_movement": "", "stability": "", "focal_length": "",
                    "depth_of_field": "", "bokeh": "", "lens_artifact": "",
                    "key_light_direction": "", "light_quality": "", "contrast": "",
                    "atmosphere": "", "color_temp": "", "color_harmony": "",
                    "film_emulation": "", "aspect_ratio": "", "framing": "",
                    "reasoning": "", "suggestion_source": "none"
                },
                "image_prompt": "", "prompt_locked": False
            })
            scene["shots"] = shots
            result = to_json(storyboard)
            history.save("storyboard", result)
            shot_ids = [s["shot_id"] for s in shots]
            return result, f"✅ Shot {new_id} added.", gr.update(choices=shot_ids, value=new_id)
    raise gr.Error(f"Scene {scene_id} not found.")

def delete_shot(storyboard_text, scene_id, shot_id):
    if scene_id is None or shot_id is None: raise gr.Error("Select a shot to delete.")
    storyboard = from_json(storyboard_text, "Storyboard")
    for scene in storyboard.get("scenes", []):
        if scene["scene_number"] == int(scene_id):
            shots = [s for s in scene.get("shots", []) if s["shot_id"] != int(shot_id)]
            for idx, s in enumerate(shots, 1):
                s["shot_id"] = idx
            scene["shots"] = shots
            result = to_json(storyboard)
            history.save("storyboard", result)
            shot_ids = [s["shot_id"] for s in shots]
            return result, "✅ Shot deleted.", gr.update(choices=shot_ids, value=shot_ids[0] if shot_ids else None)
    raise gr.Error(f"Scene {scene_id} not found.")

# ============================================================
# 🎥 INTEGRATED CINE — suggest + synthesize with character DNA
# ============================================================
def suggest_cine(storyboard_text, scene_id, shot_id,
                 action, gesture, expression, body_language, dialogue, emotion, energy,
                 series_bible_text):
    if scene_id is None or shot_id is None: raise gr.Error("Load a shot first.")
    try:
        genre = from_json(series_bible_text, "Series Bible").get("genre", "Drama")
    except Exception:
        genre = "Drama"
    narrative_payload = {
        "action": action, "gesture": gesture, "facial_expression": expression,
        "body_language": body_language, "dialogue": dialogue,
        "emotion": emotion, "energy": energy, "genre": genre
    }
    suggested = gemini(CINE_SUGGEST_PROMPT, narrative_payload)
    return (
        suggested.get("shot_size", ""), suggested.get("vertical_angle", ""),
        suggested.get("horizontal_angle", ""), suggested.get("camera_movement", ""),
        suggested.get("stability", ""), suggested.get("focal_length", ""),
        suggested.get("depth_of_field", ""), suggested.get("bokeh", ""),
        suggested.get("lens_artifact", ""), suggested.get("key_light_direction", ""),
        suggested.get("light_quality", ""), suggested.get("contrast", ""),
        suggested.get("atmosphere", ""), suggested.get("color_temp", ""),
        suggested.get("color_harmony", ""), suggested.get("film_emulation", ""),
        suggested.get("aspect_ratio", ""), suggested.get("framing", ""),
        suggested.get("reasoning", ""),
        f"✅ Cine suggested. Reasoning: {suggested.get('reasoning', '')}"
    )

def _build_character_seeds(characters_in_shot, char_bible_data):
    """Pull reference_prompt_seed for each character present in the shot."""
    seeds = {}
    for char in char_bible_data.get("characters", []):
        name = char.get("name", "")
        if name in characters_in_shot:
            seeds[name] = char.get("reference_prompt_seed", "")
    return seeds

def synthesize_prompt(
    storyboard_text, scene_id, shot_id,
    action, gesture, expression, body_language, dialogue, emotion, energy, characters,
    shot_size, v_angle, h_angle, movement, stability, focal, dof, bokeh, lens_artifact,
    key_light, light_quality, contrast, atmosphere, color_temp, color_harmony,
    film_emulation, aspect_ratio, framing,
    prompt_locked, character_bible_text, series_bible_text
):
    if prompt_locked: raise gr.Error("Prompt is locked. Unlock before re-synthesizing.")
    if scene_id is None or shot_id is None: raise gr.Error("Load a shot first.")

    char_list = [c.strip() for c in characters.split(",") if c.strip()]
    try:
        char_bible = from_json(character_bible_text, "Character Bible")
        char_seeds = _build_character_seeds(char_list, char_bible)
    except Exception:
        char_seeds = {}

    try:
        genre = from_json(series_bible_text, "Series Bible").get("genre", "Drama")
        visual_style = from_json(series_bible_text, "Series Bible").get("visual_style", "")
    except Exception:
        genre, visual_style = "Drama", ""

    full_shot = {
        "narrative": {
            "action": action, "gesture": gesture, "facial_expression": expression,
            "body_language": body_language, "dialogue": dialogue,
            "emotion": emotion, "energy": energy,
            "characters": char_list,
            "character_visual_seeds": char_seeds
        },
        "cinematography": {
            "shot_size": shot_size, "vertical_angle": v_angle,
            "horizontal_angle": h_angle, "camera_movement": movement,
            "stability": stability, "focal_length": focal,
            "depth_of_field": dof, "bokeh": bokeh,
            "lens_artifact": lens_artifact, "key_light_direction": key_light,
            "light_quality": light_quality, "contrast": contrast,
            "atmosphere": atmosphere, "color_temp": color_temp,
            "color_harmony": color_harmony, "film_emulation": film_emulation,
            "aspect_ratio": aspect_ratio, "framing": framing
        },
        "series_context": {
            "genre": genre,
            "visual_style": visual_style
        }
    }
    result = gemini(SYNTHESIZE_PROMPT, full_shot)
    prompt = result.get("image_prompt", "")
    return prompt, f"✅ Prompt synthesized for scene {scene_id} / shot {shot_id}."

def generate_all_prompts(storyboard_text, character_bible_text, series_bible_text, progress=gr.Progress()):
    storyboard = from_json(storyboard_text, "Storyboard")
    try:
        char_bible = from_json(character_bible_text, "Character Bible")
    except Exception:
        char_bible = {"characters": []}
    try:
        sb = from_json(series_bible_text, "Series Bible")
        genre = sb.get("genre", "Drama")
        visual_style = sb.get("visual_style", "")
    except Exception:
        genre, visual_style = "Drama", ""

    total_shots = sum(len(s.get("shots", [])) for s in storyboard.get("scenes", []))
    done = 0

    for scene in storyboard.get("scenes", []):
        for shot in scene.get("shots", []):
            if shot.get("prompt_locked", False):
                done += 1
                continue
            progress(done / max(total_shots, 1), desc=f"Synthesizing shot {done+1}/{total_shots}…")
            try:
                char_list = shot.get("characters", [])
                char_seeds = _build_character_seeds(char_list, char_bible)
                c = shot.get("cine", {})
                full_shot = {
                    "narrative": {
                        "action": shot.get("action", ""),
                        "gesture": shot.get("gesture", ""),
                        "facial_expression": shot.get("facial_expression", ""),
                        "body_language": shot.get("body_language", ""),
                        "dialogue": shot.get("dialogue", ""),
                        "emotion": shot.get("emotion", ""),
                        "energy": shot.get("energy", ""),
                        "characters": char_list,
                        "character_visual_seeds": char_seeds
                    },
                    "cinematography": {k: c.get(k, "") for k in [
                        "shot_size", "vertical_angle", "horizontal_angle",
                        "camera_movement", "stability", "focal_length",
                        "depth_of_field", "bokeh", "lens_artifact",
                        "key_light_direction", "light_quality", "contrast",
                        "atmosphere", "color_temp", "color_harmony",
                        "film_emulation", "aspect_ratio", "framing"
                    ]},
                    "series_context": {"genre": genre, "visual_style": visual_style}
                }
                result = gemini(SYNTHESIZE_PROMPT, full_shot)
                shot["image_prompt"] = result.get("image_prompt", "")
            except Exception as e:
                shot["image_prompt"] = f"[ERROR: {e}]"
            done += 1

    result_text = to_json(storyboard)
    history.save("storyboard", result_text)
    return result_text, f"✅ All prompts synthesized ({total_shots} shots)."

def _rewrite_shot_inline(storyboard_text, scene_id, shot_id, instruction):
    if not instruction.strip(): raise gr.Error("Enter a rewrite instruction.")
    storyboard = from_json(storyboard_text, "Storyboard")
    shot = _get_shot(storyboard, scene_id, shot_id)
    if shot is None: raise gr.Error(f"Shot {shot_id} not found.")
    rewritten = gemini(REWRITE_PROMPT + f"\n\nInstruction: {instruction}", shot)
    for key in rewritten:
        shot[key] = rewritten[key]
    result = to_json(storyboard)
    history.save("storyboard", result)
    return json.loads(result)

# ============================================================
# 📊 SERIES STATS
# ============================================================
def get_series_stats(storyboard_text, series_bible_text):
    try:
        sb = from_json(storyboard_text, "Storyboard")
        scenes = sb.get("scenes", [])
        total_shots = sum(len(s.get("shots", [])) for s in scenes)
        locked_prompts = sum(
            1 for s in scenes for sh in s.get("shots", []) if sh.get("prompt_locked")
        )
        filled_prompts = sum(
            1 for s in scenes for sh in s.get("shots", []) if sh.get("image_prompt")
        )
        genre = sb.get("genre", "?")
        budget = GENRE_SHOT_BUDGETS.get(genre, {})
        stats = (
            f"📊 Storyboard stats\n"
            f"  Scenes: {len(scenes)}\n"
            f"  Total shots: {total_shots}\n"
            f"  Prompts filled: {filled_prompts}/{total_shots}\n"
            f"  Locked prompts: {locked_prompts}\n"
            f"  Genre: {genre} | Budget: {budget.get('min_shots',0)}–{budget.get('max_shots',150)} shots/scene\n"
        )
        return stats
    except Exception as e:
        return f"No storyboard loaded yet. ({e})"

# ============================================================
# 💾 SAVE / LOAD
# ============================================================
def save_project(series_bible, character_bible, script, screenplay, storyboard, episode_number):
    project = {
        "version": "4.0",
        "saved_at": datetime.now().isoformat(),
        "current_episode": episode_number,
        "series_bible": from_json(series_bible) if series_bible and series_bible.strip() else None,
        "character_bible": from_json(character_bible) if character_bible and character_bible.strip() else None,
        "script": from_json(script) if script and script.strip() else None,
        "screenplay": from_json(screenplay) if screenplay and screenplay.strip() else None,
        "storyboard": from_json(storyboard) if storyboard and storyboard.strip() else None,
    }
    filename = f"/tmp/series_project_{int(time.time())}.json"
    with open(filename, "w") as f:
        json.dump(project, f, indent=2, ensure_ascii=False)
    return filename

def load_project(file):
    if file is None: raise gr.Error("Upload a project file.")
    with open(file.name, "r") as f:
        project = json.load(f)
    storyboard = project.get("storyboard") or {}
    scene_nums = [s["scene_number"] for s in storyboard.get("scenes", [])]
    series_bible = project.get("series_bible") or {}
    ep_nums = [e["episode_number"] for e in series_bible.get("episode_outlines", [])]
    return (
        to_json(series_bible),
        to_json(project.get("character_bible") or {}),
        to_json(project.get("script") or {}),
        to_json(project.get("screenplay") or {}),
        to_json(storyboard),
        gr.update(choices=scene_nums, value=scene_nums[0] if scene_nums else None),
        gr.update(choices=ep_nums, value=project.get("current_episode") or (ep_nums[0] if ep_nums else None)),
        f"✅ Project loaded (v{project.get('version','?')}, saved {project.get('saved_at','unknown')})"
    )

# ============================================================
# 🎨 GRADIO UI
# ============================================================

SHOT_SIZES   = ["Extreme Wide Shot", "Wide Shot", "Full Shot", "Medium Shot",
                "Medium Close-Up", "Close-Up", "Extreme Close-Up"]
V_ANGLES     = ["Eye-Level", "Low Angle", "High Angle", "Dutch Angle"]
H_ANGLES     = ["Frontal", "3/4 View", "Profile", "3/4 Rear", "Rear View"]
MOVEMENTS    = ["Static", "Panning", "Tilting", "Tracking",
                "Dolly In", "Dolly Out", "Crane", "Handheld"]
STABILITIES  = ["Tripod", "Steadicam", "Handheld", "Shaky Cam"]
FOCALS       = ["Wide Angle <35mm", "Normal 50mm", "Telephoto 85mm", "Telephoto 135mm+"]
DOFS         = ["Deep Focus", "Shallow", "Razor-Thin"]
BOKEH        = ["Creamy", "Busy", "None"]
LENS_ART     = ["None", "Lens Flare", "Chromatic Aberration", "Vignette"]
KEY_LIGHTS   = ["Frontal", "Rembrandt 3/4", "Side", "Backlight", "Top", "Under"]
LIGHT_QUAL   = ["Hard", "Soft"]
CONTRASTS    = ["High-Key", "Low-Key", "Chiaroscuro"]
ATMOSPHERES  = ["Clean", "Haze", "Fog", "Volumetric"]
COLOR_TEMPS  = ["Warm Tungsten", "Neutral Daylight", "Cool", "Mixed"]
COLOR_HARMS  = ["Monochromatic", "Analogous", "Complementary Teal-Orange", "Desaturated"]
FILM_EMULS   = ["Kodak Portra", "Kodak Vision3", "Fujifilm", "Cinestill 800T",
                "Bleach Bypass", "Digital Clean"]
ASPECTS      = ["1.33:1", "1.85:1", "2.39:1 Anamorphic"]
FRAMINGS     = ["Rule of Thirds", "Center", "Frame-within-Frame", "Negative Space"]
ENERGIES     = ["low", "medium", "high"]
GENRES       = list(GENRE_SHOT_BUDGETS.keys())
TONES        = ["Serious", "Dark", "Suspenseful", "Emotional", "Light", "Quirky", "Epic", "Grounded"]
EP_COUNTS    = [6, 8, 10, 12, 13, 16, 20, 24]

with gr.Blocks(title="🎬 AI Film Engine V4 — Series Edition") as demo:

    gr.Markdown("# 🎬 AI Film Engine V4 — Series Edition\n### Series Bible → Character Bible → Episodes → Scenes → Shots → Prompts")

    with gr.Row():
        rewrite_instruction = gr.Textbox(
            label="✏️ AI Rewrite Instruction (applies to the tab you use it in)",
            placeholder="Make it darker… add subtext… increase the action intensity…",
            lines=2, scale=4
        )
    gr.Markdown("---")

    with gr.Tabs():

        with gr.Tab("📚 Series Bible"):
            gr.Markdown("### Step 1 — Generate your series bible (all 12 episodes outlined)")
            with gr.Row():
                with gr.Column(scale=3):
                    series_concept = gr.Textbox(
                        label="🎬 Series Concept",
                        placeholder="A detective with surgically altered memories investigates cases that feel disturbingly familiar…",
                        lines=3
                    )
                with gr.Column(scale=1):
                    series_genre = gr.Dropdown(label="Genre", choices=GENRES, value="Thriller")
                    series_tone  = gr.Dropdown(label="Tone", choices=TONES, value="Dark")
                    series_ep_count = gr.Dropdown(label="Episodes", choices=EP_COUNTS, value=12)

            series_bible_status = gr.Textbox(label="", interactive=False, show_label=False, max_lines=1)
            series_bible_out = gr.Code(label="Series Bible JSON", language="json", lines=25)
            with gr.Row():
                btn_series_bible      = gr.Button("🚀 Generate Series Bible", variant="primary")
                btn_series_bible_rw   = gr.Button("✨ Rewrite")
                btn_series_bible_undo = gr.Button("↶ Undo")

        with gr.Tab("🎭 Character Bible"):
            gr.Markdown("""
            ### Step 2 — Generate character bible
            Every character gets a **visual DNA** (reference_prompt_seed) that gets injected
            into every image prompt they appear in — ensuring consistent faces, hair, build, and wardrobe
            across all episodes.
            """)
            char_bible_status = gr.Textbox(label="", interactive=False, show_label=False, max_lines=1)
            character_bible_out = gr.Code(label="Character Bible JSON", language="json", lines=25)
            with gr.Row():
                btn_char_bible      = gr.Button("🚀 Generate Character Bible", variant="primary")
                btn_char_bible_rw   = gr.Button("✨ Rewrite")
                btn_char_bible_undo = gr.Button("↶ Undo")
            gr.Markdown("*Reads from Series Bible tab automatically.*")

        with gr.Tab("📄 Episode Script"):
            gr.Markdown("### Step 3 — Generate one episode at a time")
            with gr.Row():
                episode_selector = gr.Dropdown(label="Episode", choices=list(range(1,13)), value=1, scale=1)
                btn_load_eps = gr.Button("🔄 Reload episode list from bible", size="sm", scale=1)

            script_status = gr.Textbox(label="", interactive=False, show_label=False, max_lines=1)
            script_out = gr.Code(label="Episode Script JSON", language="json", lines=22)
            with gr.Row():
                btn_script      = gr.Button("🚀 Generate Episode Script", variant="primary")
                btn_script_rw   = gr.Button("✨ Rewrite")
                btn_script_undo = gr.Button("↶ Undo")
            gr.Markdown("*Reads from Series Bible + Character Bible automatically.*")

        with gr.Tab("🎞️ Screenplay"):
            sp_status = gr.Textbox(label="", interactive=False, show_label=False, max_lines=1)
            screenplay_out = gr.Code(label="Screenplay JSON", language="json", lines=22)
            with gr.Row():
                btn_sp      = gr.Button("🚀 Generate Screenplay", variant="primary")
                btn_sp_rw   = gr.Button("✨ Rewrite")
                btn_sp_undo = gr.Button("↶ Undo")
            gr.Markdown("*Reads from Episode Script + Character Bible.*")

        with gr.Tab("🎬 Storyboard"):
            sb_status = gr.Textbox(label="", interactive=False, show_label=False, max_lines=2)
            storyboard_out = gr.Code(label="Storyboard JSON", language="json", lines=20)
            with gr.Row():
                btn_sb          = gr.Button("🚀 Decompose Scenes → Shots (genre-aware)", variant="primary")
                btn_sb_rw       = gr.Button("✨ Rewrite")
                btn_sb_undo     = gr.Button("↶ Undo")
                btn_all_prompts = gr.Button("⚡ Synthesize All Prompts")
            with gr.Row():
                btn_sb_stats = gr.Button("📊 Stats")
                sb_stats_out = gr.Textbox(label="", interactive=False, show_label=False, lines=6, scale=3)
            gr.Markdown("""
            *Shot budgets are genre-aware:*
            - **Romance** 8–30 shots/scene · **Drama** 15–50 · **Thriller** 30–80 · **Action** 60–150
            - Scene type (action/climax) scales the budget up; dialogue/transition scales it down
            """)
            scene_selector = gr.Dropdown(label="Scene", choices=[], interactive=True)

        with gr.Tab("🎥 Shot Editor"):
            gr.Markdown("""
            ### Workflow
            1. Select **scene** → **shot** → **Load Shot**
            2. Edit narrative fields
            3. **Suggest Cine** — AI reads narrative + genre → populates cinematography
            4. Tweak cine params as needed
            5. **Synthesize Prompt** — AI writes coherent prompt with **character visual DNA injected**
            6. Lock if happy → **Save Shot**
            """)
            with gr.Row():
                scene_sel2 = gr.Dropdown(label="Scene",  choices=[], interactive=True, scale=1, allow_custom_value=True)
                shot_sel   = gr.Dropdown(label="Shot",   choices=[], interactive=True, scale=1, allow_custom_value=True)
                btn_load_shot   = gr.Button("📂 Load Shot",   size="sm", scale=1)
                btn_add_shot    = gr.Button("➕ Add Shot",    size="sm", scale=1)
                btn_delete_shot = gr.Button("🗑️ Delete Shot", size="sm", scale=1, variant="stop")

            editor_status = gr.Textbox(label="", interactive=False, show_label=False, max_lines=2)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 📝 Narrative")
                    with gr.Row():
                        e_tc_start = gr.Textbox(label="TC start", value="00:00", scale=1)
                        e_tc_end   = gr.Textbox(label="TC end",   value="00:04", scale=1)
                        e_duration = gr.Slider(label="Duration (s)", minimum=1, maximum=15, value=4, step=1, scale=2)
                    e_action     = gr.Textbox(label="Action",            lines=2)
                    e_gesture    = gr.Textbox(label="Gesture",           lines=1)
                    e_expression = gr.Textbox(label="Facial expression", lines=1)
                    e_body       = gr.Textbox(label="Body language",     lines=1)
                    e_dialogue   = gr.Textbox(label="Dialogue",          lines=1)
                    with gr.Row():
                        e_emotion = gr.Textbox(label="Emotion", scale=2)
                        e_energy  = gr.Dropdown(label="Energy", choices=ENERGIES, value="medium", scale=1)
                    e_characters = gr.Textbox(label="Characters (comma-separated — must match Character Bible names)")

                with gr.Column(scale=1):
                    gr.Markdown("#### 🎥 Cinematography *(AI-suggested — override freely)*")
                    with gr.Row():
                        c_shot_size = gr.Dropdown(label="Shot size",       choices=SHOT_SIZES,  value="Medium Shot")
                        c_v_angle   = gr.Dropdown(label="Vertical angle",  choices=V_ANGLES,    value="Eye-Level")
                    with gr.Row():
                        c_h_angle   = gr.Dropdown(label="Horizontal angle",choices=H_ANGLES,    value="Frontal")
                        c_movement  = gr.Dropdown(label="Camera movement", choices=MOVEMENTS,   value="Static")
                    with gr.Row():
                        c_stability = gr.Dropdown(label="Stability",       choices=STABILITIES, value="Tripod")
                        c_focal     = gr.Dropdown(label="Focal length",    choices=FOCALS,      value="Normal 50mm")
                    with gr.Row():
                        c_dof       = gr.Dropdown(label="Depth of field",  choices=DOFS,        value="Shallow")
                        c_bokeh     = gr.Dropdown(label="Bokeh",           choices=BOKEH,       value="Creamy")
                    with gr.Row():
                        c_lens_art  = gr.Dropdown(label="Lens artifact",   choices=LENS_ART,    value="None")
                        c_key_light = gr.Dropdown(label="Key light",       choices=KEY_LIGHTS,  value="Rembrandt 3/4")
                    with gr.Row():
                        c_lquality  = gr.Dropdown(label="Light quality",   choices=LIGHT_QUAL,  value="Soft")
                        c_contrast  = gr.Dropdown(label="Contrast",        choices=CONTRASTS,   value="Low-Key")
                    with gr.Row():
                        c_atmo      = gr.Dropdown(label="Atmosphere",      choices=ATMOSPHERES, value="Clean")
                        c_ctemp     = gr.Dropdown(label="Color temp",      choices=COLOR_TEMPS, value="Warm Tungsten")
                    with gr.Row():
                        c_charmony  = gr.Dropdown(label="Color harmony",   choices=COLOR_HARMS, value="Complementary Teal-Orange")
                        c_film      = gr.Dropdown(label="Film emulation",  choices=FILM_EMULS,  value="Kodak Portra")
                    with gr.Row():
                        c_aspect    = gr.Dropdown(label="Aspect ratio",    choices=ASPECTS,     value="2.39:1 Anamorphic")
                        c_framing   = gr.Dropdown(label="Framing",         choices=FRAMINGS,    value="Rule of Thirds")
                    c_reasoning = gr.Textbox(label="AI reasoning", lines=2, interactive=False)

            gr.Markdown("#### 🖼️ Synthesized Image Prompt *(character visual DNA auto-injected)*")
            with gr.Row():
                e_image_prompt = gr.Textbox(label="Image prompt", lines=4, scale=4)
                e_locked       = gr.Checkbox(label="🔒 Lock", value=False, scale=1)

            with gr.Row():
                btn_suggest_cine  = gr.Button("🤖 Suggest Cine",      variant="secondary")
                btn_synth_prompt  = gr.Button("✨ Synthesize Prompt",  variant="primary")
                btn_save_shot     = gr.Button("💾 Save Shot",          variant="primary")
                btn_rewrite_shot  = gr.Button("✏️ Rewrite Shot")
                btn_shot_undo     = gr.Button("↶ Undo")

        with gr.Tab("💾 Save / Load"):
            gr.Markdown("### Save or load the full series project\n*(series bible + character bible + current episode script + screenplay + storyboard)*")
            sl_status = gr.Textbox(label="", interactive=False, show_label=False, max_lines=1)
            with gr.Row():
                btn_save    = gr.Button("📥 Save Project", variant="primary")
                save_output = gr.File(label="Download Project JSON")
            with gr.Row():
                file_input = gr.File(label="📤 Upload Project JSON", file_types=[".json"])
                btn_load   = gr.Button("📂 Load Project")

    # ── Series Bible ──
    btn_series_bible.click(
        generate_series_bible,
        [series_concept, series_genre, series_tone, series_ep_count],
        [series_bible_out, series_bible_status]
    )
    btn_series_bible_rw.click(rewrite_series_bible, [series_bible_out, rewrite_instruction], series_bible_out)
    btn_series_bible_undo.click(undo_series_bible, None, series_bible_out)

    series_bible_out.change(get_episode_choices, series_bible_out, episode_selector)
    btn_load_eps.click(get_episode_choices, series_bible_out, episode_selector)

    # ── Character Bible ──
    btn_char_bible.click(generate_character_bible, series_bible_out, [character_bible_out, char_bible_status])
    btn_char_bible_rw.click(rewrite_character_bible, [character_bible_out, rewrite_instruction], character_bible_out)
    btn_char_bible_undo.click(undo_character_bible, None, character_bible_out)

    # ── Episode Script ──
    btn_script.click(
        generate_episode_script,
        [series_bible_out, character_bible_out, episode_selector],
        [script_out, script_status]
    )
    btn_script_rw.click(rewrite_script, [script_out, rewrite_instruction], script_out)
    btn_script_undo.click(undo_script, None, script_out)

    # ── Screenplay ──
    btn_sp.click(generate_screenplay, [script_out, character_bible_out], [screenplay_out, sp_status])
    btn_sp_rw.click(rewrite_screenplay, [screenplay_out, rewrite_instruction], screenplay_out)
    btn_sp_undo.click(undo_screenplay, None, screenplay_out)

    # ── Storyboard ──
    btn_sb.click(
        generate_storyboard,
        [screenplay_out, series_bible_out, character_bible_out],
        [storyboard_out, scene_selector, sb_status, scene_sel2]
    )
    btn_sb_rw.click(rewrite_storyboard, [storyboard_out, rewrite_instruction], [storyboard_out, scene_selector])
    btn_sb_undo.click(undo_storyboard, None, [storyboard_out, scene_selector])
    btn_all_prompts.click(
        generate_all_prompts,
        [storyboard_out, character_bible_out, series_bible_out],
        [storyboard_out, sb_status]
    )
    btn_sb_stats.click(get_series_stats, [storyboard_out, series_bible_out], sb_stats_out)

    scene_selector.change(lambda x: x, scene_selector, scene_sel2)
    scene_sel2.change(lambda x: x, scene_sel2, scene_selector)
    scene_sel2.change(load_shots_for_scene, [storyboard_out, scene_sel2], shot_sel)

    # ── Shot Editor ──
    _narrative_outs = [
        e_tc_start, e_tc_end, e_duration,
        e_action, e_gesture, e_expression, e_body,
        e_dialogue, e_emotion, e_energy, e_characters
    ]
    _cine_outs = [
        c_shot_size, c_v_angle, c_h_angle, c_movement, c_stability,
        c_focal, c_dof, c_bokeh, c_lens_art, c_key_light,
        c_lquality, c_contrast, c_atmo, c_ctemp, c_charmony,
        c_film, c_aspect, c_framing, c_reasoning
    ]
    _all_load_outs = _narrative_outs + _cine_outs + [e_image_prompt, e_locked, editor_status]

    btn_load_shot.click(load_shot, [storyboard_out, scene_sel2, shot_sel], _all_load_outs)

    btn_suggest_cine.click(
        suggest_cine,
        [storyboard_out, scene_sel2, shot_sel,
         e_action, e_gesture, e_expression, e_body, e_dialogue, e_emotion, e_energy,
         series_bible_out],
        _cine_outs + [editor_status]
    )

    btn_synth_prompt.click(
        synthesize_prompt,
        [storyboard_out, scene_sel2, shot_sel,
         e_action, e_gesture, e_expression, e_body, e_dialogue, e_emotion, e_energy, e_characters,
         c_shot_size, c_v_angle, c_h_angle, c_movement, c_stability,
         c_focal, c_dof, c_bokeh, c_lens_art, c_key_light,
         c_lquality, c_contrast, c_atmo, c_ctemp, c_charmony,
         c_film, c_aspect, c_framing,
         e_locked, character_bible_out, series_bible_out],
        [e_image_prompt, editor_status]
    )

    btn_save_shot.click(
        save_shot,
        [storyboard_out, scene_sel2, shot_sel,
         e_tc_start, e_tc_end, e_duration,
         e_action, e_gesture, e_expression, e_body, e_dialogue, e_emotion, e_energy, e_characters,
         c_shot_size, c_v_angle, c_h_angle, c_movement, c_stability,
         c_focal, c_dof, c_bokeh, c_lens_art, c_key_light,
         c_lquality, c_contrast, c_atmo, c_ctemp, c_charmony,
         c_film, c_aspect, c_framing, c_reasoning,
         e_image_prompt, e_locked],
        [storyboard_out, editor_status]
    )

    btn_add_shot.click(add_shot, [storyboard_out, scene_sel2], [storyboard_out, editor_status, shot_sel])
    btn_delete_shot.click(delete_shot, [storyboard_out, scene_sel2, shot_sel], [storyboard_out, editor_status, shot_sel])

    btn_rewrite_shot.click(
        lambda sb, sc, sh, instr: (
            lambda data: (to_json(data), "✅ Shot rewritten.")
        )(_rewrite_shot_inline(sb, sc, sh, instr)),
        [storyboard_out, scene_sel2, shot_sel, rewrite_instruction],
        [storyboard_out, editor_status]
    )

    btn_shot_undo.click(undo_storyboard, None, [storyboard_out, scene_selector])

    # ── Save / Load ──
    btn_save.click(
        save_project,
        [series_bible_out, character_bible_out, script_out, screenplay_out, storyboard_out, episode_selector],
        save_output
    )
    btn_load.click(
        load_project, file_input,
        [series_bible_out, character_bible_out, script_out, screenplay_out,
         storyboard_out, scene_selector, episode_selector, sl_status]
    )

# ============================================================
# 🚀 LAUNCH
# ============================================================
if __name__ == "__main__":
    demo.launch(share=True, debug=True)