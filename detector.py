

import sys
import traceback

def excepthook(type, value, tb):
    print("".join(traceback.format_exception(type, value, tb)))

sys.excepthook = excepthook

import os
import re
import cv2
import json
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter, deque
import mediapipe as mp
import openai
import base64

from groq import Groq


# -- API client (OpenAI only) ---------------------------------------------------
open_ai_key=os.environ.get("OPENAI_API_KEY")
openai_client = openai.OpenAI(api_key=open_ai_key) 


groqs_key=os.environ.get("GROQ_API_KEY")
groq_client=Groq(api_key=groqs_key)


# -- MediaPipe -----------------------------------------------------------------
mp_face_mesh   = mp.solutions.face_mesh
mp_face_detect = mp.solutions.face_detection

LEFT_IRIS   = [474, 475, 476, 477]
RIGHT_IRIS  = [469, 470, 471, 472]
LEFT_EAR_H  = (362, 263)
RIGHT_EAR_H = (33,  133)

MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),
    (0.0,   -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0,  -150.0, -125.0),
], dtype=np.float64)


# ==============================================================================
# Speech naturalness regexes (local, zero API cost)
# ==============================================================================

STRUCTURED_CONNECTORS = re.compile(
    r"\b(firstly|secondly|thirdly|in\s+conclusion|to\s+summarize|"
    r"in\s+summary|to\s+conclude|lastly|finally,\s+I\s+would|"
    r"in\s+addition\s+to\s+that|moving\s+on\s+to|as\s+a\s+result\s+of)\b",
    re.IGNORECASE,
)

BOOKISH_PHRASES = re.compile(
    r"\b(it\s+is\s+important\s+to\s+note|furthermore|moreover|"
    r"it\s+is\s+worth\s+mentioning|one\s+must\s+consider|"
    r"this\s+can\s+be\s+attributed|in\s+the\s+context\s+of|"
    r"it\s+should\s+be\s+noted|as\s+previously\s+mentioned|"
    r"in\s+other\s+words|to\s+elaborate\s+on\s+this|"
    r"a\s+key\s+aspect\s+of|plays\s+a\s+crucial\s+role)\b",
    re.IGNORECASE,
)

HESITATION_WORDS = re.compile(
    r"\b(um+|uh+|hmm+|err+|ah+|like,|you\s+know,|i\s+mean,|"
    r"sort\s+of|kind\s+of|basically|honestly|literally|right\?)\b",
    re.IGNORECASE,
)

SELF_CORRECTIONS = re.compile(
    r"\b(i\s+mean|no\s+wait|actually,|correction|sorry,\s+i\s+meant|"
    r"let\s+me\s+rephrase|what\s+i\s+meant\s+to\s+say)\b",
    re.IGNORECASE,
)

CONTRACTIONS = re.compile(
    r"\b(i'm|i've|i'll|i'd|don't|doesn't|didn't|can't|won't|"
    r"wouldn't|couldn't|shouldn't|it's|that's|there's|we're|"
    r"they're|you're|wasn't|aren't|isn't)\b",
    re.IGNORECASE,
)


@dataclass
class Violation:
    frame:    int
    time_s:   float
    type:     str
    detail:   str
    severity: str


@dataclass
class DetectionResult:
    student_id:          str
    video_path:          str
    duration_s:          float
    total_frames:        int
    processed_frames:    int
    interviewee_name:    str  = "unknown"
    interviewee_region:  dict = field(default_factory=dict)
    transcript_full:     str  = ""
    transcript_summary:  str  = ""
    interview_topic:     str  = ""
    violations:          list = field(default_factory=list)
    gaze_pattern:        dict = field(default_factory=dict)
    speech_naturalness:  dict = field(default_factory=dict)
    cheating_score:      int  = 0
    risk_level:          str  = "LOW"
    counts:              dict = field(default_factory=dict)
    timeline:            list = field(default_factory=list)
    summary:             str  = ""

    def to_dict(self):
        return {
            "student_id":          self.student_id,
            "video_path":          self.video_path,
            "duration_s":          round(self.duration_s, 2),
            "total_frames":        self.total_frames,
            "processed_frames":    self.processed_frames,
            "interviewee_name":    self.interviewee_name,
            "interviewee_region":  self.interviewee_region,
            "transcript_full":     self.transcript_full,
            "transcript_summary":  self.transcript_summary,
            "interview_topic":     self.interview_topic,
            "cheating_score":      self.cheating_score,
            "risk_level":          self.risk_level,
            "total_violations":    len(self.violations),
            "gaze_pattern":        self.gaze_pattern,
            "speech_naturalness":  self.speech_naturalness,
            "ai_speech_score":     self.speech_naturalness.get("overall_ai_speech_score", 0),
            "speech_risk":         self.speech_naturalness.get("speech_risk", "LOW"),
            "counts":              self.counts,
            "violations":          [v.__dict__ for v in self.violations],
            "timeline":            self.timeline,
            "summary":             self.summary,
        }


# ==============================================================================
# STEP 1 -- Extract audio
# ==============================================================================
def extract_audio(video_path: str) -> Optional[str]:
    try:
        audio_path = tempfile.mktemp(suffix=".mp3")
        result = subprocess.run([
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "libmp3lame",
            "-ar", "16000", "-ac", "1", "-q:a", "4",
            audio_path, "-y", "-loglevel", "error"
        ], capture_output=True, timeout=300)
        if result.returncode == 0 and Path(audio_path).exists():
            print(f"[AUDIO] Extracted: {Path(audio_path).stat().st_size/1024/1024:.1f}MB")
            return audio_path
        print(f"[AUDIO] ffmpeg failed: {result.stderr.decode()}")
        return None
    except FileNotFoundError:
        print("[AUDIO] ffmpeg not found -- install ffmpeg and add it to PATH. Skipping transcription.")
        return None
    except Exception as e:
        print(f"[AUDIO] Exception: {e}")
        return None


# ==============================================================================
# STEP 2 -- Transcribe with Whisper (chunked for long videos)
# ==============================================================================
def transcribe_audio(audio_path: str) -> Optional[dict]:
    try:
        size_mb    = Path(audio_path).stat().st_size / 1024 / 1024
        duration_s = get_audio_duration(audio_path)
        print(f"[WHISPER] Groq Audio: {size_mb:.1f}MB, {duration_s:.0f}s")

        CHUNK_MINUTES = 10
        CHUNK_S       = CHUNK_MINUTES * 60

        if duration_s <= CHUNK_S and size_mb <= 24:
            chunks = [(audio_path, 0)]
        else:
            print(f"[WHISPER] Splitting into {CHUNK_MINUTES}min chunks...")
            chunks = split_audio(audio_path, CHUNK_S)

        all_segments    = []
        full_text_parts = []

        for chunk_path, offset_s in chunks:
            chunk_mb = Path(chunk_path).stat().st_size / 1024 / 1024
            print(f"[WHISPER] Transcribing chunk offset={offset_s:.0f}s size={chunk_mb:.1f}MB")

            with open(chunk_path, "rb") as f:
                response = groq_client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )

            full_text_parts.append(response.text)

            for seg in (response.segments or []):
                if isinstance(seg, dict):
                    all_segments.append({
                        "start": round(seg["start"] + offset_s, 2),
                        "end":   round(seg["end"]   + offset_s, 2),
                        "text":  seg["text"].strip(),
                    })
                else:
                    all_segments.append({
                        "start": round(seg.start + offset_s, 2),
                        "end":   round(seg.end   + offset_s, 2),
                        "text":  seg.text.strip(),
                    })

            if chunk_path != audio_path:
                try: Path(chunk_path).unlink()
                except: pass

        full_text = " ".join(full_text_parts)
        print(f"[WHISPER] Complete: {len(all_segments)} segments, {len(full_text)} chars")

        return {
            "full_text": full_text,
            "segments":  all_segments,
            "language":  response.language,
        }
    except Exception as e:
        print(f"[WHISPER] Groq Failed: {e}")
        return None


def get_audio_duration(audio_path: str) -> float:
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path
        ], capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except:
        return 0


def split_audio(audio_path: str, chunk_s: int) -> list:
    duration  = get_audio_duration(audio_path)
    if duration == 0:
        return [(audio_path, 0)]

    chunks    = []
    offset    = 0
    chunk_idx = 0

    while offset < duration:
        chunk_path = tempfile.mktemp(suffix=f"_chunk{chunk_idx}.mp3")
        result = subprocess.run([
            "ffmpeg", "-i", audio_path,
            "-ss", str(offset),
            "-t",  str(chunk_s),
            "-acodec", "libmp3lame",
            "-ar", "16000", "-ac", "1", "-q:a", "4",
            chunk_path, "-y", "-loglevel", "error"
        ], capture_output=True, timeout=120)

        if result.returncode == 0 and Path(chunk_path).exists():
            chunk_mb = Path(chunk_path).stat().st_size / 1024 / 1024
            if chunk_mb > 0.1:
                chunks.append((chunk_path, offset))

        offset    += chunk_s
        chunk_idx += 1

    print(f"[WHISPER] Split into {len(chunks)} chunks")
    return chunks


# ==============================================================================
# STEP 3 -- Opneai identifies interviewee name + summary
# ==============================================================================
def identify_interviewee_name(transcript: dict) -> dict:
    try:
        full_text = transcript.get("full_text", "")
        if not full_text or len(full_text) < 30:
            return {"interviewee_name": None, "summary": "Too short", "interview_topic": "unknown"}

        prompt = f"""You are analyzing a job interview transcript.

TRANSCRIPT:
{full_text[:5000]}

1. Who is the INTERVIEWEE? (they answer questions about their background/skills/experience)
2. Who is the INTERVIEWER? (they ask questions)
3. If the interviewee's name is mentioned anywhere in the conversation, extract it.
   Names are often mentioned at the start: "Hi I'm John" or "Can you introduce yourself, Priya?"
4. What is the interview about? (job role / skill being evaluated)
5. Write a 3-5 sentence summary of what was discussed.

Respond ONLY in this exact JSON format, no other text:
{{
  "interviewee_name": "first name or full name if found, else null",
  "interviewer_name": "first name or full name if found, else null",
  "interview_topic": "e.g. Oracle EBS Functional Consultant",
  "summary": "3-5 sentence summary of the interview",
  "confidence": "HIGH or MEDIUM or LOW",
  "name_mention": "quote from transcript where name was mentioned, or null"
}}"""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        text   = response.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        parsed = json.loads(text)
        print(f"[GPT] Interviewee: {parsed.get('interviewee_name')} | Topic: {parsed.get('interview_topic')} | Confidence: {parsed.get('confidence')}")
        return parsed

    except Exception as e:
        print(f"[GPT-4o-MINI] identify_interviewee_name failed: {e}")
        return {"interviewee_name": None, "summary": f"Error: {e}", "interview_topic": "unknown"}


# ==============================================================================
# STEP 4 -- OCR: find name label on screen, get face region above it
# ==============================================================================
def find_interviewee_region_by_ocr(video_path: str, interviewee_name: Optional[str], frame_w: int, frame_h: int) -> dict:
    if interviewee_name:
        region = _openai_vision_find_interviewee(video_path, interviewee_name, frame_w, frame_h)
        if region:
            return region
        print("[REGION] Groq Vision Failed -- trying largest face fallback")

    return find_largest_face_region(video_path, frame_w, frame_h)






def _openai_vision_find_interviewee(video_path: str, interviewee_name: Optional[str], frame_w: int, frame_h: int) -> Optional[dict]:
    """Uses vision model to locate the interviewee tile using normalized coordinates."""
    try:
        cap= cv2.VideoCapture(video_path)
        frame_to_use = None

        for t in [30, 60, 15, 90, 10]:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if ret and frame is not None:
                if cv2.mean(frame)[0] > 20:
                    
                    frame_to_use = frame
                    print(f"[Groq VISION] Using frame at t={t}s")
                    break
        cap.release()

        if frame_to_use is None:
            return None

        # Encode image
        _, buf    = cv2.imencode(".jpg", frame_to_use, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_frame = base64.b64encode(buf).decode("utf-8")

        name_hint = f"The interviewee's name is '{interviewee_name}'." if interviewee_name else ""

        system_msg = (
            "You are a JSON-only API. You NEVER write explanations, sentences, or analysis. "
            "You output ONLY a single raw JSON object and nothing else. "
            "No markdown, no backticks, no preamble, no postamble. Just the JSON."
        )

        prompt = f"""Image: screenshot from a Google Meet / Zoom interview.
{name_hint}

Task: identify the interviewee tile (person answering questions, usually largest tile).
Important:

- There may be multiple people visible (interviewer + interviewee)
- The interviewee is the person answering questions
- The interviewee's name is: {interviewee_name}

PRIORITY RULES:

1. If a name label matching the interviewee_name is visible, you MUST choose that tile
2. If multiple faces exist, you MUST NOT choose randomly
3. The face linked to interviewee_name has highest priority over all other faces
4. If no name label is visible, choose the person who appears to be the main speaker (larger or more centered tile)
5. IGNORE code/editor or screen share content completely
6. Always return a region containing a human face, never a screen area
7. Do NOT assume the largest tile is always the interviewee
8. The bounding box MUST contain ONLY ONE face (never multiple people)

FAIL CONDITIONS (DO NOT DO):
- Do NOT return a wide region covering multiple participants
- Do NOT include interviewer + interviewee together
- Do NOT select empty or screen regions

Output ONLY this JSON object, nothing else:
{{"interviewee_name_on_screen":"<name or empty string>","region":{{"x1_pct":<0.0-1.0>,"y1_pct":<0.0-1.0>,"x2_pct":<0.0-1.0>,"y2_pct":<0.0-1.0>}},"layout_type":"<SINGLE|SIDE_BY_SIDE|GRID|BIG_PLUS_SMALL>","interviewer_side":"<LEFT|RIGHT|TOP|BOTTOM|NONE>","confidence":"<HIGH|MEDIUM|LOW>","reasoning":"<10 words max>"}}"""

        # ================= MODEL CALL =================
        raw_text = None
        last_err = None

        for attempt in range(3):
            try:
                response = groq_client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    max_tokens=400,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64_frame}",
                                    }
                                },
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    raw_text = content.strip()
                    break
                print(f"[VISION] Attempt {attempt+1}: empty response, retrying...")
            except Exception as e:
                last_err = e
                print(f"[VISION] Attempt {attempt+1} exception: {e}")

        if not raw_text:
            print(f"[VISION] All attempts failed (last={last_err}) -- falling back")
            return None

        # ================= PARSE =================
        import re

        text = raw_text.replace("```json", "").replace("```", "").strip()

        if not text.startswith("{"):
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                text = m.group(0)
            else:
                print(f"[VISION] No JSON found in response: {text[:200]}")
                return None

        # Try strict parse first, fall back to regex extraction
        parsed = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"[VISION] JSON parse error: {e} | raw: {text[:300]}")
            # Regex fallback — extract region coords and other fields individually
            region_match = re.search(
                r'"x1_pct"\s*:\s*([0-9.]+).*?"y1_pct"\s*:\s*([0-9.]+).*?"x2_pct"\s*:\s*([0-9.]+).*?"y2_pct"\s*:\s*([0-9.]+)',
                text, re.DOTALL
            )
            if not region_match:
                print("[VISION] Regex fallback also failed")
                return None

            confidence   = "HIGH" if "HIGH" in text else "MEDIUM" if "MEDIUM" in text else "LOW"
            layout_match = re.search(r'"layout_type"\s*:\s*"([^"]+)"', text)
            side_match   = re.search(r'"interviewer_side"\s*:\s*"([^"]+)"', text)
            name_match   = re.search(r'"interviewee_name_on_screen"\s*:\s*"([^"]*)"', text)

            parsed = {
                "region": {
                    "x1_pct": float(region_match.group(1)),
                    "y1_pct": float(region_match.group(2)),
                    "x2_pct": float(region_match.group(3)),
                    "y2_pct": float(region_match.group(4)),
                },
                "confidence":                 confidence,
                "layout_type":                layout_match.group(1) if layout_match else "",
                "interviewer_side":           side_match.group(1)   if side_match   else "NONE",
                "interviewee_name_on_screen": name_match.group(1)   if name_match   else "",
                "reasoning":                  "",
            }

        # ================= EXTRACT REGION =================
        region = parsed.get("region", {})

        x1_pct = float(region.get("x1_pct", 0))
        y1_pct = float(region.get("y1_pct", 0))
        x2_pct = float(region.get("x2_pct", 1))
        y2_pct = float(region.get("y2_pct", 1))

        # Clamp (safety)
        x1_pct = max(0.0, min(1.0, x1_pct))
        y1_pct = max(0.0, min(1.0, y1_pct))
        x2_pct = max(0.0, min(1.0, x2_pct))
        y2_pct = max(0.0, min(1.0, y2_pct))

        # Convert to pixels
        x1 = int(x1_pct * frame_w)
        y1 = int(y1_pct * frame_h)
        x2 = int(x2_pct * frame_w)
        y2 = int(y2_pct * frame_h)

        print(f"[VISION] Interviewee: '{parsed.get('interviewee_name_on_screen')}'")
        print(f"[VISION] Region (pct): ({x1_pct:.2f},{y1_pct:.2f}) → ({x2_pct:.2f},{y2_pct:.2f})")
        print(f"[VISION] Region (px): ({x1},{y1}) → ({x2},{y2})")
        print(f"[VISION] Confidence: {parsed.get('confidence')}")
        print(f"[VISION] Reason: {parsed.get('reasoning')}")

        # Reject garbage regions
        if (x2 - x1) < frame_w * 0.1 or (y2 - y1) < frame_h * 0.1:
            print("[VISION] Region too small → ignoring")
            return None

        interviewer_side = parsed.get("interviewer_side", "NONE").upper()
        if interviewer_side not in ["LEFT", "RIGHT"]:
            interviewer_side = "NONE"

        return {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "method": "vision_pct",
            "name_found":       parsed.get("interviewee_name_on_screen", ""),
            "confidence":       parsed.get("confidence", ""),
            "reasoning":        parsed.get("reasoning", ""),
            "layout_type":      parsed.get("layout_type", ""),
            "interviewer_side": interviewer_side,
        }

    except Exception as e:
        print(f"[VISION] Vision failed: {e}")
        return None
    
def find_largest_face_region(video_path: str, frame_w: int, frame_h: int) -> dict:
    cap       = cv2.VideoCapture(video_path)
    best_area = 0
    best_box  = None

    with mp_face_detect.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        for t in [5, 15, 30, 45, 60, 75, 90, 105, 120]:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                continue

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = fd.process(rgb)

            if not results.detections:
                continue

            for det in results.detections:
                bb   = det.location_data.relative_bounding_box
                w    = int(bb.width  * frame_w)
                h    = int(bb.height * frame_h)
                area = w * h
                if area > best_area:
                    best_area = area
                    x = int(bb.xmin * frame_w)
                    y = int(bb.ymin * frame_h)
                    best_box = (x, y, w, h)

    cap.release()

    if best_box:
        x, y, w, h = best_box
        pad_x = int(w * 0.6)
        pad_y = int(h * 0.8)
        x1    = max(0,       x - pad_x)
        y1    = max(0,       y - pad_y)
        x2    = min(frame_w, x + w + pad_x)
        y2    = min(frame_h, y + h + pad_y)
        print(f"[REGION] Largest face at ({x},{y}) size={w}x{h} -> region ({x1},{y1})-({x2},{y2})")
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "method": "largest_face"}

    print("[REGION] No face found, using full frame")
    return {"x1": 0, "y1": 0, "x2": frame_w, "y2": frame_h, "method": "full_frame"}


# ==============================================================================
# Head pose + gaze helpers
# ==============================================================================
def get_head_pose(landmarks, w, h):
    pts = np.array([
        (landmarks[1].x*w,   landmarks[1].y*h),
        (landmarks[152].x*w, landmarks[152].y*h),
        (landmarks[33].x*w,  landmarks[33].y*h),
        (landmarks[263].x*w, landmarks[263].y*h),
        (landmarks[61].x*w,  landmarks[61].y*h),
        (landmarks[291].x*w, landmarks[291].y*h),
    ], dtype=np.float64)
    cam  = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float64)
    dist = np.zeros((4,1))
    _, rvec, _ = cv2.solvePnP(MODEL_POINTS, pts, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    rmat, _    = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles


def get_iris_gaze(landmarks, iris_idx, eye_h_pair, w, h):
    def pt(i): return np.array([landmarks[i].x*w, landmarks[i].y*h])
    iris  = np.mean([pt(i) for i in iris_idx], axis=0)
    width = np.linalg.norm(pt(eye_h_pair[1]) - pt(eye_h_pair[0])) + 1e-6
    return (iris[0] - pt(eye_h_pair[0])[0]) / width


def classify_gaze_direction(yaw: float, pitch: float, gaze_x: float, interviewer_side: str = "NONE") -> str:
    if yaw < -35:
        raw = "LEFT"
    elif yaw > 35:
        raw = "RIGHT"
    elif pitch < -22:
        raw = "DOWN"
    elif gaze_x < 0.28:
        raw = "LEFT"
    elif gaze_x > 0.72:
        raw = "RIGHT"
    else:
        return "CENTER"

    return raw


def is_speaking(time_s: float, segments: list, window: float = 1.5) -> bool:
    for seg in segments:
        if seg["start"] <= time_s + window and seg["end"] >= time_s - window:
            return True
    return False


# ==============================================================================
# STEP 5 -- Detect repeated same-direction gaze patterns
# ==============================================================================
def detect_direction_patterns(gaze_log: list, min_occurrences: int = 4, window_s: float = 45.0) -> list:
    if not gaze_log:
        return []

    patterns = []

    glance_events          = []
    current_dir            = None
    current_start          = None
    CENTER_FRAMES_TO_RESET = 3
    center_streak          = 0

    for entry in gaze_log:
        d = entry["direction"]
        t = entry["time_s"]

        if d == "CENTER" or d == "ABSENT":
            center_streak += 1
            if center_streak >= CENTER_FRAMES_TO_RESET and current_dir:
                glance_events.append({"direction": current_dir, "time_s": current_start})
                current_dir   = None
                current_start = None
        else:
            center_streak = 0
            if d != current_dir:
                if current_dir:
                    glance_events.append({"direction": current_dir, "time_s": current_start})
                current_dir   = d
                current_start = t

    if current_dir:
        glance_events.append({"direction": current_dir, "time_s": current_start})

    print(f"[PATTERN] Total glance events: {len(glance_events)}")

    # Direction concentration check
    all_off_center = [e for e in glance_events if e["direction"] in ["LEFT","RIGHT","DOWN"]]
    if all_off_center:
        dir_counts = Counter(e["direction"] for e in all_off_center)
        for direction, count in dir_counts.items():
            concentration = count / len(all_off_center)
            if concentration >= 0.75 and count >= min_occurrences:
                print(f"[PATTERN] Direction concentration: {direction} = {concentration:.0%} of off-center looks")
                patterns.append({
                    "direction":    direction,
                    "time_s":       all_off_center[0]["time_s"],
                    "count":        count,
                    "duration_s":   round(all_off_center[-1]["time_s"] - all_off_center[0]["time_s"], 1),
                    "detail":       f"Gaze concentration: {concentration:.0%} of all off-center looks are {direction} ({count} times) -- not random thinking",
                    "trigger":      "concentration",
                })

    # Rolling window + gap analysis per direction
    for direction in ["LEFT", "RIGHT", "DOWN"]:
        dir_events = [e for e in glance_events if e["direction"] == direction]

        if len(dir_events) < min_occurrences:
            continue

        flagged_windows = []

        for i in range(len(dir_events)):
            window_events = [
                e for e in dir_events
                if dir_events[i]["time_s"] <= e["time_s"] <= dir_events[i]["time_s"] + window_s
            ]
            if len(window_events) < min_occurrences:
                continue

            if direction == "DOWN":
                if is_typing_pattern(window_events):
                    print("[PATTERN] Suppressed DOWN -- typing detected")
                    continue

            t_start = window_events[0]["time_s"]
            t_end   = window_events[-1]["time_s"]

            already_flagged = any(
                abs(f["time_s"] - t_start) < window_s / 2
                for f in flagged_windows
            )
            if already_flagged:
                continue

            gaps = [
                window_events[j]["time_s"] - window_events[j-1]["time_s"]
                for j in range(1, len(window_events))
            ]
            avg_gap = sum(gaps) / len(gaps) if gaps else 999
            max_gap = max(gaps) if gaps else 999
            gap_std = np.std(gaps) if len(gaps) > 1 else 999

            is_reading_rhythm = avg_gap < 6 and gap_std < 2.5

            if not is_reading_rhythm:
                continue

            rhythm_note = f"avg gap={avg_gap:.1f}s std={gap_std:.1f}s -- confirmed reading rhythm"

            detail = (
                f"Looked {direction} {len(window_events)} times in {t_end-t_start:.0f}s window "
                f"(t={t_start:.0f}s-{t_end:.0f}s) | {rhythm_note} | "
                f"possible {'notes on screen' if direction in ['LEFT','RIGHT'] else 'notes on desk/lap'}"
            )

            flagged_windows.append({
                "direction":         direction,
                "time_s":            round(t_start, 2),
                "count":             len(window_events),
                "duration_s":        round(t_end - t_start, 1),
                "avg_gap_s":         round(avg_gap, 1),
                "is_reading_rhythm": is_reading_rhythm,
                "detail":            detail,
                "trigger":           "rolling_window",
            })

        if flagged_windows:
            print(f"[PATTERN] {direction}: {len(flagged_windows)} suspicious windows")
            patterns.extend(flagged_windows)

    return sorted(patterns, key=lambda p: p["time_s"])


def is_typing_pattern(events):
    down_count   = sum(1 for e in events if e["direction"] == "DOWN")
    center_count = sum(1 for e in events if e["direction"] == "CENTER")
    total        = len(events)
    if total == 0:
        return False
    if down_count / total > 0.35 and center_count / total > 0.2:
        return True
    return False


def time_since_last_speech(t, segments):
    last_end = 0
    for seg in segments:
        if seg["end"] <= t:
            last_end = seg["end"]
    return t - last_end


def is_interviewer_speaking(t, segments):
    for seg in segments:
        if seg["start"] <= t <= seg["end"]:
            text = seg.get("text", "").lower()
            if "?" in text or len(text.split()) > 12:
                return True
    return False


# ==============================================================================
# Speech naturalness analysis (local, zero API cost)
# ==============================================================================

def _score_segment_locally(text: str, word_count: int, pause_before_s: float) -> dict:
    signals = {}
    score   = 0

    conn_hits = STRUCTURED_CONNECTORS.findall(text)
    if conn_hits:
        signals["structured_connectors"] = conn_hits
        score += min(len(conn_hits) * 15, 30)

    book_hits = BOOKISH_PHRASES.findall(text)
    if book_hits:
        signals["bookish_phrases"] = book_hits
        score += min(len(book_hits) * 12, 24)

    if word_count >= 30:
        if not HESITATION_WORDS.search(text) and not SELF_CORRECTIONS.search(text):
            signals["zero_hesitation"] = True
            score += 20

    if word_count >= 25:
        if not CONTRACTIONS.search(text):
            signals["no_contractions"] = True
            score += 10

    if pause_before_s >= 4.0 and word_count >= 20:
        signals["pause_then_clean"] = round(pause_before_s, 1)
        score += min(int(pause_before_s * 2), 20)

    return {"signals": signals, "local_score": min(score, 100)}


def _build_speech_reason(signals: dict) -> str:
    parts = []
    if "structured_connectors" in signals:
        parts.append(f"structured connectors ({signals['structured_connectors']})")
    if "bookish_phrases" in signals:
        parts.append(f"formal/bookish phrases ({signals['bookish_phrases']})")
    if signals.get("zero_hesitation"):
        parts.append("long answer with zero hesitation words")
    if signals.get("no_contractions"):
        parts.append("no contractions in long answer")
    if "pause_then_clean" in signals:
        parts.append(f"clean burst after {signals['pause_then_clean']}s pause")
    return "; ".join(parts) if parts else "multiple scripted-speech signals"


def analyze_speech_naturalness(transcript_segments: list, min_local_score: int = 30) -> dict:
    """
    Detects AI-assisted / scripted speech using local regex signals only.
    No API calls — zero extra cost.

    Signals: structured connectors, bookish formality, zero hesitation,
             no contractions, pause → clean burst.
    """
    empty = {
        "flagged_segments":        [],
        "overall_ai_speech_score": 0,
        "speech_risk":             "LOW",
        "summary_signals":         [],
        "violations":              [],
    }

    if not transcript_segments:
        return empty

    print(f"\n[SPEECH] Analyzing {len(transcript_segments)} segments for AI-like speech (local, no API)...")

    # Enrich segments with pause_before_s
    enriched = []
    for i, seg in enumerate(transcript_segments):
        pause = max(0.0, seg["start"] - transcript_segments[i-1]["end"]) if i > 0 else 0.0
        enriched.append({
            "start_s":        seg["start"],
            "end_s":          seg["end"],
            "text":           seg["text"],
            "word_count":     len(seg["text"].split()),
            "pause_before_s": round(pause, 2),
        })

    # Score every segment
    all_flagged = []
    for seg in enriched:
        scored = _score_segment_locally(seg["text"], seg["word_count"], seg["pause_before_s"])
        if scored["local_score"] >= min_local_score:
            all_flagged.append({
                "start_s":        seg["start_s"],
                "end_s":          seg["end_s"],
                "text":           seg["text"],
                "ai_probability": scored["local_score"],
                "signals":        list(scored["signals"].keys()),
                "reason":         _build_speech_reason(scored["signals"]),
            })

    all_flagged.sort(key=lambda x: x["start_s"])
    print(f"[SPEECH] {len(all_flagged)} suspicious segments (score >= {min_local_score})")

    # Overall score
    if not all_flagged:
        overall = 0
    else:
        avg        = sum(f["ai_probability"] for f in all_flagged) / len(all_flagged)
        freq_boost = min(len(all_flagged) * 4, 25)
        overall    = min(100, int(avg * 0.75 + freq_boost))

    # Summary signals
    signal_counter: dict = {}
    for f in all_flagged:
        for s in f["signals"]:
            signal_counter[s] = signal_counter.get(s, 0) + 1
    summary_signals = [
        f"{s} (x{c})"
        for s, c in sorted(signal_counter.items(), key=lambda x: -x[1])
    ]

    # Risk level
    if overall >= 55:
        speech_risk = "HIGH"
    elif overall >= 25:
        speech_risk = "MEDIUM"
    else:
        speech_risk = "LOW"

    # Violations with cooldown
    violations     = []
    COOLDOWN_S     = 20.0
    last_flagged_t = -999.0

    for f in all_flagged:
        if f["ai_probability"] < 40:
            continue
        if f["start_s"] - last_flagged_t < COOLDOWN_S:
            continue

        sev     = "HIGH" if f["ai_probability"] >= 65 else "MEDIUM"
        excerpt = f["text"][:80] + ("..." if len(f["text"]) > 80 else "")
        violations.append({
            "frame":    0,
            "time_s":   round(f["start_s"], 2),
            "type":     "ai_assisted_speech",
            "detail":   (
                f"[{f['ai_probability']}% scripted] {f['reason']} | "
                f"signals={f['signals']} | excerpt: \"{excerpt}\""
            ),
            "severity": sev,
        })
        last_flagged_t = f["start_s"]

    print(f"[SPEECH] overall_score={overall} | risk={speech_risk} | violations={len(violations)}")
    for v in violations:
        print(f"         [{v['severity']}] t={v['time_s']}s  {v['detail'][:110]}")

    return {
        "flagged_segments":        all_flagged,
        "overall_ai_speech_score": overall,
        "speech_risk":             speech_risk,
        "summary_signals":         summary_signals,
        "violations":              violations,
    }

def detect_screen_share_time(transcript_segments):
    keywords = [
        "share your screen",
        "can you share screen",
        "please share screen",
        "start sharing",
        "screen share",
        "show your code",
        "open your editor",
        "write code"
    ]

    for seg in transcript_segments:
        text = seg["text"].lower()

        if any(k in text for k in keywords):
            print(f"[SCREEN SHARE] Detected at t={seg['start']}s -> '{seg['text'][:60]}'")
            return seg["start"]

    return None

def get_frame_at_time(video_path, time_s):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[SCREEN SHARE] Failed to grab frame")
        return None

    return frame

def openai_vision_find_interviewee_from_frame(frame, interviewee_name, frame_w, frame_h):
    try:
        # Encode frame
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_frame = base64.b64encode(buf).decode("utf-8")

        name_hint = f"The interviewee's name is '{interviewee_name}'." if interviewee_name else ""

        system_msg = "You are a JSON-only API. Output ONLY valid JSON. "
        "Do not include quotes inside numbers. No trailing commas. No explanations."

        prompt = f"""
Image: screenshot from a coding interview after screen share.
{name_hint}

Important:

- Layout may have changed (screen share active)
- There may be multiple people visible (interviewer + interviewee)
- The interviewee is the person answering questions
- The interviewee's name is: {interviewee_name}

PRIORITY RULES:
1. If a name label matching the interviewee_name is visible, choose that tile
2. If multiple faces exist, DO NOT choose randomly
3. Prefer the face associated with the interviewee_name label
4. If name is not visible, choose the main speaker (larger/center tile)
5. IGNORE code/editor or screen share content completely
6. Always return a region containing a human face
7. IGNORE code/editor or screen share content completely
8. Always return a region containing a human face, never a screen area
9. Do NOT assume the largest tile is always the interviewee
10. The bounding box MUST contain ONLY ONE face (never multiple people)

FAIL CONDITIONS (DO NOT DO):
- Do NOT return a wide region covering multiple participants
- Do NOT include interviewer + interviewee together
- Do NOT select empty or screen regions


Return ONLY:
{{"region":{{"x1_pct":0-1,"y1_pct":0-1,"x2_pct":0-1,"y2_pct":0-1}},
"confidence":"HIGH|MEDIUM|LOW",
"reason":"<10 words>"}}"""

        # CALL MODEL
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=300,
            messages=[
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_frame}",
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        )
        import re
        import json

        raw = response.choices[0].message.content

        print("[VISION-SS RAW]:", raw[:200])

        if not raw:
            return None

        # Remove markdown
        text = raw.replace("```json", "").replace("```", "").strip()

        
        # Extract ONLY the region block manually
        region_match = re.search(
            r'"x1_pct"\s*:\s*([0-9.]+)[,\s]+(?:"y1_pct"\s*:\s*)?([0-9.]+).*?"x2_pct"\s*:\s*([0-9.]+)[,\s]+(?:"y2_pct"\s*:\s*)?([0-9.]+)',
            text,
            re.DOTALL
        )
        if not region_match:
            print("[VISION-SS] Could not extract region")
            return None

        try:
            x1_pct = float(region_match.group(1))
            y1_pct = float(region_match.group(2))
            x2_pct = float(region_match.group(3))
            y2_pct = float(region_match.group(4))
        except:
            print("[VISION-SS] Failed to parse numbers")
            return None

        # Optional fields
        confidence = "LOW"
        if "HIGH" in text:
            confidence = "HIGH"
        elif "MEDIUM" in text:
            confidence = "MEDIUM"

        # Convert to pixels
        x1 = int(x1_pct * frame_w)
        y1 = int(y1_pct * frame_h)
        x2 = int(x2_pct * frame_w)
        y2 = int(y2_pct * frame_h)

        return {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "confidence": confidence,
            "method": "vision_screen_share"
        }
    except Exception as e:
        print(f"[VISION-SS] Failed: {e}")
        return None
# ==============================================================================
# Main detector class
# ==============================================================================
class InterviewCheatingDetector:
    def __init__(
        self,
        video_path:             str,
        student_id:             str   = "",
        process_every_n_frames: int   = 3,
        offscreen_duration_s:   float = 8.0,
        cooldown_s:             float = 15.0,
        dir_min_occurrences:    int   = 6,
        dir_window_s:           float = 35.0,
    ):
        self.video_path    = video_path
        self.student_id    = student_id or Path(video_path).stem
        self.process_every = process_every_n_frames
        self.offscreen_dur = offscreen_duration_s
        self.cooldown      = cooldown_s
        self.dir_min_occ   = dir_min_occurrences
        self.dir_window    = dir_window_s
        self._last_v: dict = {}

    def _can_add(self, vtype, t):
        if t - self._last_v.get(vtype, -999) >= self.cooldown:
            self._last_v[vtype] = t
            return True
        return False

    def analyze(self) -> DetectionResult:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {self.video_path}")
        fps        = cap.get(cv2.CAP_PROP_FPS) or 25
        total_f    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_f / fps
        frame_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print(f"\n{'='*60}")
        print(f"[ANALYZE] {self.student_id}")
        print(f"[ANALYZE] {frame_w}x{frame_h} | {duration_s:.0f}s | {total_f} frames @ {fps:.0f}fps")
        print(f"{'='*60}")

        result = DetectionResult(
            student_id=self.student_id, video_path=self.video_path,
            duration_s=duration_s, total_frames=total_f, processed_frames=0,
        )

        # -- PHASE 1: Audio -> Whisper -> GPT-4o name extraction -----------------
        print(f"\n[PHASE 1] Audio transcription + interviewee identification")
        transcript_full     = ""
        transcript_summary  = ""
        interview_topic     = ""
        interviewee_name    = None
        transcript_segments = []
        has_transcript      = False

        audio_path = extract_audio(self.video_path)
        if audio_path:
            transcript = transcribe_audio(audio_path)
            if transcript:
                transcript_full     = transcript["full_text"]
                transcript_segments = transcript.get("segments", [])
                has_transcript      = len(transcript_segments) > 0

                print(f"[PHASE 1] Asking GPT-4o-MINI to identify interviewee...")
                info = identify_interviewee_name(transcript)
                interviewee_name   = info.get("interviewee_name")
                transcript_summary = info.get("summary", "")
                interview_topic    = info.get("interview_topic", "")

            try: Path(audio_path).unlink()
            except: pass
        else:
            print("[PHASE 1] No audio -- gaze thresholds will be raised significantly to avoid false positives")

        result.interviewee_name   = interviewee_name or "unknown"
        result.transcript_full    = transcript_full
        result.transcript_summary = transcript_summary
        result.interview_topic    = interview_topic
        
        screen_share_time = None

        if has_transcript:
            screen_share_time = detect_screen_share_time(transcript_segments)

            if screen_share_time:
                screen_share_time +=120  # buffer after instruction
                print(f"[SCREEN SHARE] Adjusted time: {screen_share_time}s")

        # -- PHASE 1b: Speech naturalness analysis (local, zero API cost) --------
        print(f"\n[PHASE 1b] Speech naturalness / AI-assisted speech detection")
        violations_from_speech  = []
        speech_naturalness_result = {
            "flagged_segments": [], "overall_ai_speech_score": 0,
            "speech_risk": "LOW", "summary_signals": [], "violations": [],
        }

        if has_transcript and transcript_segments:
            speech_naturalness_result  = analyze_speech_naturalness(transcript_segments)
            result.speech_naturalness  = speech_naturalness_result

            for sv in speech_naturalness_result["violations"]:
                violations_from_speech.append(
                    Violation(
                        frame    = sv["frame"],
                        time_s   = sv["time_s"],
                        type     = sv["type"],
                        detail   = sv["detail"],
                        severity = sv["severity"],
                    )
                )
        else:
            print("[PHASE 1b] Skipping -- no transcript available")
            result.speech_naturalness = speech_naturalness_result

        # -- PHASE 2: Find interviewee region ------------------------------------
        print(f"\n[PHASE 2] Locating interviewee on screen")
        region = find_interviewee_region_by_ocr(
            self.video_path, self.student_id, frame_w, frame_h
        )
        result.interviewee_region = region
        x1, y1, x2, y2  = region["x1"], region["y1"], region["x2"], region["y2"]
        crop_w           = x2 - x1
        crop_h           = y2 - y1
        interviewer_side = region.get("interviewer_side", "NONE")
        print(f"[PHASE 2] Region: ({x1},{y1})->({x2},{y2}) size={crop_w}x{crop_h} method={region['method']}")
        
        
        secondary_region = None
        secondary_start_time = None

        if screen_share_time:
            print(f"[SCREEN SHARE] Capturing frame at t={screen_share_time}s")

            ss_frame = get_frame_at_time(self.video_path, screen_share_time)

            if ss_frame is not None:
                

                # Run vision AGAIN on this frame
                new_region = openai_vision_find_interviewee_from_frame(
                    ss_frame,
                    self.student_id,
                    frame_w,
                    frame_h
                )

                if new_region:
                    secondary_region = new_region
                    secondary_start_time = screen_share_time

                    print(f"[SCREEN SHARE] New region: {secondary_region}")

                    sx1, sy1, sx2, sy2 = (
                        new_region["x1"],
                        new_region["y1"],
                        new_region["x2"],
                        new_region["y2"]
                    )

                    crop_ss = ss_frame[sy1:sy2, sx1:sx2]
                   
            

        if has_transcript:
            down_flag_dur = 20.0
            lr_flag_dur   = self.offscreen_dur
            down_flag_msg = "(silent per transcript)"
        else:
            down_flag_dur = 45.0
            lr_flag_dur   = 25.0
            down_flag_msg = "(no transcript -- conservative threshold)"

        print(f"[PHASE 3] Thresholds -- DOWN: {down_flag_dur}s | LEFT/RIGHT: {lr_flag_dur}s | transcript={'YES' if has_transcript else 'NO'}")

        # -- PHASE 3: Frame-by-frame gaze detection ------------------------------
        print(f"\n[PHASE 3] Gaze-only detection (clean logic)")

        cap        = cv2.VideoCapture(self.video_path)
        violations = []
        gaze_log   = []

        bucket_size = 10
        n_buckets   = max(1, int(duration_s / bucket_size) + 1)
        timeline    = [{"time_s": i*bucket_size, "violations": 0} for i in range(n_buckets)]

        offscreen_start = None
        offscreen_dir   = None
        down_start_time = None
        processed       = 0

        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh, \
        mp_face_detect.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        ) as face_det:

            frame_idx = 0
            switched_logged=False
            last_speaking_log_time=-999
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx   += 1
                current_time = frame_idx / fps

                if frame_idx % self.process_every != 0:
                    continue

                processed += 1

                # decide which region to use
                if secondary_region and current_time >= secondary_start_time:
                    if not switched_logged:
                        print("switched at ",current_time)
                        rx1, ry1, rx2, ry2 = (
                            secondary_region["x1"],
                            secondary_region["y1"],
                            secondary_region["x2"],
                            secondary_region["y2"]
                        )
                        switched_logged=True
                        down_flag_dur=30
                else:
                    rx1, ry1, rx2, ry2 = x1, y1, x2, y2

                crop = frame[ry1:ry2, rx1:rx2]
                if crop.size == 0:
                    continue
                
                

                rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                bucket = min(int(current_time / bucket_size), n_buckets - 1)

                def add_v(vtype, detail, severity="MEDIUM"):
                    if self._can_add(vtype, current_time):
                        violations.append(
                            Violation(frame_idx, round(current_time, 2), vtype, detail, severity)
                        )
                        timeline[bucket]["violations"] += 1

                fd = face_det.process(rgb)

                if not fd.detections:
                    gaze_log.append({"time_s": round(current_time, 2), "direction": "ABSENT"})
                    offscreen_start = None
                    offscreen_dir   = None
                    continue

                mesh = face_mesh.process(rgb)
                if not mesh.multi_face_landmarks:
                    continue

                lm = mesh.multi_face_landmarks[0].landmark

                try:
                    pitch, yaw, roll = get_head_pose(lm, crop_w, crop_h)
                except:
                    pitch, yaw, roll = 0, 0, 0

                try:
                    lg     = get_iris_gaze(lm, LEFT_IRIS,  LEFT_EAR_H,  crop_w, crop_h)
                    rg     = get_iris_gaze(lm, RIGHT_IRIS, RIGHT_EAR_H, crop_w, crop_h)
                    gaze_x = (lg + rg) / 2
                except:
                    gaze_x = 0.5

                direction = classify_gaze_direction(yaw, pitch, gaze_x, interviewer_side)

                if direction == "DOWN":
                    if down_start_time is None:
                        down_start_time = current_time
                    if current_time - down_start_time < 0.8:
                        continue
                else:
                    down_start_time = None

                gaze_log.append({"time_s": round(current_time, 2), "direction": direction})

                if direction in ["LEFT", "RIGHT", "DOWN"]:

                    if offscreen_start is None:
                        offscreen_start = current_time
                        offscreen_dir   = direction
                        continue

                    elapsed       = current_time - offscreen_start
                    recent_events = gaze_log[-40:]

                    speaking             = is_speaking(current_time, transcript_segments) if has_transcript else False
                    interviewer_speaking = is_interviewer_speaking(current_time, transcript_segments) if has_transcript else False

                    if speaking or interviewer_speaking:
                        if current_time-last_speaking_log_time>5:
                           print(f"[GAZE] Suppressed at t={current_time:.0f}s -- Speaking/Listening detected")
                           last_speaking_log_time=current_time
                        offscreen_start = None
                        offscreen_dir   = None
                        continue

                    if direction == "DOWN" and has_transcript and is_typing_pattern(recent_events):
                        print(f"[GAZE] Suppressed DOWN at t={current_time:.0f}s -- typing detected")
                        offscreen_start = None
                        offscreen_dir   = None
                        continue

                    center_count = sum(1 for e in recent_events if e["direction"] == "CENTER")
                    if center_count > len(recent_events) * 0.3:
                        offscreen_start = None
                        continue

                    threshold = down_flag_dur if direction == "DOWN" else lr_flag_dur

                    if elapsed >= threshold:
                        silence_gap = (
                            time_since_last_speech(current_time, transcript_segments)
                            if has_transcript else 999
                        )
                        if silence_gap > 6:
                            sev = "LOW" if direction == "DOWN" and not has_transcript else "MEDIUM"
                            add_v(
                                "sustained_gaze",
                                f"Looking {direction} continuously for {elapsed:.1f}s {down_flag_msg}",
                                sev
                            )
                        offscreen_start = current_time
                        offscreen_dir   = direction

                else:
                    offscreen_start = None
                    offscreen_dir   = None

        cap.release()

        # Merge speech violations into main list
        violations.extend(violations_from_speech)

        print(f"[PHASE 3] Processed {processed} frames, {len(gaze_log)} gaze readings, {len(violations)} raw violations")

        # -- PHASE 4: Pattern analysis -------------------------------------------
        print(f"\n[PHASE 4] Gaze direction pattern analysis")

        patterns = detect_direction_patterns(
            gaze_log,
            min_occurrences=self.dir_min_occ,
            window_s=self.dir_window,
        )

        direction_counts = Counter(g["direction"] for g in gaze_log)
        total_tracked    = len(gaze_log) or 1

        gaze_pattern = {
            "LEFT_pct":             round(direction_counts.get("LEFT",   0) / total_tracked * 100, 1),
            "RIGHT_pct":            round(direction_counts.get("RIGHT",  0) / total_tracked * 100, 1),
            "DOWN_pct":             round(direction_counts.get("DOWN",   0) / total_tracked * 100, 1),
            "CENTER_pct":           round(direction_counts.get("CENTER", 0) / total_tracked * 100, 1),
            "ABSENT_pct":           round(direction_counts.get("ABSENT", 0) / total_tracked * 100, 1),
            "total_glances_logged": len(gaze_log),
            "repeated_patterns":    patterns,
            "has_transcript":       has_transcript,
        }

        print(f"[PHASE 4] LEFT={gaze_pattern['LEFT_pct']}% RIGHT={gaze_pattern['RIGHT_pct']}% DOWN={gaze_pattern['DOWN_pct']}% CENTER={gaze_pattern['CENTER_pct']}% ABSENT={gaze_pattern['ABSENT_pct']}%")
        print(f"[PHASE 4] Suspicious patterns: {len(patterns)}")

        for p in patterns:
            print(f"          -> {p['detail']}")

        for p in patterns:
            if self._can_add(f"pattern{p['direction']}", p["time_s"]):
                violations.append(Violation(
                    0, p["time_s"], "repeated_direction_pattern", p["detail"], "HIGH"
                ))

        # -- SCORING ---------------------------------------------------------------
        counts = {
            "sustained_gaze":             sum(1 for v in violations if v.type == "sustained_gaze"),
            "repeated_direction_pattern": sum(1 for v in violations if v.type == "repeated_direction_pattern"),
            "ai_assisted_speech":         sum(1 for v in violations if v.type == "ai_assisted_speech"),
        }

        down_patterns = sum(1 for p in patterns if p["direction"] == "DOWN" and p["count"] >= 6)
        lr_patterns   = sum(1 for p in patterns if p["direction"] in ["LEFT", "RIGHT"])

        lr_score      = min(lr_patterns,   5) * 8
        if(gaze_pattern["DOWN_pct"]>80):
            down_patterns=0
        down_score    = min(down_patterns, 8) * 4
        pattern_score = lr_score + down_score

        sustained_score = min(counts["sustained_gaze"], 6) * 4

        raw = pattern_score + sustained_score

        speech_score = speech_naturalness_result.get("overall_ai_speech_score", 0)
        raw += int(speech_score * 0.45)

        dominant_pct = max(
            gaze_pattern["LEFT_pct"],
            gaze_pattern["RIGHT_pct"],
            gaze_pattern["DOWN_pct"]
        )
        if dominant_pct > 75:
            boost = int((dominant_pct - 75) * 0.6)
            raw  += boost
            print(f"[SCORE] Direction dominance boost: +{boost} ({dominant_pct:.0f}% off-center)")

        if speech_score >= 55 and (lr_patterns + down_patterns) >= 2:
            pre_boost = raw
            raw = int(raw * 1.25)
            print(f"[SCORE] Cross-signal multiplier applied (speech={speech_score}, gaze_patterns={lr_patterns + down_patterns}): {pre_boost} → {raw}")

        score = min(100, raw)
        if score >= 55:
            risk = "HIGH"
        elif score >= 25:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        # -- SUMMARY ---------------------------------------------------------------
        parts = []

        if not has_transcript:
            parts.append("[WARN] No transcript -- gaze-only analysis")

        if counts["repeated_direction_pattern"] > 0:
            dirs     = [p["direction"] for p in patterns]
            dominant = Counter(dirs).most_common(1)[0][0] if dirs else "unknown"
            parts.append(f"Repeated {dominant} look {counts['repeated_direction_pattern']}x")

        if counts["sustained_gaze"] > 0:
            parts.append(f"Sustained off-screen gaze {counts['sustained_gaze']}x")

        if gaze_pattern["LEFT_pct"] > 30:
            parts.append(f"Eyes LEFT {gaze_pattern['LEFT_pct']}%")

        if gaze_pattern["RIGHT_pct"] > 30:
            parts.append(f"Eyes RIGHT {gaze_pattern['RIGHT_pct']}%")

        if gaze_pattern["DOWN_pct"] > 35:
            parts.append(f"Eyes DOWN {gaze_pattern['DOWN_pct']}%")

        speech_risk = speech_naturalness_result.get("speech_risk", "LOW")
        if speech_risk in ["MEDIUM", "HIGH"]:
            sig_str = ", ".join(speech_naturalness_result.get("summary_signals", [])[:3])
            parts.append(f"AI-speech risk={speech_risk} [{sig_str}]")

        summary = " | ".join(parts) if parts else "No suspicious activity detected"

        # -- FINAL OUTPUT ----------------------------------------------------------
        print(f"\n[RESULT] Score={score}/100 | Risk={risk}")
        print(f"[RESULT] {summary}")
        print(f"[RESULT] Violations breakdown: {counts}")
        print(f"[RESULT] All violations ({len(violations)} total):")

        for v in violations:
            print(f"         [{v.severity}] t={v.time_s}s  {v.type}: {v.detail}")

        print(f"{'='*60}\n")

        result.violations       = violations
        result.processed_frames = processed
        result.cheating_score   = score
        result.risk_level       = risk
        result.counts           = counts
        result.gaze_pattern     = gaze_pattern
        result.timeline         = timeline
        result.summary          = summary

        return result
