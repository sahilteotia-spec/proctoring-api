# -*- coding: utf-8 -*-



import os
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

# -- API client (OpenAI only) ---------------------------------------------------
openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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


@dataclass
class Violation:
    frame:    int
    time_s:   float
    type:     str
    detail:   str
    severity: str


@dataclass
class DetectionResult:
    student_id:         str
    video_path:         str
    duration_s:         float
    total_frames:       int
    processed_frames:   int
    interviewee_name:   str  = "unknown"
    interviewee_region: dict = field(default_factory=dict)
    transcript_full:    str  = ""
    transcript_summary: str  = ""
    interview_topic:    str  = ""
    violations:         list = field(default_factory=list)
    gaze_pattern:       dict = field(default_factory=dict)
    cheating_score:     int  = 0
    risk_level:         str  = "LOW"
    counts:             dict = field(default_factory=dict)
    timeline:           list = field(default_factory=list)
    summary:            str  = ""

    def to_dict(self):
        return {
            "student_id":         self.student_id,
            "video_path":         self.video_path,
            "duration_s":         round(self.duration_s, 2),
            "total_frames":       self.total_frames,
            "processed_frames":   self.processed_frames,
            "interviewee_name":   self.interviewee_name,
            "interviewee_region": self.interviewee_region,
            "transcript_full":    self.transcript_full,
            "transcript_summary": self.transcript_summary,
            "interview_topic":    self.interview_topic,
            "cheating_score":     self.cheating_score,
            "risk_level":         self.risk_level,
            "total_violations":   len(self.violations),
            "gaze_pattern":       self.gaze_pattern,
            "counts":             self.counts,
            "violations":         [v.__dict__ for v in self.violations],  # ALL violations, no cap
            "timeline":           self.timeline,
            "summary":            self.summary,
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
        print(f"[WHISPER] Audio: {size_mb:.1f}MB, {duration_s:.0f}s")

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
                response = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )

            full_text_parts.append(response.text)

            for seg in (response.segments or []):
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
        print(f"[WHISPER] Failed: {e}")
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
# STEP 3 -- OpenAI identifies interviewee name + summary
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
            model="gpt-4o",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        text   = response.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        parsed = json.loads(text)
        print(f"[GPT] Interviewee: {parsed.get('interviewee_name')} | Topic: {parsed.get('interview_topic')} | Confidence: {parsed.get('confidence')}")
        return parsed

    except Exception as e:
        print(f"[GPT] identify_interviewee_name failed: {e}")
        return {"interviewee_name": None, "summary": f"Error: {e}", "interview_topic": "unknown"}


# ==============================================================================
# STEP 4 -- OCR: find name label on screen, get face region above it
# ==============================================================================
def find_interviewee_region_by_ocr(video_path: str, interviewee_name: Optional[str], frame_w: int, frame_h: int) -> dict:
    if interviewee_name:
        region = _ocr_find_name(video_path, interviewee_name, frame_w, frame_h)
        if region:
            return region
        print("[REGION] OCR failed -- trying OpenAI Vision")

    region = _openai_vision_find_interviewee(video_path, interviewee_name, frame_w, frame_h)
    if region:
        return region
    print("[REGION] OpenAI Vision failed -- using largest face fallback")

    return find_largest_face_region(video_path, frame_w, frame_h)


def _ocr_find_name(video_path: str, interviewee_name: str, frame_w: int, frame_h: int) -> Optional[dict]:
    try:
        import pytesseract
        from PIL import Image

        cap        = cv2.VideoCapture(video_path)
        first_name = interviewee_name.split()[0].lower()
        last_name  = interviewee_name.split()[-1].lower() if len(interviewee_name.split()) > 1 else ""

        for t in [5, 15, 30, 60, 90, 120, 180]:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                continue

            strip_y = int(frame_h * 0.65)
            bottom  = frame[strip_y:, :]
            gray    = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)

            for thresh_val in [160, 180, 200]:
                _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
                data = pytesseract.image_to_data(
                    Image.fromarray(thresh),
                    output_type=pytesseract.Output.DICT,
                    config="--psm 11 --oem 3"
                )

                for i, word in enumerate(data["text"]):
                    word_lower = word.lower().strip()
                    if not word_lower or len(word_lower) < 3:
                        continue

                    if first_name in word_lower or (last_name and last_name in word_lower):
                        nx = data["left"][i]
                        ny = data["top"][i] + strip_y
                        nw = max(data["width"][i], 80)

                        print(f"[OCR] Matched '{word}' at t={t}s pos=({nx},{ny})")
                        cap.release()

                        tile_x1 = max(0,       nx - 60)
                        tile_x2 = min(frame_w, nx + nw + 200)
                        tile_w  = tile_x2 - tile_x1
                        tile_h  = int(tile_w * 9 / 16)
                        tile_y1 = max(0,       ny - tile_h)
                        tile_y2 = min(frame_h, ny + 30)

                        return {
                            "x1": tile_x1, "y1": tile_y1,
                            "x2": tile_x2, "y2": tile_y2,
                            "method": "ocr",
                            "name_found": word,
                        }

        cap.release()
        return None

    except ImportError:
        print("[OCR] pytesseract not installed")
        return None
    except Exception as e:
        print(f"[OCR] Error: {e}")
        return None


def _openai_vision_find_interviewee(video_path: str, interviewee_name: Optional[str], frame_w: int, frame_h: int) -> Optional[dict]:
    """Uses GPT-4o vision to locate the interviewee tile."""
    try:
        import base64

        cap          = cv2.VideoCapture(video_path)
        frame_to_use = None

        for t in [30, 60, 15, 90, 10]:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if ret and frame is not None:
                if cv2.mean(frame)[0] > 20:
                    frame_to_use = frame
                    print(f"[VISION] Using frame at t={t}s")
                    break
        cap.release()

        if frame_to_use is None:
            return None

        _, buf    = cv2.imencode(".jpg", frame_to_use, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_frame = base64.b64encode(buf).decode("utf-8")
        name_hint = f"The interviewee's name is '{interviewee_name}'." if interviewee_name else ""

        prompt = f"""This is a screenshot from a video interview recorded on Google Meet or Zoom.
{name_hint}

The interviewee is the person BEING interviewed (answering questions about their experience/skills).
The interviewer is the person ASKING the questions.

Look at the screen carefully:
- Name labels appear at the bottom of each person's video tile
- The layout could be: one big tile, side by side, big+small thumbnail, etc.
- There may be an avatar/initial circle if someone has camera off

The full frame is {frame_w}x{frame_h} pixels.

Respond ONLY in this exact JSON format, no other text:
{{
  "interviewee_name_on_screen": "name you can read from the screen",
  "tile_x1": left edge pixel of interviewee tile,
  "tile_y1": top edge pixel of interviewee tile,
  "tile_x2": right edge pixel of interviewee tile,
  "tile_y2": bottom edge pixel of interviewee tile,
  "interviewer_side": "LEFT or RIGHT or TOP or BOTTOM or NONE",
  "confidence": "HIGH or MEDIUM or LOW",
  "reasoning": "brief explanation"
}}"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":    f"data:image/jpeg;base64,{b64_frame}",
                            "detail": "high",
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
        )

        text   = response.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        parsed = json.loads(text)

        x1 = max(0,       int(parsed["tile_x1"]))
        y1 = max(0,       int(parsed["tile_y1"]))
        x2 = min(frame_w, int(parsed["tile_x2"]))
        y2 = min(frame_h, int(parsed["tile_y2"]))

        print(f"[VISION] GPT-4o identified: '{parsed.get('interviewee_name_on_screen')}' at ({x1},{y1})->({x2},{y2}) confidence={parsed.get('confidence')}")
        print(f"[VISION] Reasoning: {parsed.get('reasoning')}")

        if x2 - x1 < 50 or y2 - y1 < 50:
            print("[VISION] Region too small, ignoring")
            return None

        interviewer_side = parsed.get("interviewer_side", "NONE").upper()
        if interviewer_side not in ["LEFT", "RIGHT"]:
            interviewer_side = "NONE"

        print(f"[VISION] Interviewer is on: {interviewer_side} side of interviewee")

        return {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "method":           "openai_vision",
            "name_found":       parsed.get("interviewee_name_on_screen", ""),
            "confidence":       parsed.get("confidence", ""),
            "reasoning":        parsed.get("reasoning", ""),
            "interviewer_side": interviewer_side,
        }

    except Exception as e:
        print(f"[VISION] OpenAI Vision failed: {e}")
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

    if raw == interviewer_side:
        return "CENTER"

    return raw


def is_speaking(time_s: float, segments: list, window: float = 1.5) -> bool:
    """Returns True if a transcript segment overlaps the given time window."""
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


def is_typing_pattern(window_events):
    if len(window_events) < 8:
        return False
    gaps = [
        window_events[i]["time_s"] - window_events[i-1]["time_s"]
        for i in range(1, len(window_events))
    ]
    if not gaps:
        return False
    avg_gap = sum(gaps) / len(gaps)
    gap_std = np.std(gaps)
    return avg_gap < 3.5 and gap_std > 1.0




def detect_coding_phase(frame):
    try:
        # encode frame → base64
        _, buffer = cv2.imencode(".jpg", frame)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Is this screen showing a coding environment like VS Code, terminal, or code editor? Answer strictly with one word : YES or NO."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=5
        )

        answer = response.choices[0].message.content.strip().upper()

        return "YES" in answer

    except Exception as e:
        print(f"[VISION] Error detecting coding phase: {e}")
        return False

# ==============================================================================
# GPT-4o phone verification -- called only after YOLO detects phone 5x in a row
# Prevents false positives from YOLO misclassifying remotes, notebooks, hands etc.
# ==============================================================================
def _verify_phone_with_gpt4o(rgb_frame) -> bool:
    """Returns True only if GPT-4o confirms a phone is visible in the frame."""
    try:
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, 75])
        b64 = base64.b64encode(buf).decode("utf-8")

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=10,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}
                    },
                    {
                        "type": "text",
                        "text": (
                            "Look at this image carefully. Is there a mobile phone or smartphone "
                            "clearly visible being held or placed in front of a person? "
                            "Do NOT count laptops, tablets, remotes, books, notebooks, or hands without a phone. "
                            "Reply with only YES or NO."
                        )
                    }
                ]
            }]
        )

        answer = response.choices[0].message.content.strip().upper()
        print(f"[PHONE] GPT-4o verification answer: {answer}")
        return "YES" in answer

    except Exception as e:
        print(f"[PHONE] GPT-4o verification failed: {e} -- defaulting to YOLO result")
        return True  # if GPT call fails, trust YOLO to avoid silently missing real phones


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
        object_conf:            float = 0.75,
        cooldown_s:             float = 15.0,
        dir_min_occurrences:    int   = 6,
        dir_window_s:           float = 35.0,
    ):
        self.video_path    = video_path
        self.student_id    = student_id or Path(video_path).stem
        self.process_every = process_every_n_frames
        self.offscreen_dur = offscreen_duration_s
        self.obj_conf      = object_conf
        self.cooldown      = cooldown_s
        self.dir_min_occ   = dir_min_occurrences
        self.dir_window    = dir_window_s
        self._last_v: dict = {}
        self._yolo         = None

    def _load_yolo(self):
        if self._yolo is None:
            try:
                from ultralytics import YOLO
                self._yolo = YOLO("yolov8n.pt")
            except Exception as e:
                print(f"[YOLO] Unavailable: {e}")
                self._yolo = False
        return self._yolo

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
        has_transcript      = False  # <- track whether we have real speech timing data

        audio_path = extract_audio(self.video_path)
        if audio_path:
            transcript = transcribe_audio(audio_path)
            if transcript:
                transcript_full     = transcript["full_text"]
                transcript_segments = transcript.get("segments", [])
                has_transcript      = len(transcript_segments) > 0

                print(f"[PHASE 1] Asking GPT-4o to identify interviewee...")
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

        # -- PHASE 2: Find interviewee region ----------------------------------
        print(f"\n[PHASE 2] Locating interviewee on screen")
        region = find_interviewee_region_by_ocr(
            self.video_path, interviewee_name, frame_w, frame_h
        )
        result.interviewee_region = region
        x1, y1, x2, y2  = region["x1"], region["y1"], region["x2"], region["y2"]
        crop_w           = x2 - x1
        crop_h           = y2 - y1
        interviewer_side = region.get("interviewer_side", "NONE")
        print(f"[PHASE 2] Region: ({x1},{y1})->({x2},{y2}) size={crop_w}x{crop_h} method={region['method']}")
        print(f"[PHASE 2] Interviewer is on {interviewer_side} side -- glances that direction will be ignored")

        # -- Compute adaptive thresholds based on whether transcript exists --
        # Without audio we can't distinguish "silent = suspicious" vs "silent = no mic"
        # so we require a much longer, more obvious sustained gaze to flag anything.
        if has_transcript:
            # Normal operation: flag DOWN after 20s, LEFT/RIGHT after offscreen_dur
            down_flag_dur      = 20.0
            lr_flag_dur        = self.offscreen_dur
            down_flag_msg      = "(silent per transcript)"
        else:
            # No transcript: only flag truly egregious gaze (45s DOWN, 25s LEFT/RIGHT)
            down_flag_dur      = 45.0
            lr_flag_dur        = 25.0
            down_flag_msg      = "(no transcript -- conservative threshold)"

        print(f"[PHASE 3] Thresholds -- DOWN: {down_flag_dur}s | LEFT/RIGHT: {lr_flag_dur}s | transcript={'YES' if has_transcript else 'NO'}")
        
        # -- PHASE 3: Frame-by-frame detection ---------------------------------
        print(f"\n[PHASE 3] CV detection on interviewee region")
        
        
        coding_phase=False
        last_coding_check=0

        cap        = cv2.VideoCapture(self.video_path)
        violations = []
        gaze_log   = []

        bucket_size     = 10
        n_buckets       = max(1, int(duration_s / bucket_size) + 1)
        timeline        = [{"time_s": i*bucket_size, "violations": 0} for i in range(n_buckets)]

        offscreen_start = None
        offscreen_dir   = None
        down_start_time=None
        processed       = 0
        yolo            = self._load_yolo()
        phone_counter=0

        with mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        ) as face_mesh, \
        mp_face_detect.FaceDetection(
            model_selection=0, min_detection_confidence=0.5,
        ) as face_det:

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                current_time = frame_idx / fps
                
                if current_time > 600 :   # start after 10 mins, stop once found
                    if current_time - last_coding_check > (300 if coding_phase else 180):  # check every 200 sec till 600sec and then 300 sec after detection
                        coding_phase = detect_coding_phase(frame)
                        last_coding_check = current_time
                        print(f"[CONTEXT] Coding phase = {coding_phase} at t={current_time:.0f}s")

                if frame_idx % self.process_every != 0:
                    continue

                processed += 1

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                bucket = min(int(current_time / bucket_size), n_buckets - 1)

                def add_v(vtype, detail, severity="MEDIUM"):
                    if self._can_add(vtype, current_time):
                        violations.append(Violation(frame_idx, round(current_time,2), vtype, detail, severity))
                        timeline[bucket]["violations"] += 1

                fd = face_det.process(rgb)
                n_faces = len(fd.detections) if fd.detections else 0

                if n_faces == 0:
                    gaze_log.append({"time_s": round(current_time,2), "direction": "ABSENT"})
                    offscreen_start = None
                    offscreen_dir   = None
                    continue

                real_faces = sum(
                    1 for det in (fd.detections or [])
                    if det.location_data.relative_bounding_box.width *
                    det.location_data.relative_bounding_box.height > 0.15
                )

                if real_faces > 1:
                    add_v("third_person_detected",
                        f"{real_faces} people in interviewee frame -- possible coaching", "HIGH")

                mesh = face_mesh.process(rgb)
                if mesh.multi_face_landmarks:
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
                 

                    #  FILTER SHORT DOWN GLANCES
                    if direction == "DOWN":
                        if down_start_time is None:
                            down_start_time = current_time

                        duration = current_time - down_start_time

                        if duration < 0.8:
                            continue  # ignore tiny down glance
                    else:
                        down_start_time = None  # reset when not down

                    gaze_log.append({"time_s": round(current_time,2), "direction": direction})

                    # ================= FIXED LOGIC START =================
                    if direction in ["LEFT", "RIGHT", "DOWN"]:
                        if offscreen_start is None:
                            offscreen_start = current_time
                            offscreen_dir   = direction
                        else:
                            elapsed = current_time - offscreen_start
                            recent_events = gaze_log[-15:]
                            #  FILTER: ignore normal behavior (too much CENTER)
                            center_count = sum(1 for e in recent_events if e["direction"] == "CENTER")

                            if center_count > len(recent_events) * 0.3:
                                continue
                            
                             # Typing suppression (MAIN FIX)
                            if direction == "DOWN":
                                if is_typing_pattern(recent_events):
                                        print(f"[GAZE] Suppressed DOWN at t={current_time:.0f}s -- typing detected")
                                        offscreen_start = None
                                        offscreen_dir   = None
                                        continue


                          # dynamic threshold based on coding phase
                            if direction == "DOWN":
                                if coding_phase:
                                    threshold = 30.0   # relaxed (coding)
                                else:
                                    threshold = 20.0   # strict (non-coding)
                            else:
                                threshold = lr_flag_dur
                                

                            if elapsed >= threshold:
                                speaking = is_speaking(current_time, transcript_segments) if has_transcript else False

                               
                                if not speaking:
                                    sev = "LOW" if direction == "DOWN" and not has_transcript else "MEDIUM"
                                    add_v(
                                        "sustained_gaze",
                                        f"Looking {direction} continuously for {elapsed:.1f}s {down_flag_msg}",
                                        sev
                                    )
                                else:
                                    print(f"[GAZE] Suppressed sustained {direction} at t={current_time:.0f}s -- person is speaking")

                                # Reset after evaluation
                                offscreen_start = current_time
                                offscreen_dir   = direction
                    else:
                        offscreen_start = None
                        offscreen_dir   = None
                    # ================= FIXED LOGIC END =================

                # Object detection
                if yolo and frame_idx % 15 == 0:
                    try:
                        phone_in_frame = False  # defined ONCE per frame, outside result loop

                        for r in yolo(rgb, verbose=False, conf=self.obj_conf):
                            for box in r.boxes:
                                cls = yolo.model.names[int(box.cls)].lower()

                                # size filtering -- ignore tiny detections
                                bx1, by1, bx2, by2 = box.xyxy[0]
                                bw = bx2 - bx1
                                bh = by2 - by1
                                area = bw * bh

                                if area < 5000:
                                    continue

                                # center filtering -- ignore detections far to the edge
                                frame_center_x = crop_w / 2
                                box_center_x   = (bx1 + bx2) / 2
                                if abs(box_center_x - frame_center_x) > crop_w * 0.4:
                                    continue

                                # phone candidate
                                if "phone" in cls:
                                    phone_in_frame = True

                                # other prohibited objects
                                elif any(x in cls for x in ["book","laptop","tablet","earphone","airpod","headphone","notebook"]):
                                    add_v("prohibited_object", f"Detected: {cls}", "MEDIUM")

                        # multi-frame counter: phone must appear in 5 consecutive checks
                        if phone_in_frame:
                            phone_counter += 1
                            print(f"[PHONE] YOLO candidate frame {phone_counter}/5 at t={current_time:.0f}s")

                            if phone_counter >= 5:
                                # GPT-4o vision verification before flagging
                                # Prevents false positives from YOLO misclassifying
                                # remotes, wallets, notebooks, hands etc as phones
                                verified = _verify_phone_with_gpt4o(rgb)
                                if verified:
                                    add_v("phone_detected", "Phone confirmed by vision AI", "HIGH")
                                    phone_counter = 0  # reset after confirmed flag
                                else:
                                    print(f"[PHONE] GPT-4o rejected YOLO phone detection at t={current_time:.0f}s -- false positive suppressed")
                                    phone_counter = 0  # reset -- was a false positive
                        else:
                            phone_counter = 0

                    except:
                        pass
                
        cap.release()
        print(f"[PHASE 3] Processed {processed} frames, {len(gaze_log)} gaze readings, {len(violations)} raw violations")

       

        # -- PHASE 4: Pattern analysis ------------------------------------------
        print(f"\n[PHASE 4] Gaze direction pattern analysis")
        patterns = detect_direction_patterns(
            gaze_log,
            min_occurrences = self.dir_min_occ,
            window_s        = self.dir_window,
        )

        direction_counts = Counter(g["direction"] for g in gaze_log)
        total_tracked    = len(gaze_log) or 1
        gaze_pattern = {
            "LEFT_pct":             round(direction_counts.get("LEFT",  0) / total_tracked * 100, 1),
            "RIGHT_pct":            round(direction_counts.get("RIGHT", 0) / total_tracked * 100, 1),
            "DOWN_pct":             round(direction_counts.get("DOWN",  0) / total_tracked * 100, 1),
            "CENTER_pct":           round(direction_counts.get("CENTER",0) / total_tracked * 100, 1),
            "ABSENT_pct":           round(direction_counts.get("ABSENT",0) / total_tracked * 100, 1),
            "total_glances_logged": len(gaze_log),
            "repeated_patterns":    patterns,
            "has_transcript":       has_transcript,
        }

        print(f"[PHASE 4] LEFT={gaze_pattern['LEFT_pct']}% RIGHT={gaze_pattern['RIGHT_pct']}% DOWN={gaze_pattern['DOWN_pct']}% CENTER={gaze_pattern['CENTER_pct']}% ABSENT={gaze_pattern['ABSENT_pct']}%")
        print(f"[PHASE 4] Suspicious patterns: {len(patterns)}")
        for p in patterns:
            print(f"          -> {p['detail']}")

        for p in patterns:
            if self._can_add(f"pattern_{p['direction']}", p["time_s"]):
                violations.append(Violation(
                    0, p["time_s"],
                    "repeated_direction_pattern",
                    p["detail"],
                    "HIGH"
                ))

        # -- SCORING -----------------------------------------------------------
        counts = {
            "third_person_detected":      sum(1 for v in violations if v.type=="third_person_detected"),
            "phone_detected":             sum(1 for v in violations if v.type=="phone_detected"),
            "prohibited_object":          sum(1 for v in violations if v.type=="prohibited_object"),
            "sustained_gaze":             sum(1 for v in violations if v.type=="sustained_gaze"),
            "repeated_direction_pattern": sum(1 for v in violations if v.type=="repeated_direction_pattern"),
        }

        # sustained_gaze without a transcript is much weaker evidence -- weight it less
        sustained_weight = 12 if has_transcript else 6

        raw = (
            counts["third_person_detected"]      * 25 +
            counts["phone_detected"]             * 22 +
            min(counts["repeated_direction_pattern"] * 6, 40) +
            counts["prohibited_object"]          * 15 +
            counts["sustained_gaze"]             * sustained_weight
        )

        dominant_pct = max(gaze_pattern["LEFT_pct"], gaze_pattern["RIGHT_pct"], gaze_pattern["DOWN_pct"])
        if dominant_pct > 65:
            boost = 15
            raw  += boost
            print(f"[SCORE] Direction dominance boost: +{boost} ({dominant_pct:.0f}% off-center)")

        score = min(100, raw)
        risk  = "HIGH" if score >= 60 else "MEDIUM" if score >= 25 else "LOW"

        # -- Build summary ------------------------------------------------------
        parts = []
        if not has_transcript:
            parts.append("[WARN] No audio transcript (ffmpeg missing) -- gaze analysis used conservative thresholds")
        if counts["third_person_detected"] > 0:
            parts.append(f"3rd person {counts['third_person_detected']}x")
        if counts["phone_detected"] > 0:
            parts.append(f"Phone {counts['phone_detected']}x")
        if counts["repeated_direction_pattern"] > 0:
            dirs     = [p["direction"] for p in patterns]
            dominant = Counter(dirs).most_common(1)[0][0] if dirs else "unknown"
            parts.append(f"Repeated {dominant} look {counts['repeated_direction_pattern']}x")
        if counts["prohibited_object"] > 0:
            parts.append(f"Notes/objects {counts['prohibited_object']}x")
        if counts["sustained_gaze"] > 0:
            parts.append(f"Sustained silent gaze off-screen {counts['sustained_gaze']}x")
        if gaze_pattern["LEFT_pct"] > 25:
            parts.append(f"Eyes LEFT {gaze_pattern['LEFT_pct']}% of interview")
        if gaze_pattern["RIGHT_pct"] > 25:
            parts.append(f"Eyes RIGHT {gaze_pattern['RIGHT_pct']}% of interview")
        if gaze_pattern["DOWN_pct"] > 20:
            parts.append(f"Eyes DOWN {gaze_pattern['DOWN_pct']}% of interview")

        summary = " | ".join(parts) if parts else "No suspicious activity detected"

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

