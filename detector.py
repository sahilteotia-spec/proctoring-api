import sys
import traceback

def excepthook(type, value, tb):
    print("".join(traceback.format_exception(type, value, tb)))

sys.excepthook = excepthook

from dotenv import load_dotenv
load_dotenv()

import os
import json
import tempfile
import subprocess
import base64
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import openai
from groq import Groq


# -- API clients ---------------------------------------------------------------
openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
groq_client   = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ==============================================================================
# Result dataclass
# ==============================================================================
@dataclass
class TranscriptResult:
    student_id:         str
    video_path:         str
    duration_s:         float
    transcript_full:    str  = ""
    interviewee_name:   str  = "unknown"
    interviewer_name:   str  = "unknown"
    interview_topic:    str  = ""
    transcript_summary: str  = ""
    language:           str  = ""
    video_analysis:     Optional[dict] = None

    def to_dict(self):
        return {
            "student_id":           self.student_id,
            "video_path":           self.video_path,
            "duration_s":           round(self.duration_s, 2),
            "interviewee_name":     self.interviewee_name,
            "interviewer_name":     self.interviewer_name,
            "interview_topic":      self.interview_topic,
            "transcript_summary":   self.transcript_summary,
            "language":             self.language,
            "transcript_full":      self.transcript_full,
            "video_analysis":       self.video_analysis,
        }


# ==============================================================================
# STEP 1 — Extract audio from video
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
            print(f"[AUDIO] Extracted: {Path(audio_path).stat().st_size / 1024 / 1024:.1f} MB")
            return audio_path

        print(f"[AUDIO] ffmpeg failed: {result.stderr.decode()}")
        return None

    except FileNotFoundError:
        print("[AUDIO] ffmpeg not found — install ffmpeg and add it to PATH.")
        return None
    except Exception as e:
        print(f"[AUDIO] Exception: {e}")
        return None


# ==============================================================================
# STEP 2 — Transcribe with Groq Whisper (chunked for long audio)
# ==============================================================================
def get_audio_duration(audio_path: str) -> float:
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path
        ], capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


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
            if Path(chunk_path).stat().st_size / 1024 / 1024 > 0.1:
                chunks.append((chunk_path, offset))

        offset    += chunk_s
        chunk_idx += 1

    print(f"[WHISPER] Split into {len(chunks)} chunks")
    return chunks


def transcribe_audio(audio_path: str) -> Optional[dict]:
    try:
        size_mb    = Path(audio_path).stat().st_size / 1024 / 1024
        duration_s = get_audio_duration(audio_path)
        print(f"[WHISPER] Audio: {size_mb:.1f} MB, {duration_s:.0f}s")

        CHUNK_S = 10 * 60  # 10-minute chunks

        if duration_s <= CHUNK_S and size_mb <= 24:
            chunks = [(audio_path, 0)]
        else:
            print(f"[WHISPER] Splitting into 10-min chunks...")
            chunks = split_audio(audio_path, CHUNK_S)

        all_segments    = []
        full_text_parts = []
        last_response   = None

        for chunk_path, offset_s in chunks:
            chunk_mb = Path(chunk_path).stat().st_size / 1024 / 1024
            print(f"[WHISPER] Transcribing chunk offset={offset_s:.0f}s  size={chunk_mb:.1f} MB")

            with open(chunk_path, "rb") as f:
                response = groq_client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )

            last_response = response
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
                try:
                    Path(chunk_path).unlink()
                except Exception:
                    pass

        full_text = " ".join(full_text_parts)
        language  = getattr(last_response, "language", "") if last_response else ""
        print(f"[WHISPER] Done: {len(all_segments)} segments, {len(full_text)} chars, lang={language}")

        return {
            "full_text": full_text,
            "segments":  all_segments,
            "language":  language,
        }

    except Exception as e:
        print(f"[WHISPER] Failed: {e}")
        return None


# ==============================================================================
# STEP 3 — GPT-4o-mini: identify interviewee name + summary
# ==============================================================================
def identify_interviewee(transcript: dict) -> dict:
    fallback = {
        "interviewee_name": None,
        "interviewer_name": None,
        "interview_topic":  "unknown",
        "summary":          "",
        "confidence":       "LOW",
    }

    try:
        full_text = transcript.get("full_text", "")
        if not full_text or len(full_text) < 30:
            print("[GPT] Transcript too short to identify participants.")
            return fallback

        prompt = f"""You are analyzing a job interview transcript.

TRANSCRIPT:
{full_text[:5000]}

1. Who is the INTERVIEWEE? (answers questions about background/skills/experience)
2. Who is the INTERVIEWER? (asks questions)
3. Extract the interviewee's name if mentioned (e.g. "Hi I'm John" or "Can you introduce yourself, Priya?")
4. What is the interview about? (job role / skill being evaluated)
5. Write a 3-5 sentence summary of what was discussed.

Respond ONLY in this exact JSON format, no other text:
{{
  "interviewee_name": "first name or full name if found, else null",
  "interviewer_name": "first name or full name if found, else null",
  "interview_topic":  "e.g. Oracle EBS Functional Consultant",
  "summary":          "3-5 sentence summary of the interview",
  "confidence":       "HIGH or MEDIUM or LOW",
  "name_mention":     "quote from transcript where name was mentioned, or null"
}}"""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )

        text   = response.choices[0].message.content.strip()
        text   = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)

        print(
            f"[GPT] Interviewee: {parsed.get('interviewee_name')} | "
            f"Topic: {parsed.get('interview_topic')} | "
            f"Confidence: {parsed.get('confidence')}"
        )
        return parsed

    except Exception as e:
        print(f"[GPT] identify_interviewee failed: {e}")
        return fallback

# ==============================================================================
# STEP 4 — Video analysis (camera framing + lighting)
# ==============================================================================
def get_frame_timestamps(duration_s: float, interval_s: int = 240) -> list:
    if duration_s <= 0:
        return []
    if duration_s <= interval_s:
        return [round(duration_s / 2, 2)]
    
    timestamps = []
    current = interval_s
    while current <= duration_s:
        timestamps.append(float(current))
        current += interval_s
    
    if not timestamps:
        timestamps.append(round(duration_s / 2, 2))
    return timestamps


def extract_frame_at_timestamp(video_path: str, timestamp_s: float) -> Optional[str]:
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[VIDEO] Failed to open video file: {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_number = int(timestamp_s * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            fd, frame_path = tempfile.mkstemp(suffix=f"_frame_{int(timestamp_s)}.jpg")
            os.close(fd)
            cv2.imwrite(frame_path, frame)
            return frame_path
        else:
            print(f"[VIDEO] Failed to read frame at timestamp {timestamp_s}")
            return None
    except Exception as e:
        print(f"[VIDEO] Exception extracting frame at {timestamp_s}: {e}")
        return None


def encode_image_base64(image_path: str) -> Optional[str]:
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"[VIDEO] Base64 encoding failed for {image_path}: {e}")
        return None


def extract_json_from_text(text: str) -> Optional[dict]:
    # Strip any thinking blocks first to avoid matching draft JSON blocks inside thoughts
    if "</think>" in text:
        parts = text.split("</think>", 1)
        text = parts[1]
        
    text = text.strip()
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        return None
    
    json_str = text[start_idx:end_idx+1]
    try:
        return json.loads(json_str)
    except Exception as e:
        print(f"[JSON_PARSE] Error decoding JSON substring: {e}")
        return None


def analyze_frame_with_groq(frame_path: str, timestamp_formatted: str, interviewee_name: str, student_id: str) -> Optional[dict]:
    base64_image = encode_image_base64(frame_path)
    if not base64_image:
        return None
    
    candidate_display_name = interviewee_name if interviewee_name and interviewee_name.lower() != "unknown" else "the candidate"
    
    prompt = (
        f"You are an expert video quality reviewer. Analyze the candidate's webcam setup and lighting in this video frame.\n"
        f"Note: The video frame might show a video call interface, grid layout, split screens, or cropped panels. "
        f"You must ignore the overall video call layout and focus EXCLUSIVELY on the candidate '{candidate_display_name}''s own webcam feed tile. "
        f"To identify the correct feed tile on the screen, look for the video tile labeled with the candidate's name '{candidate_display_name}' (usually displayed at the bottom-left or bottom of their video feed tile).\n"
        f"\n"
        f"Please evaluate two aspects separately:\n"
        f"1. Camera positioning/framing: Focus ONLY on '{candidate_display_name}''s own video feed. Is their face fully visible? "
        f"Is their camera positioned upfront at a normal angle and height? Are they sitting normally and centered within their own camera feed? "
        f"Ignore if the video call interface crops them or if they are in a split layout—evaluate only their physical seating/camera angle.\n"
        f"2. Lighting quality: Focus ONLY on '{candidate_display_name}''s face. Is their face well-lit? Are there harsh shadows or severe backlighting?\n"
        f"\n"
        f"You must respond ONLY with a raw JSON object matching this schema, without any markdown formatting, backticks, or code blocks:\n"
        f"{{\n"
        f"  \"camera_score\": 7,\n"
        f"  \"camera_feedback\": \"description of {candidate_display_name}'s camera positioning\",\n"
        f"  \"lighting_score\": 6,\n"
        f"  \"lighting_feedback\": \"description of {candidate_display_name}'s face lighting\"\n"
        f"}}"
    )
    
    content = ""
    try:
        response = groq_client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt + "\nNote: Keep your internal reasoning/thoughts extremely brief (under 50 words) to prevent running out of tokens, and output the JSON immediately."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        content = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[VIDEO] Groq vision API call failed for frame at {timestamp_formatted}: {e}")
        return None

    parsed = extract_json_from_text(content)
    if not parsed:
        print(f"[VIDEO] JSON parsing failed for frame at {timestamp_formatted}. Response content was: {content}")
        return None

    return {
        "camera_score": int(parsed.get("camera_score", 5)),
        "camera_feedback": str(parsed.get("camera_feedback", "N/A")),
        "lighting_score": int(parsed.get("lighting_score", 5)),
        "lighting_feedback": str(parsed.get("lighting_feedback", "N/A")),
    }


def generate_overall_summary(frame_analyses: list) -> dict:
    if not frame_analyses:
        return {
            "overall_camera_score": 0.0,
            "overall_lighting_score": 0.0,
            "overall_summary": "No frames analyzed.",
            "frame_analyses": []
        }
    
    avg_camera = sum(f["camera_score"] for f in frame_analyses) / len(frame_analyses)
    avg_lighting = sum(f["lighting_score"] for f in frame_analyses) / len(frame_analyses)
    
    summaries = []
    for f in frame_analyses:
        summaries.append(
            f"Frame at {f['timestamp_formatted']}:\n"
            f"- Camera Score: {f['camera_score']}/10. Feedback: {f['camera_feedback']}\n"
            f"- Lighting Score: {f['lighting_score']}/10. Feedback: {f['lighting_feedback']}"
        )
    
    analysis_text = "\n\n".join(summaries)
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional video quality analyzer. Summarize the overall camera positioning and lighting based on the segment reports."
                },
                {
                    "role": "user",
                    "content": (
                        f"Here are the reports of frames sampled at regular intervals from a video:\n\n"
                        f"{analysis_text}\n\n"
                        f"Based on these segment reports, write a brief, professional overall summary (3-4 sentences) "
                        f"evaluating the lighting and camera positioning of the entire video."
                    )
                }
            ],
            max_tokens=250,
            temperature=0.3,
        )
        overall_summary = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[VIDEO] Failed to generate overall summary via Groq: {e}")
        overall_summary = f"Average camera positioning is scored {avg_camera:.1f}/10 and average lighting is scored {avg_lighting:.1f}/10."
        
    return {
        "overall_camera_score": round(avg_camera, 2),
        "overall_lighting_score": round(avg_lighting, 2),
        "overall_summary": overall_summary,
        "frame_analyses": frame_analyses
    }


def format_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def analyze_video(video_path: str, duration_s: float, interviewee_name: str, student_id: str) -> dict:
    print(f"\n[VIDEO] Starting video analysis for: {video_path}")
    timestamps = get_frame_timestamps(duration_s, interval_s=240)
    print(f"[VIDEO] Generated {len(timestamps)} timestamps to analyze: {timestamps}")
    
    frame_analyses = []
    for ts in timestamps:
        ts_formatted = format_timestamp(ts)
        print(f"[VIDEO] Extracting and analyzing frame at {ts_formatted} ({ts}s)...")
        
        frame_path = extract_frame_at_timestamp(video_path, ts)
        if not frame_path:
            continue
            
        try:
            analysis = analyze_frame_with_groq(frame_path, ts_formatted, interviewee_name, student_id)
            if analysis:
                analysis["timestamp_s"] = ts
                analysis["timestamp_formatted"] = ts_formatted
                frame_analyses.append(analysis)
                print(f"[VIDEO] Frame at {ts_formatted} analyzed: Camera={analysis['camera_score']}/10, Lighting={analysis['lighting_score']}/10")
            else:
                print(f"[VIDEO] Frame analysis returned None for {ts_formatted}")
        finally:
            try:
                os.unlink(frame_path)
            except Exception:
                pass
                
    result = generate_overall_summary(frame_analyses)
    print(f"[VIDEO] Analysis complete. Overall Camera Score: {result['overall_camera_score']}, Overall Lighting Score: {result['overall_lighting_score']}")
    return result


# ==============================================================================
# Main transcript extractor class
# ==============================================================================
class TranscriptExtractor:
    def __init__(self, video_path: str, student_id: str = "", interviewee_name: str = ""):
        self.video_path = video_path
        self.student_id = student_id or Path(video_path).stem
        self.interviewee_name = interviewee_name

    def extract(self) -> TranscriptResult:
        import cv2
        cap        = cv2.VideoCapture(self.video_path)
        fps        = cap.get(cv2.CAP_PROP_FPS) or 25
        total_f    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_f / fps
        cap.release()

        print(f"\n{'='*60}")
        print(f"[EXTRACT] {self.student_id}")
        print(f"[EXTRACT] Duration: {duration_s:.0f}s  |  Frames: {total_f}")
        print(f"{'='*60}")

        result = TranscriptResult(
            student_id=self.student_id,
            video_path=self.video_path,
            duration_s=duration_s,
        )

        # Step 1 — audio
        audio_path = extract_audio(self.video_path)
        if not audio_path:
            print("[EXTRACT] Could not extract audio. Aborting.")
            return result

        # Step 2 — transcribe
        transcript = transcribe_audio(audio_path)
        try:
            Path(audio_path).unlink()
        except Exception:
            pass

        if not transcript:
            print("[EXTRACT] Transcription failed.")
            return result

        result.transcript_full     = transcript["full_text"]
        result.language            = transcript.get("language", "")

        # Step 3 — identify participants
        if self.interviewee_name:
            print(f"\n[EXTRACT] Using provided interviewee name: {self.interviewee_name}")
            result.interviewee_name = self.interviewee_name
            info = identify_interviewee(transcript)
            result.interviewer_name   = info.get("interviewer_name") or "unknown"
            result.interview_topic    = info.get("interview_topic",  "")
            result.transcript_summary = info.get("summary",          "")
        else:
            print(f"\n[EXTRACT] Identifying participants via GPT-4o-mini...")
            info = identify_interviewee(transcript)
            result.interviewee_name   = info.get("interviewee_name") or "unknown"
            result.interviewer_name   = info.get("interviewer_name") or "unknown"
            result.interview_topic    = info.get("interview_topic",  "")
            result.transcript_summary = info.get("summary",          "")

        # Step 5 — Video Analysis (camera framing + lighting)
        print(f"\n[EXTRACT] Analyzing video camera and lighting via Groq Vision...")
        video_analysis_res = analyze_video(
            video_path=self.video_path,
            duration_s=duration_s,
            interviewee_name=result.interviewee_name,
            student_id=self.student_id
        )
        result.video_analysis = video_analysis_res

        return result