import sys
import traceback

def excepthook(type, value, tb):
    print("".join(traceback.format_exception(type, value, tb)))

sys.excepthook = excepthook

import os
import json
import tempfile
import subprocess
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
# Main transcript extractor class
# ==============================================================================
class TranscriptExtractor:
    def __init__(self, video_path: str, student_id: str = ""):
        self.video_path = video_path
        self.student_id = student_id or Path(video_path).stem

    def extract(self) -> TranscriptResult:
        duration_s = get_audio_duration(self.video_path)

        print(f"\n{'='*60}")
        print(f"[EXTRACT] {self.student_id}")
        print(f"[EXTRACT] Duration: {duration_s:.0f}s")
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
        print(f"\n[EXTRACT] Identifying participants via GPT-4o-mini...")
        info = identify_interviewee(transcript)

        result.interviewee_name   = info.get("interviewee_name") or "unknown"
        result.interviewer_name   = info.get("interviewer_name") or "unknown"
        result.interview_topic    = info.get("interview_topic",  "")
        result.transcript_summary = info.get("summary",          "")

        return result
