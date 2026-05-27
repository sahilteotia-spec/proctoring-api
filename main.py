"""

"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from detector import TranscriptExtractor


OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./results"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def check_env():
    missing = []
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.environ.get("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY")
    if missing:
        for key in missing:
            print(f"[ERROR] {key} is not set.  →  export {key}=your_key_here")
        sys.exit(1)


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    else:
        try:
            return obj.item()          # handles numpy scalar types
        except Exception:
            return str(obj)


def run(video_path: str, student_id: str = None):
    video = Path(video_path)
    if not video.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    student_id = student_id or video.stem

    print("=" * 60)
    print(f"  Transcript Extractor")
    print(f"  Video   : {video}")
    print(f"  Student : {student_id}")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    extractor = TranscriptExtractor(
        video_path=str(video),
        student_id=student_id,
    )

    t0      = time.time()
    result  = extractor.extract()
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("  RESULT SUMMARY")
    print("=" * 60)
    print(f"  Interviewee  : {result.interviewee_name}")
    print(f"  Interviewer  : {result.interviewer_name}")
    print(f"  Topic        : {result.interview_topic}")
    print(f"  Language     : {result.language}")
    print(f"  Duration     : {result.duration_s:.1f}s")
    print(f"  Transcript   : {len(result.transcript_full)} chars")
    print(f"  Summary      : {result.transcript_summary}")
    print(f"  Processed in : {elapsed:.1f}s")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file  = OUTPUT_DIR / f"{student_id}_{timestamp}.json"

    data = result.to_dict()
    data["processing_time_s"] = round(elapsed, 2)
    data = make_json_safe(data)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n  JSON saved → {out_file}")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python runtest.py <video_path> [student_id]")
        sys.exit(1)

    check_env()
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
