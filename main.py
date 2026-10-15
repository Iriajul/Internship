"""Generate a video with Higgsfield's Seedance 2.5 text-to-video model.

Credentials are read at runtime from HF_KEY (format key-id:key-secret),
loaded from .env.local. Never print or log the key.
"""

import sys

import httpx
from dotenv import load_dotenv

import higgsfield_client

MODEL = "bytedance/seedance-2.5/text-to-video"
ARGUMENTS = {
    "prompt": "A cinematic scene at sunset",
    "duration": 5,
    "resolution": "720p",
    "aspect_ratio": "16:9",
}

TERMINAL_FAILURES = {
    higgsfield_client.Failed: "failed",
    higgsfield_client.NSFW: "was moderated (flagged NSFW)",
    higgsfield_client.Cancelled: "was canceled",
}


def find_video_url(result: dict) -> str | None:
    video = result.get("video")
    if isinstance(video, dict) and video.get("url"):
        return video["url"]
    videos = result.get("videos")
    if isinstance(videos, list) and videos and isinstance(videos[0], dict):
        return videos[0].get("url")
    return None


def main() -> int:
    load_dotenv(".env.local")

    last_status = None

    def on_enqueue(request_id: str) -> None:
        print(f"Request enqueued: {request_id}")

    def on_queue_update(status) -> None:
        nonlocal last_status
        last_status = status
        print(f"Status: {type(status).__name__}")

    try:
        result = higgsfield_client.subscribe(
            MODEL,
            arguments=ARGUMENTS,
            on_enqueue=on_enqueue,
            on_queue_update=on_queue_update,
        )
    except higgsfield_client.CredentialsMissedError:
        print("HF_KEY is not set. Add it to .env.local (key-id:key-secret).", file=sys.stderr)
        return 1
    except higgsfield_client.HiggsfieldClientError as exc:
        print(f"Higgsfield API error: {exc}", file=sys.stderr)
        return 1
    except httpx.HTTPError as exc:
        print(f"Could not reach the Higgsfield API: {exc}", file=sys.stderr)
        return 1

    # subscribe() returns the final payload even when the request did not succeed,
    # so check the terminal status explicitly.
    for status_cls, reason in TERMINAL_FAILURES.items():
        if isinstance(last_status, status_cls):
            print(f"Generation {reason}. Response: {result}", file=sys.stderr)
            return 1

    if not isinstance(last_status, higgsfield_client.Completed) or result.get("status") not in (None, "completed"):
        print(f"Generation did not complete. Response: {result}", file=sys.stderr)
        return 1

    url = find_video_url(result)
    if not url:
        print(f"Completed, but no video URL found in response: {result}", file=sys.stderr)
        return 1

    print(f"Video URL: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
