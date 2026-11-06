"""Provider-neutral credentials and live checks for the native API judges."""

from __future__ import annotations

import base64
import hashlib
import io
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit


def add_api_arguments(parser):
    parser.add_argument(
        "--api-key-file",
        type=Path,
        help="Private key file; alternatively OPENAI_API_KEY or OPENAI_API_KEY_FILE.",
    )
    parser.add_argument(
        "--api-base-url",
        help="OpenAI-compatible base URL including /v1; alternatively OPENAI_BASE_URL.",
    )


def configure_api(key_file: Path | None = None, base_url: str | None = None) -> str:
    """Configure this scoring process only; never persist a credential."""
    base = (
        base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    ).rstrip("/")
    url = urlsplit(base)
    if (
        url.scheme not in {"https", "http"}
        or not url.netloc
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError(
            "API base URL must be an http(s) address without credentials/query/fragment"
        )
    path = key_file or os.environ.get("OPENAI_API_KEY_FILE")
    key = (
        Path(path).expanduser().read_text().strip()
        if path
        else os.environ.get("OPENAI_API_KEY", "").strip()
    )
    if not key or any(c.isspace() for c in key):
        raise ValueError(
            "Supply one non-empty API key via --api-key-file or OPENAI_API_KEY"
        )
    os.environ["OPENAI_API_KEY"] = key
    # Children inherit the resolved credential, not a potentially different file.
    os.environ.pop("OPENAI_API_KEY_FILE", None)
    os.environ["OPENAI_BASE_URL"] = base
    return base


def redact_secret(text: str) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    return text.replace(key, "[REDACTED]") if key else text


def _test_images(count: int) -> tuple[list[str], list[str]]:
    from PIL import Image

    colors = ["red", "blue", "green"]
    names = [colors[i % len(colors)] for i in range(count)]
    urls = []
    for name in names:
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), name).save(buffer, format="PNG")
        urls.append(
            "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        )
    return names, urls


# Leave ordinary requests untouched; six full-resolution PNGs can exceed a
# proxy's request-body limit even though each image is accepted individually.
_MSE_INLINE_COMPRESSION_THRESHOLD = 24 * 1024 * 1024
# Leave headroom for JSON/prompt overhead; this is not a discovered server limit.
_MSE_JPEG_FALLBACK_THRESHOLD = 30 * 1024 * 1024


def _mse_jpeg_fallback(images, audit_dir: Path | None):
    """Last-resort, explicitly audited lossy transport; never resize/save images."""
    before = sum(len(image["url"]) for image in images)
    if before <= _MSE_JPEG_FALLBACK_THRESHOLD:
        return

    from PIL import Image

    changes = []
    for index, image in enumerate(images):
        header, encoded = image["url"].split(",", 1)
        if not header.endswith(";base64"):
            continue
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as decoded:
            # Do not flatten alpha, change bit depth, or repeatedly quantize JPEG.
            if decoded.format not in {"PNG", "WEBP"} or decoded.mode != "RGB":
                continue
            buffer = io.BytesIO()
            decoded.save(
                buffer, format="JPEG", quality=98, subsampling=0, optimize=True
            )
            compact = buffer.getvalue()
            if len(compact) >= len(raw):
                continue
            with Image.open(io.BytesIO(compact)) as restored:
                if restored.size != decoded.size:
                    raise ValueError("MSE JPEG transport changed image dimensions")
                transport_pixels = hashlib.sha256(restored.tobytes()).hexdigest()
            image["url"] = (
                "data:image/jpeg;base64," + base64.b64encode(compact).decode()
            )
            changes.append(
                {
                    "image_index": index,
                    "size": list(decoded.size),
                    "mode": decoded.mode,
                    "pixel_sha256": hashlib.sha256(decoded.tobytes()).hexdigest(),
                    "transport_pixel_sha256": transport_pixels,
                    "original_sha256": hashlib.sha256(raw).hexdigest(),
                    "transport_sha256": hashlib.sha256(compact).hexdigest(),
                    "original_bytes": len(raw),
                    "transport_bytes": len(compact),
                }
            )
    if not changes:
        return
    after = sum(len(image["url"]) for image in images)
    print(
        f"MSE lossy JPEG fallback (quality=98, 4:4:4, no resize): "
        f"{before} -> {after} inline bytes",
        flush=True,
    )
    if audit_dir is not None:
        from .common import digest, write_json

        report = {
            "version": "mse-jpeg-fallback-v1",
            "lossy": True,
            "quality": 98,
            "subsampling": 0,
            "optimize": True,
            "resized": False,
            "inline_bytes_before": before,
            "inline_bytes_after": after,
            "images": changes,
        }
        write_json(audit_dir / f"jpeg-{digest(report)}.json", report)


def _compact_mse_images(messages, audit_dir: Path | None):
    images = [
        item["image_url"]
        for message in messages
        for item in message["content"]
        if item["type"] == "image_url"
        and item["image_url"]["url"].startswith("data:image/")
    ]
    before = sum(len(image["url"]) for image in images)
    if before <= _MSE_INLINE_COMPRESSION_THRESHOLD:
        return

    from PIL import Image

    changes = []
    for index, image in enumerate(images):
        header, encoded = image["url"].split(",", 1)
        if not header.endswith(";base64"):
            continue
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as decoded:
            # The upstream labels PNG bytes as image/jpeg. Inspect the bytes,
            # not that label; never re-encode lossy JPEGs or high-bit-depth data.
            if decoded.format != "PNG" or decoded.mode not in {"RGB", "RGBA"}:
                continue
            buffer = io.BytesIO()
            decoded.save(buffer, format="WEBP", lossless=True, exact=True, method=4)
            compact = buffer.getvalue()
            if len(compact) >= len(raw):
                continue
            pixels = decoded.tobytes()
            with Image.open(io.BytesIO(compact)) as restored:
                if (
                    restored.size != decoded.size
                    or restored.convert(decoded.mode).tobytes() != pixels
                ):
                    raise ValueError("MSE lossless transport changed image pixels")
            image["url"] = (
                "data:image/webp;base64," + base64.b64encode(compact).decode()
            )
            changes.append(
                {
                    "image_index": index,
                    "size": list(decoded.size),
                    "mode": decoded.mode,
                    "pixel_sha256": hashlib.sha256(pixels).hexdigest(),
                    "original_sha256": hashlib.sha256(raw).hexdigest(),
                    "transport_sha256": hashlib.sha256(compact).hexdigest(),
                    "original_bytes": len(raw),
                    "transport_bytes": len(compact),
                }
            )
    after = sum(len(image["url"]) for image in images)
    print(f"MSE lossless transport: inline image bytes {before} -> {after}", flush=True)
    if audit_dir is not None:
        from .common import digest, write_json

        report = {
            "version": "mse-lossless-webp-v1",
            "inline_bytes_before": before,
            "inline_bytes_after": after,
            "images": changes,
        }
        write_json(audit_dir / f"{digest(report)}.json", report)
    _mse_jpeg_fallback(images, audit_dir)


def mse_responses_via_chat(client, *, audit_dir: Path | None = None):
    """Translate only MSE's text+image Responses request; keep its native retry loop.

    This is not a general Responses API emulator. Prompts, image sizes, model
    and sampling options stay native. Try pixel-verified lossless WebP first;
    still-oversized RGB payloads use audited JPEG quality=98 / 4:4:4. The latter
    is lossy, not pixel-identical official scoring. Cached images are unchanged.
    """

    def create(*, model, input):
        messages = []
        for message in input:
            content = []
            for item in message["content"]:
                if item["type"] == "input_text":
                    content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "input_image":
                    content.append(
                        {"type": "image_url", "image_url": {"url": item["image_url"]}}
                    )
                else:
                    raise ValueError(f"Unsupported MSE input type: {item['type']}")
            messages.append({"role": message["role"], "content": content})
        _compact_mse_images(messages, audit_dir)
        response = client.chat.completions.create(model=model, messages=messages)
        text = response.choices[0].message.content
        if not text:
            raise ValueError("MSE judge returned no text")
        return SimpleNamespace(output_text=text)

    return SimpleNamespace(responses=SimpleNamespace(create=create))


def probe_judges(client, judges: dict[str, str], *, apis: dict | None = None) -> dict:
    """Paid, small multiview requests; keep exact judge names and API families."""
    report = {
        "api_base": str(client.base_url).rstrip("/"),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {},
    }
    try:
        listed = {m.id for m in client.models.list()}
        report["model_list"] = {
            "status": "passed",
            "required": {m: m in listed for m in judges.values()},
        }
    except Exception as error:
        listed = None
        report["model_list"] = {"status": "failed", "error": redact_secret(str(error))}
    for bench, model in judges.items():
        api = (apis or {}).get(bench, "responses" if bench == "mse" else "chat")
        endpoint = "responses" if api == "responses" else "chat/completions"
        colors, images = _test_images(
            6 if bench == "mse" else 3 if bench == "rise" else 2
        )
        check = {
            "requested_model": model,
            "endpoint": endpoint,
            "images": len(images),
            "listed": model in listed if listed is not None else None,
        }
        prompt = (
            "Name the dominant color of each image in order. "
            "Reply only with comma-separated English color names."
        )
        print(
            f"Checking {bench}: {model} /{endpoint} ({len(images)} images)", flush=True
        )
        try:
            if endpoint == "responses":
                response = client.responses.create(
                    model=model,
                    input=[
                        {
                            "role": "user",
                            "type": "message",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                *[
                                    {"type": "input_image", "image_url": image}
                                    for image in images
                                ],
                            ],
                        }
                    ],
                )
                answer = response.output_text
            else:
                options = (
                    {"max_tokens": 1400}
                    if bench == "gedit"
                    else {"max_tokens": 4096, "temperature": 0, "n": 1}
                    if bench == "rise"
                    else {"stream": False}
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                *[
                                    {"type": "image_url", "image_url": {"url": image}}
                                    for image in images
                                ],
                            ],
                        }
                    ],
                    **options,
                )
                answer = response.choices[0].message.content
            normalized = [
                word.strip(" .\n").lower() for word in (answer or "").split(",")
            ]
            check.update(
                {
                    "status": "passed"
                    if normalized == colors
                    else "unexpected_response",
                    "returned_model": response.model,
                    "response_id": response.id,
                    "answer": redact_secret(answer or ""),
                    "usage": response.usage.model_dump() if response.usage else None,
                }
            )
        except Exception as error:
            check.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "http_status": getattr(error, "status_code", None),
                    "error": redact_secret(str(error)),
                }
            )
        report["checks"][bench] = check
        print(f"{bench}: {check['status']}", flush=True)
    report["ready"] = all(c["status"] == "passed" for c in report["checks"].values())
    return report
