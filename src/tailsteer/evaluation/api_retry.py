"""Bounded transport retries and cache-aware recovery, independent of scoring rules."""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from uuid import uuid4

from .api import redact_secret
from .common import digest, read_json, write_json

REQUEST_ATTEMPTS = 5
SCORING_PASSES = 3
REQUEST_TIMEOUT = 180


class JudgeAPIError(RuntimeError):
    def __init__(
        self, message, *, retryable=True, status=None, headers=None, body=None
    ):
        super().__init__(redact_secret(message))
        self.retryable = retryable
        self.status = status
        self.headers = headers or {}
        self.body = body


class InvalidJudgement(ValueError):
    """A response was received but cannot be scored, rather than a low score."""


class IncompleteJudgements(RuntimeError):
    def __init__(self, message, *, retryable=True):
        super().__init__(message)
        self.retryable = retryable


def _retryable_status(status):
    return status in {408, 409, 425, 429} or 500 <= status < 600


def _api_error(error):
    if isinstance(error, JudgeAPIError):
        return error
    response = getattr(error, "response", None)
    status = getattr(error, "status_code", None)
    if status is not None:
        return JudgeAPIError(
            str(error),
            retryable=_retryable_status(status),
            status=status,
            headers=getattr(response, "headers", {}),
            body=getattr(response, "text", None),
        )
    # Import only in the isolated API environment, not during training/planning.
    import httpx
    import requests

    if isinstance(error, requests.exceptions.SSLError):
        # A peer closing TLS mid-response is transient, unlike certificate errors.
        # Keep verification enabled and preserve the same bounded retry budget.
        return JudgeAPIError(
            str(error), retryable="UNEXPECTED_EOF_WHILE_READING" in str(error)
        )
    if isinstance(
        error,
        (
            TimeoutError,
            ConnectionError,
            httpx.TransportError,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ),
    ):
        return JudgeAPIError(str(error))
    if type(error).__name__ in {"APIConnectionError", "APITimeoutError"}:
        return JudgeAPIError(str(error))
    return None  # Do not retry programming, asset or configuration errors.


def retry_delay(attempt, headers=None):
    """Respect Retry-After (seconds/date), bounded to one minute per pause."""
    value = (headers or {}).get("retry-after", (headers or {}).get("Retry-After"))
    try:
        delay = float(value)
    except (TypeError, ValueError):
        try:
            delay = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, AttributeError, OverflowError):
            delay = 0
    return min(60.0, max(delay, min(30.0, 2**attempt) + random.uniform(0, 1)))


class JudgeAPI:
    """Bound transport retries without multiplying exhausted upstream requests.

    No successful score is retried here. Exhausted requests are remembered for
    this pass; a later cache-aware scoring pass can retry failed samples only.
    """

    def __init__(self, work: Path, bundle: dict):
        self.events = Path(
            bundle.get("_retry_events", work / "retry-events" / uuid4().hex)
        )
        self.failed = {}
        self.locks = defaultdict(Lock)
        self.auth_error = None

    def call(self, send, request):
        key = digest(request)  # Never log prompts, base64 images or credentials.
        with self.locks[key]:
            return self._call(send, key)

    def _call(self, send, key):
        if self.auth_error is not None:
            raise self.auth_error
        if key in self.failed:
            raise self.failed[key]
        for attempt in range(1, REQUEST_ATTEMPTS + 1):
            if self.auth_error is not None:
                raise self.auth_error
            try:
                return send()
            except Exception as original:
                error = _api_error(original)
                if error is None:
                    raise
                delay = retry_delay(attempt, error.headers)
                again = error.retryable and attempt < REQUEST_ATTEMPTS
                write_json(
                    self.events / f"{key}-{time.time_ns()}-{uuid4().hex}.json",
                    {
                        "request": key,
                        "attempt": attempt,
                        "status": error.status,
                        "retryable": error.retryable,
                        "retry_in_seconds": delay if again else None,
                        "error": str(error),
                        "response": redact_secret(str(error.body))
                        if error.body is not None
                        else None,
                    },
                )
                print(
                    f"[judge-api] {key[:10]} attempt {attempt}/{REQUEST_ATTEMPTS}: "
                    f"{type(original).__name__}, HTTP {error.status}; "
                    + (f"retry in {delay:.1f}s" if again else "retained as failure"),
                    flush=True,
                )
                if not again:
                    self.failed[key] = error
                    if error.status in {401, 403}:
                        self.auth_error = error
                    if error is original:
                        raise
                    raise error from original
                time.sleep(delay)

    def post(self, url, *, return_errors=False, **kwargs):
        import requests

        kwargs.setdefault("timeout", REQUEST_TIMEOUT)

        def send():
            response = requests.post(url, **kwargs)
            status = response.status_code
            if not 200 <= status < 300:
                raise JudgeAPIError(
                    f"Judge HTTP {status}",
                    retryable=_retryable_status(status),
                    status=status,
                    headers=response.headers,
                    body=response.text,
                )
            try:
                content = response.json()["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Empty judge response")
            except (ValueError, KeyError, IndexError, TypeError) as error:
                raise JudgeAPIError(
                    "Malformed/empty chat response", body=response.text
                ) from error
            return response

        try:
            return self.call(
                send, {"url": url, "payload": kwargs.get("json", kwargs.get("data"))}
            )
        except JudgeAPIError as error:
            if not return_errors:
                raise
            # RISE's native parser records the failed sample and continues. Do
            # not throw into its thread pool and lose other finished samples.
            body = {"error": {"code": "judge_request_failed", "message": str(error)}}
            return SimpleNamespace(
                status_code=error.status or 599,
                text=json.dumps(body),
                json=lambda: body,
            )

    def client(self, factory, **kwargs):
        client = factory(**{**kwargs, "max_retries": 0, "timeout": REQUEST_TIMEOUT})

        def create(kind, **payload):
            def send():
                method = (
                    client.chat.completions.create
                    if kind == "chat"
                    else client.responses.create
                )
                response = method(**payload)
                try:
                    text = (
                        response.choices[0].message.content
                        if kind == "chat"
                        else response.output_text
                    )
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError("Empty judge response")
                except (AttributeError, IndexError, TypeError, ValueError) as error:
                    dump = (
                        response.model_dump_json()
                        if hasattr(response, "model_dump_json")
                        else str(response)
                    )
                    raise JudgeAPIError(
                        "Malformed/empty SDK response", body=dump
                    ) from error
                return response

            return self.call(send, {"api": kind, "payload": payload})

        return SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: create("chat", **kw))
            ),
            responses=SimpleNamespace(create=lambda **kw: create("responses", **kw)),
        )


def score_with_recovery(scorer, code, bundle, work, workers):
    """Re-enter native resumable scorers; never regenerate images or train."""
    history = []
    for attempt in range(1, SCORING_PASSES + 1):
        events = work / "retry-events" / uuid4().hex
        current = {**bundle, "_retry_events": str(events)}
        entry = {
            "pass": attempt,
            "workers": workers,
            "events": str(events),
            "status": "running",
        }
        history.append(entry)
        write_json(work / "retry_status.json", {"passes": history})
        try:
            result = scorer(code, current, work, workers)
        except IncompleteJudgements as error:
            permanent = any(
                not read_json(p)["retryable"] for p in events.glob("*.json")
            )
            entry.update(status="incomplete", error=redact_secret(str(error)))
            write_json(work / "retry_status.json", {"passes": history})
            if permanent or not error.retryable or attempt == SCORING_PASSES:
                raise
            delay = min(60, 30 * attempt)
            print(
                f"[scoring] Pass {attempt} incomplete; retry in {delay}s; "
                "successful caches are retained.",
                flush=True,
            )
            time.sleep(delay)
            workers = max(1, workers // 2)
        except Exception as error:
            entry.update(status="failed", error=redact_secret(str(error)))
            write_json(work / "retry_status.json", {"passes": history})
            raise
        else:
            entry["status"] = "complete"
            write_json(work / "retry_status.json", {"passes": history})
            return result
