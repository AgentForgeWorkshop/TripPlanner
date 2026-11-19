"""Share the model request budget across the synchronous trip-planning agents."""
import logging
import math
import os
import threading
import time
from email.utils import parsedate_to_datetime

from langchain.agents.middleware import wrap_model_call

_lock = threading.Lock()
_next_request = 0.0


def _wait_for_slot():
    global _next_request
    rpm = int(os.getenv("TRIPPLANNER_MODEL_RPM", "20"))
    if rpm <= 0:
        raise ValueError("TRIPPLANNER_MODEL_RPM must be positive")
    while True:
        with _lock:
            now = time.monotonic()
            delay = _next_request - now
            if delay <= 0:
                _next_request = now + 60 / rpm
                return
        time.sleep(delay)


def _retry_delay(error):
    current = error
    while current is not None:
        response = getattr(current, "response", None)
        headers = getattr(current, "headers", None) or getattr(response, "headers", {})
        value = headers.get("retry-after")
        if value:
            try:
                seconds = float(value)
            except ValueError:
                try:
                    seconds = parsedate_to_datetime(value).timestamp() - time.time()
                except (TypeError, ValueError, OverflowError):
                    seconds = None
            if seconds is not None and math.isfinite(seconds):
                return max(0, seconds)
        current = current.__cause__
    return 60.0


def call_with_quota(call, *args):
    global _next_request
    for attempt in range(3):
        _wait_for_slot()
        try:
            return call(*args)
        except Exception as error:
            if getattr(error, "status_code", None) != 429:
                raise
            delay = _retry_delay(error)
            with _lock:
                _next_request = max(_next_request, time.monotonic() + delay)
            if attempt == 2:
                raise
            logging.getLogger(__name__).warning(
                "Model rate limit reached. Retrying after %.1f seconds.", delay
            )


@wrap_model_call
def pace_model_request(request, handler):
    return call_with_quota(handler, request)
