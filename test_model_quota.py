import os
import unittest
from unittest.mock import Mock, patch

import httpx
from openrouter.errors import OpenRouterError

import model_quota


class ModelQuotaTest(unittest.TestCase):
    def test_shared_spacing_and_rate_limit_recovery(self):
        for headers, cooldown in [({}, 60), ({"Retry-After": "12"}, 12)]:
            now = [100.0]

            def sleep(seconds):
                now[0] += seconds

            timestamps = []
            error = OpenRouterError("rate limited", httpx.Response(429, headers=headers))

            def accommodation():
                timestamps.append(now[0])
                if len(timestamps) == 1:
                    raise error
                return "rooms"

            with self.subTest(headers=headers), patch.object(model_quota, "_next_request", 0), \
                    patch.dict(os.environ, {"TRIPPLANNER_MODEL_RPM": "20"}), \
                    patch.object(model_quota.time, "monotonic", side_effect=lambda: now[0]), \
                    patch.object(model_quota.time, "sleep", side_effect=sleep):
                model_quota.call_with_quota(lambda: "visa")
                self.assertEqual(model_quota.call_with_quota(accommodation), "rooms")
                self.assertEqual(timestamps, [103, 103 + cooldown])

    def test_retry_bound_and_non_rate_errors(self):
        for code, attempts in [(429, 3), (401, 1)]:
            call = Mock(side_effect=OpenRouterError("provider error", httpx.Response(code)))
            with patch.object(model_quota, "_wait_for_slot"), patch.object(model_quota, "_next_request", 0):
                with self.assertRaises(OpenRouterError):
                    model_quota.call_with_quota(call)
            self.assertEqual(call.call_count, attempts)
