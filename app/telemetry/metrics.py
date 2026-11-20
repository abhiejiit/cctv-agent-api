"""Small bounded-cardinality Prometheus metrics for the single-process MVP."""

from __future__ import annotations

from collections import Counter, defaultdict
from threading import Lock


_STAGES = frozenset(
    {"guardrails", "guardrail_llm", "classifier", "agent", "camera_tool", "conversation_history", "request"}
)
_OUTCOMES = frozenset({"ok", "error", "timeout", "rejected", "blocked"})
_OBSERVATION_STATUSES = frozenset({"mock", "fresh", "stale", "unavailable"})
_AGE_BUCKETS = frozenset({"le_10s", "le_60s", "gt_60s"})
_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


class AppMetrics:
    """Process-local metrics; labels are restricted to known stage/outcome values."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._stage_counts: Counter[tuple[str, str]] = Counter()
        self._stage_sums: Counter[str] = Counter()
        self._stage_buckets: dict[str, list[int]] = defaultdict(lambda: [0] * len(_BUCKETS))
        self._request_counts: Counter[str] = Counter()
        self._rate_limited = 0
        self._observation_counts: Counter[tuple[str, str]] = Counter()

    def observe_stage(self, stage: str, outcome: str, elapsed_seconds: float) -> None:
        if stage not in _STAGES or outcome not in _OUTCOMES:
            return
        elapsed_seconds = max(0.0, elapsed_seconds)
        with self._lock:
            self._stage_counts[(stage, outcome)] += 1
            self._stage_sums[stage] += elapsed_seconds
            for index, bucket in enumerate(_BUCKETS):
                if elapsed_seconds <= bucket:
                    self._stage_buckets[stage][index] += 1

    def observe_request(self, outcome: str) -> None:
        if outcome in _OUTCOMES:
            with self._lock:
                self._request_counts[outcome] += 1

    def observe_rate_limit(self) -> None:
        with self._lock:
            self._rate_limited += 1

    def observe_camera_observation(self, status: str, age_seconds: float) -> None:
        if status not in _OBSERVATION_STATUSES:
            return
        age_bucket = "le_10s" if age_seconds <= 10 else "le_60s" if age_seconds <= 60 else "gt_60s"
        with self._lock:
            self._observation_counts[(status, age_bucket)] += 1

    def render_prometheus(self) -> str:
        with self._lock:
            lines = [
                "# HELP cctv_stage_operations_total Completed operations by stage and outcome.",
                "# TYPE cctv_stage_operations_total counter",
            ]
            for (stage, outcome), count in sorted(self._stage_counts.items()):
                lines.append(
                    f'cctv_stage_operations_total{{stage="{stage}",outcome="{outcome}"}} {count}'
                )
            lines.extend(
                [
                    "# HELP cctv_stage_duration_seconds Stage duration distribution.",
                    "# TYPE cctv_stage_duration_seconds histogram",
                ]
            )
            for stage in sorted(_STAGES):
                for index, bucket in enumerate(_BUCKETS):
                    lines.append(
                        f'cctv_stage_duration_seconds_bucket{{stage="{stage}",le="{bucket}"}} {self._stage_buckets[stage][index]}'
                    )
                count = sum(self._stage_counts[(stage, outcome)] for outcome in _OUTCOMES)
                lines.append(
                    f'cctv_stage_duration_seconds_bucket{{stage="{stage}",le="+Inf"}} {count}'
                )
                lines.append(f'cctv_stage_duration_seconds_count{{stage="{stage}"}} {count}')
                lines.append(f'cctv_stage_duration_seconds_sum{{stage="{stage}"}} {self._stage_sums[stage]}')
            lines.extend(
                [
                    "# HELP cctv_requests_total Completed requests by outcome.",
                    "# TYPE cctv_requests_total counter",
                ]
            )
            for outcome, count in sorted(self._request_counts.items()):
                lines.append(f'cctv_requests_total{{outcome="{outcome}"}} {count}')
            lines.extend(
                [
                    "# HELP cctv_camera_observations_total Camera tool results by status and age bucket.",
                    "# TYPE cctv_camera_observations_total counter",
                ]
            )
            for observation_status in sorted(_OBSERVATION_STATUSES):
                for age_bucket in sorted(_AGE_BUCKETS):
                    count = self._observation_counts[(observation_status, age_bucket)]
                    lines.append(
                        f'cctv_camera_observations_total{{status="{observation_status}",age_bucket="{age_bucket}"}} {count}'
                    )
            lines.extend(
                [
                    "# HELP cctv_rate_limited_total Requests rejected by the local rate limiter.",
                    "# TYPE cctv_rate_limited_total counter",
                    f"cctv_rate_limited_total {self._rate_limited}",
                ]
            )
        return "\n".join(lines) + "\n"
