"""Privacy-conscious standard logging setup."""

import logging
from contextvars import ContextVar, Token
from threading import Lock

from opentelemetry import trace
from opentelemetry.trace import Span

_request_log_context: ContextVar[tuple[Span, list[str]] | None] = ContextVar(
    "request_log_context", default=None
)
_LOG_ATTRIBUTE = "debug.logs"
_MAX_LOG_TEXT_LENGTH = 16_000
_log_lock = Lock()


def bind_request_log_span(span: Span) -> Token:
    """Remember the HTTP request span so child-span logs are visible together."""
    return _request_log_context.set((span, []))


def reset_request_log_span(token: Token) -> None:
    _request_log_context.reset(token)


class SpanTextLogHandler(logging.Handler):
    """Append readable application log lines to the parent HTTP span."""

    _timestamp_formatter = logging.Formatter()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            request_context = _request_log_context.get()
            span = request_context[0] if request_context else trace.get_current_span()
            if not span.is_recording():
                return
            message = record.getMessage().replace("\r", "\\r").replace("\n", "\\n")
            timestamp = self._timestamp_formatter.formatTime(record)
            log_line = f"{timestamp} {record.levelname} {record.name}: {message}"

            # Span attributes are directly visible as tags in Jaeger. Serialize
            # read/append/write so concurrent request tasks do not lose lines.
            with _log_lock:
                lines = request_context[1] if request_context else []
                lines.append(log_line)
                combined = "\n".join(lines)
                if len(combined) > _MAX_LOG_TEXT_LENGTH:
                    combined = "[earlier log text truncated]\n" + combined[-(_MAX_LOG_TEXT_LENGTH - 31):]
                    lines[:] = combined.splitlines()
                span.set_attribute(_LOG_ATTRIBUTE, combined)
        except Exception:
            self.handleError(record)


def configure_logging(level: str = "INFO", capture_logs_in_spans: bool = True) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    if capture_logs_in_spans:
        logging.getLogger().addHandler(SpanTextLogHandler())
