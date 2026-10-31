# CCTV Finder Agent

Initial API implementation for the first three architecture steps: application
configuration, authenticated request intake with request state and input
guardrails, and LLM-based intent classification through a configurable local
LLM provider.

This slice ends after classification. It does not yet run a CrewAI flow, query a
camera service, inspect live video, persist conversations, or use Redis/MongoDB.
The endpoint returns an intent so the next implementation step can route that
state into the corresponding CrewAI agent.

## Requirements

- Python 3.11 or newer
- A locally hosted LLM provider that supports structured output; Ollama is the
  configured adapter by default
- A model available to the configured provider

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull llama3.2:3b 
```

The provider, model name, base URL, chat/readiness paths, sampling options,
request limits, authentication mode, and guardrail patterns are in
`configs/app.yaml`. Change the provider settings there, or override the URL and
model with `CCTV_LLM_BASE_URL` and `CCTV_LLM_MODEL`. Camera intent descriptions and aliases are in
`configs/cameras.yaml`.

The classifier has a 128-token output budget by default so the model can finish
the structured JSON response, including both the intent and short reason.

Start the API:

```bash
uvicorn app.main:app --reload
```

## Jaeger request tracing

Tracing is enabled in `configs/app.yaml` and exports OpenTelemetry traces over
OTLP/HTTP to `http://localhost:4318/v1/traces`. Start Jaeger locally in a
separate terminal:

```bash
docker run --rm --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  jaegertracing/jaeger:latest
```

Then start/restart the API, send a request to `/cctv-agent`, and open
<http://localhost:16686>. In Jaeger, select service `cctv-finder-api`, choose
operation `POST /cctv-agent`, and search. A trace contains the FastAPI server
span plus workflow, guardrail, intent-classification, LLM, and outgoing HTTPX
spans. The request ID is attached as the `cctv.request_id` span attribute.

Request-associated Python logs are combined into the readable `debug.logs`
text attribute on the top-level HTTP request span. Expand the `POST
/cctv-agent` span and inspect its tags/attributes to see timestamped log lines
from the guardrail, workflow, and classifier together. The attribute is capped
at 16,000 characters; older lines are truncated if a request produces more.
LLM prompt and response capture are
controlled by `tracing.capture_llm_prompts` and `tracing.capture_llm_outputs`
in `configs/app.yaml`. With both enabled, inspect the
`cctv.input_guardrail.llm` and `cctv.llm.classify` spans for
`llm.prompt.system`, `llm.prompt.user`, and `llm.response.structured`.
The classifier span also has `llm.response.intent` and
`llm.response.reason`; `reason` is the short structured rationale returned by
the classifier, not private model reasoning. These fields can contain user
messages and model output, so disable prompt/output capture when trace access
should not include that data.

The capture settings can also be overridden with `CCTV_TRACE_CAPTURE_LOGS`,
`CCTV_TRACE_CAPTURE_LLM_PROMPTS`, and `CCTV_TRACE_CAPTURE_LLM_OUTPUTS` using
`true` or `false`.

For an API process running on the host, `localhost:4318` is correct. If the API
itself runs in a Docker container on the same Docker network as Jaeger, set
`CCTV_OTLP_ENDPOINT=http://jaeger:4318/v1/traces` (use the Jaeger container or
service name instead of `localhost`). Set `CCTV_TRACING_ENABLED=false` to turn
tracing off. Sampling ratio and service name are configurable in
`configs/app.yaml` under `tracing`.

## Request

Development mode is enabled by default and uses a local development identity.
It can be overridden with `X-User-ID` and `X-Household-ID` headers for local
experiments only.

```bash
curl http://127.0.0.1:8000/cctv-agent \
  -H 'Content-Type: application/json' \
  -H 'X-User-ID: demo-user' \
  -d '{"message":"Is there a car in the parking lot?"}'
```

Example response:

```json
{
  "request_id": "...",
  "intent": "parking",
  "reason": "The question asks about a vehicle in the parking area.",
  "status": "classified",
  "guardrail_status": "allowed",
  "guardrail_findings": [],
  "classification_source": "llm"
}
```

The `reason` field is a short, user-visible explanation limited to 30
whitespace-separated tokens. It explains the routing decision without exposing
private model reasoning.

## LLM input guardrails

After deterministic size/pattern checks and before intent classification, the
API calls the configured LLM guardrail filter. Its topics and policy are
configured under `guardrails.llm_filter` in `configs/app.yaml`. The default
topics are escalation, dispute, and harassment; each topic has a description
and examples to guide the filter. Edit those settings to adjust the policy or
topics. Set `enabled: false` to disable the LLM filter.

When the filter flags a request, intent classification is skipped and the API
returns the common message `we're sorry, we cannot proceed with that`, with
`status: guardrail_blocked`. If the filter cannot reach the LLM or validate its
response, the request also stops safely with `status: guardrail_unavailable`;
the camera intent classifier is not called in either case. A request that
passes the filter proceeds to the normal intent classifier.

The API supports these intent labels: `front_entrance`, `parking`,
`back_entrance`, `common_lawn`, `greetings`, `intent_clarify`, and
`out_of_scope`. A request that spans multiple camera zones, or mixes a camera
question with an unsupported task, is routed to `intent_clarify`; a wholly
unrelated request is routed to `out_of_scope`. Requests blocked by deterministic
input checks are not sent to either LLM. Requests flagged by the topical LLM
filter are not sent to intent classification. If the intent classifier is
unavailable or returns an invalid response, the request fails closed to
`intent_clarify` with `classification_fallback` status.

Readiness and liveness are available at `/health/ready` and `/health/live`.
The ready endpoint checks reachability of the configured LLM provider.

## Bearer authentication mode

For a basic non-development setup, edit `configs/app.yaml` to set
`auth.mode: bearer`, then set each referenced token environment variable. For
example:

```bash
export CCTV_DEMO_BEARER_TOKEN='replace-with-a-long-random-token'
```

The bearer token is mapped to the configured user and household in the YAML.
This is a bootstrap mechanism; production deployments should connect the
service to the selected identity provider and secret manager. Development
header authentication is rejected when `CCTV_ENVIRONMENT=production`.

## Safety and initial deployment limits

- Guardrails are deterministic, configurable checks, not a complete defense
  against every adversarial message. Keep the classifier instruction bounded
  and never treat model output as authorization.
- This endpoint has no conversation history yet; Redis/MongoDB history arrives
  in the persistence step of the architecture.
- The rate limiter is in-process and applies to one API process. Use shared
  storage when operating multiple replicas.
- Keep the LLM provider on a trusted network and apply appropriate network
  controls. Configure `llm.base_url` and endpoint paths for the API adapter.
- The current classifier produces a route label only. It does not claim that a
  person, vehicle, door, or other object is present in camera footage.

The configured adapter receives the `IntentClassification` JSON schema so the
provider can constrain its output to the supported intent set. Ollama is the
default local adapter; see its [structured outputs guide](https://ollama.com/blog/structured-outputs)
and [API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md).
