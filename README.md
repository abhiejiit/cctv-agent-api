# CCTV Finder Agent

Mock-backed CCTV question-answering API. Requests pass through authentication,
input guardrails, LLM intent classification, and a CrewAI agent that can call a
camera-scoped tool. The tools currently return synthetic fixtures. Recent
conversation context is stored in Redis; live video ingestion and MongoDB
persistence are not implemented yet.

## Requirements

- Python 3.11 or newer
- A Gemini API key available as `GEMINI_API_KEY` (Gemini 3.5 Flash-Lite is
  configured in the sample settings; Ollama remains supported)
- A model available to the configured provider
- CrewAI with LiteLLM and Google Gen AI support for Ollama and Gemini agent calls

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Use Gemini API instead of Ollama

The classifier, input guardrail, and CrewAI agents use Gemini in the sample
configuration. Keep the API key in an environment variable; do not put it in
YAML or commit it. The model ID in the sample is `gemini-3.5-flash-lite`; change
it if that model is not enabled for your key:

```yaml
llm:
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
  model: gemini-3.5-flash-lite
  api_key_environment_variable: GEMINI_API_KEY

guardrail_llm:
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
  model: gemini-3.5-flash-lite
  api_key_environment_variable: GEMINI_API_KEY
```

Set the key in the shell before starting the API. You can also keep the
guardrail on Ollama by setting its provider back to `ollama`; the main
classifier and CrewAI agents use the `llm` setting. Model availability and
free-tier quotas are controlled by Google for the API key. The application
readiness endpoint checks that the configured model is listed and supports
`generateContent`.

Start Redis locally in a separate terminal:

```bash
docker run --rm --name cctv-redis -p 6379:6379 redis:7-alpine
```

Conversation history is enabled by default. The API stores at most five
completed exchanges (five user messages and five agent replies) per authenticated
user and household. When a sixth exchange is appended, Redis removes the oldest
pair. The queue expires after seven days without activity. Override the local
Redis URL with `CCTV_REDIS_URL`; for authenticated Redis deployments, put the
credentials in that environment variable rather than YAML. If Redis is
unavailable, camera requests continue without history and `/health/ready`
reports the dependency as unavailable.

For Jaeger debugging, `tracing.capture_conversation_history` is enabled in the
development config. The request workflow parent span and the
`cctv.conversation_history.load` / `cctv.conversation_history.append` child spans
show the queue as formatted JSON in `cctv.conversation_history.queue`, along
with its status, turn count, and truncation flag. This attribute contains user
messages and agent replies. Turn it off outside local debugging, or set
`CCTV_TRACE_CAPTURE_CONVERSATION_HISTORY=false` to override it.

Clear the current authenticated user's queue with `POST /conversation-history`.
The endpoint deletes only that household/user key and returns `{"status":"cleared"}`;
it does not flush the shared Redis database.

The API dependency set does not install the optional vision experiment
packages. Install `requirements-vision.txt` only when working on the local YOLO
experiment.

The provider, model name, base URL, chat/readiness paths, sampling options,
request limits, authentication mode, and guardrail patterns are in
`configs/app.yaml`. Configure each provider there, or override the classifier
with `CCTV_LLM_PROVIDER`, `CCTV_LLM_BASE_URL`, and `CCTV_LLM_MODEL`. Use
`CCTV_GUARDRAIL_LLM_PROVIDER`, `CCTV_GUARDRAIL_LLM_BASE_URL`, and
`CCTV_GUARDRAIL_LLM_MODEL` for the guardrail. Camera intent descriptions and aliases are in
`configs/cameras.yaml`.

The classifier uses `llm`, while the topical input guardrail
uses the separate `guardrail_llm` configuration (Llama3 by default). After a
successful classification, `configs/agent.yaml` provides a per-intent role, goal,
backstory, instructions, examples, expected reply, and tool assignment. Set
`tools_enabled: true` and list tool names under `tools` to expose that agent's
functions. Agents receive only their configured functions: parking, for example,
has both entity details and occupancy tools, while the entrance and lawn agents
have their own region-specific details functions. Greetings, clarification, and
out-of-scope agents have no tools. Tool names are resolved from a fixed Python
registry in `app/orchestration/camera_tools.py`; YAML cannot import or execute
arbitrary code. Each function declares its input types, so CrewAI fills tool
arguments such as `entity` from the user's question when it calls the function.

Each tool is a Python function in `app/orchestration/camera_tools.py`. It
validates its inputs with its dedicated Pydantic schema from
`app/orchestration/tool_schemas.py`, then calls the camera data service through
`app/services/camera_service_provider.py`. The provider currently returns
`MockCameraDataService` from `app/services/mock_camera_data_service.py`, which
returns YOLO-shaped JSON with camera ID, capture time, detected class,
confidence, bounding box, and attributes. Replacing the provider with a live
`CameraDataService` implementation will leave the agent tools and their schemas
unchanged. Camera agents read the service JSON and answer the user's question
from it. Each agent also has a configurable `capabilities` list. When a camera
agent cannot resolve a request from its tool result, it politely explains the
limit and offers only its listed capabilities. The greetings agent responds
only with the capability menu so the user can choose a supported question.
The classifier and selected agent review all recent exchanges as context, so
they can follow topic changes, corrections, unresolved clarifications, and
short references across turns. The current request remains primary, and
history is passed as untrusted context. Current camera-state
questions still require a camera-tool call; prior answers may be stale. Only
guardrail-allowed requests with an agent response are appended. Blocked and
failed requests are not saved. Redis keys are derived from a hash of the
authenticated household and user identifiers.
Tool names and durations are logged in CrewAI spans. Prompt, response, and tool
payload content capture is disabled by default; enable it only in a controlled
development environment. Greetings, clarification, and out-of-scope agents
have no tools. `crewai.verbose` defaults to false to avoid verbose provider and
tool output. The agent file is selected by `agent_config_file` in
`configs/app.yaml`.

The classifier has a bounded 256-token output budget by default so the model can finish
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
CrewAI route, kickoff, and completion logs are also attached to the
`cctv.crewai.intent_agent` span under the same `debug.logs` attribute. Expand
that span to inspect agent role, configured tools, model settings, execution
time, step count, and token usage when returned by CrewAI.
When prompt capture is enabled, the `cctv.crewai.task.execute` child span shows
the task description and user input. User input is serialized as JSON in the
task prompt so delimiter-like text remains data.

Each tool call creates a child span named
`cctv.crewai.tool.<tool_name>`. The child span contains the validated input under
`crewai.tool.input.json` and the service response under
`crewai.tool.output.json` only when their respective capture flags are enabled;
camera ID, data source, duration, response size, and success/error details are
always recorded. The latest tool input and response are also copied onto
the parent intent-agent span as `crewai.tool.last.input.json` and
`crewai.tool.last.output.json` only when payload capture is enabled. Agent step
spans/logs record the selected tool and payload lengths by default; arguments,
results, and final output require their respective capture options. Private
reasoning text is not captured. Tool payload capture is controlled by
`tracing.capture_tool_inputs` and `tracing.capture_tool_outputs` in
`configs/app.yaml` (both disabled by default). Large tool
payloads are capped at 12,000 characters per span attribute, with truncation
recorded explicitly.

LLM prompt and response capture are
controlled by `tracing.capture_llm_prompts` and `tracing.capture_llm_outputs`
in `configs/app.yaml`. With both enabled, inspect the
`cctv.input_guardrail.llm` and `cctv.llm.classify` spans for
`llm.prompt.system`, `llm.prompt.user`, and `llm.response.structured`.
The classifier span also has `llm.response.intent` and
`llm.response.reason`; `reason` is the short structured rationale returned by
the classifier, not private model reasoning. These fields can contain user
messages and model output, so disable prompt/output capture when trace access
should not include that data. CrewAI step inputs/results and final output obey
the same capture settings.

The capture settings can also be overridden with `CCTV_TRACE_CAPTURE_LOGS`,
`CCTV_TRACE_CAPTURE_LLM_PROMPTS`, `CCTV_TRACE_CAPTURE_LLM_OUTPUTS`,
`CCTV_TRACE_CAPTURE_CONVERSATION_HISTORY`,
`CCTV_TRACE_CAPTURE_TOOL_INPUTS`, and `CCTV_TRACE_CAPTURE_TOOL_OUTPUTS` using
`true` or `false`.

The classifier and guardrail can use separate provider and model settings.
When using Ollama, a small non-thinking model usually has lower latency; choose
a thinking model when its classification improvement justifies the added delay:

```yaml
# Main intent classifier
llm:
  model: qwen3:8b
  thinking_model: true
  think: true

# Input guardrail
guardrail_llm:
  model: llama3
  thinking_model: false
  think: false
```

The API sends the `think` field only when `thinking_model` is enabled. The
guardrail uses `guardrail_llm`, while intent classification uses `llm`. If a
provider returns a separate `message.thinking` field, the classifier trace
records `llm.response.thinking_present` and the application log records its
character count. Private reasoning text is not copied into logs or traces; the
classifier's concise structured `reason` field remains available for routing
diagnostics.

For an API process running on the host, `localhost:4318` is correct. If the API
itself runs in a Docker container on the same Docker network as Jaeger, set
`CCTV_OTLP_ENDPOINT=http://jaeger:4318/v1/traces` (use the Jaeger container or
service name instead of `localhost`). Set `CCTV_TRACING_ENABLED=false` to turn
tracing off. Sampling ratio and service name are configurable in
`configs/app.yaml` under `tracing`.

The API exposes Prometheus text metrics at `/metrics`. Metrics are process
local and use only fixed stage, status, and age labels; they do not include user
messages, user IDs, or camera IDs. In production, configure
`metrics.auth_token_environment_variable` and provide that bearer token through
the environment; production startup fails if it is missing.

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
  "answer": "Demo data only (mock snapshot, not live footage): I detected one car.",
  "reason": "The question asks about a vehicle in the parking area.",
  "status": "classified",
  "guardrail_status": "allowed",
  "guardrail_findings": [],
  "classification_source": "llm",
  "observation": {
    "source": "mock_yolo_camera_service",
    "data_status": "mock",
    "captured_at": "2026-11-08T08:00:00Z"
  }
}
```

The `reason` field is a short, user-visible explanation limited to 30
whitespace-separated tokens. It explains the routing decision without exposing
private model reasoning.

## LLM input guardrails

After deterministic size/pattern checks and before intent classification, the
API calls the configured LLM guardrail filter. Its topics and policy are
configured under `guardrails.llm_filter` in `configs/app.yaml`. The default
topics are escalation, dispute, harassment, and footage tampering; each topic
has a description and examples to guide the filter. Direct footage deletion,
alteration, concealment, and recording-disable requests also match the
configurable deterministic `footage_tampering` pattern. Edit the policy,
topics, examples, or pattern in `configs/app.yaml` to adjust these checks. Set
`enabled: false` to disable the LLM filter; deterministic pattern checks still
run.

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
Readiness checks the configured classifier model and, when enabled, guardrail
model. `/metrics` exposes process-local Prometheus metrics with bounded labels
for stage duration, operation outcomes, request outcomes, and rate limiting.
These metrics and the in-process rate limiter are single-process MVP features;
keep the metrics endpoint private to the service network and use shared
infrastructure before running multiple API replicas.

## Bearer authentication mode

For a basic non-development setup, edit `configs/app.yaml` to set
`auth.mode: bearer`, then set each referenced token environment variable. For
example:

```bash
export CCTV_DEMO_BEARER_TOKEN='replace-with-a-long-random-token'
```

The bearer token is mapped to the configured user, household, and allowed
camera IDs in the YAML. Camera tools resolve camera IDs and entity/property
allowlists from `configs/cameras.yaml` and check those IDs against the
authenticated principal before fetching data.
This is a bootstrap mechanism; production deployments should connect the
service to the selected identity provider and secret manager. Development
header authentication is rejected when `CCTV_ENVIRONMENT=production`.

## Safety and initial deployment limits

- Guardrails are deterministic, configurable checks, not a complete defense
  against every adversarial message. Keep the classifier instruction bounded
  and never treat model output as authorization.
- Conversation history is Redis-backed and process-independent, but this first
  version uses one rolling history per user and household rather than separate
  named chat threads.
- The rate limiter is in-process and applies to one API process. Use shared
  storage when operating multiple replicas.
- Keep the LLM provider on a trusted network and apply appropriate network
  controls. Configure `llm.base_url` and endpoint paths for the API adapter.
- The current camera provider is synthetic demo data. API responses identify
  mock observations and do not represent live camera state.
- The API enforces the configured request deadline and bounded request/agent
  concurrency. The defaults are intended for a small pilot, not a multi-replica
  deployment.

The configured adapter receives the `IntentClassification` JSON schema so the
provider can constrain its output to the supported intent set. Ollama is the
default local adapter; see its [structured outputs guide](https://ollama.com/blog/structured-outputs)
and [API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md).
