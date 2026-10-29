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

Start the API:

```bash
uvicorn app.main:app --reload
```

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

The API supports these intent labels: `front_entrance`, `parking`,
`back_entrance`, `common_lawn`, `greetings`, and `intent_clarify`. A blocked
message is not sent to the LLM and is routed to `intent_clarify`. If the
configured provider is unavailable or returns an invalid response, the request fails closed to
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
