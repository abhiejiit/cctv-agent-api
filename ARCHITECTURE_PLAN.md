# CCTV question-answering assistant: architecture review and execution plan

## 1. Executive summary

The product idea is sound: let an authenticated resident ask natural-language questions about current conditions in named camera zones, resolve the zone and question, obtain fresh structured observations, and answer with clear uncertainty. The key design boundary is that computer vision should produce timestamped observations, deterministic application code should decide which observations the user may access and whether they answer the question, and the language model should help interpret and explain—not invent facts or control cameras.

For the first production-shaped prototype, use **one authenticated API, a stateful request workflow, an intent-classification LLM call, and a post-classification CrewAI Flow**. The classifier writes one allowed intent to state. CrewAI always starts afterward and switches on that value to select a camera agent or the greetings/clarification agent. Each camera agent is instructed to call its scoped tool with arguments it extracts from the user message (for example, `fetch_front_entrance_details(entity="door")`), receive tailored JSON from the observation service, and answer from that result. Application code enforces authorization, freshness, tool validation, and response grounding.

The current repository contains only a YOLO tracking script, a model weights file, and a short requirements file. It currently loads a local MP4 path, tracks objects, and displays/saves results; there is no HTTP service, live stream ingestion, authentication, LLM, persistence, or tests. This plan starts with a safe mock-backed end-to-end API, then integrates live video incrementally.

## 2. Product scope and assumptions

### Initial user promise

Answer questions such as “Is there a car in the parking lot?”, “Is the front entrance door closed?”, “Is the lawn clear?”, and “Are there vehicles outside?” from observations associated with cameras the authenticated user is allowed to query. Answers must identify when the observation was captured and say when the camera is stale, unavailable, obscured, or unable to support the requested inference.

### Recommended MVP boundary

- Read-only observation questions only. No door unlocking, alarm changes, camera movement, or other actuation.
- One household/account per request, with server-derived user and tenant identity.
- Camera/zone aliases are explicitly configured per household; do not infer that a camera exists from a model response.
- Tool data is mocked initially but follows the same schema and freshness rules as the eventual vision service.
- “Current” has an explicit freshness limit (initially configurable, e.g. 10–30 seconds for a live snapshot); older data is labeled stale or triggers a refresh.
- No face recognition, identity claims, or conclusions about intent/behavior in the MVP. Describe visible objects and configured device states only.

### Questions to decide during discovery (do not block scaffolding)

1. What camera protocol and hardware are in scope (RTSP, vendor API, NVR, uploaded clips)?
2. What latency/freshness target is acceptable, and how many concurrent cameras/users are expected?
3. Which regions/zones and household membership model should be supported?
4. Which language(s), retention duration, deployment region, and privacy obligations apply?
5. Is “fan on” required to come from visible motion, a smart-home sensor, or both? Video alone may not prove electrical state.

## 3. Review of the proposed design

### Strong parts to preserve

- Keeping the API behind authentication and authorization.
- Asking a clarification question when the room/object cannot be resolved safely.
- Using structured tools/data for camera observations instead of asking the LLM to guess from its general knowledge.
- Limiting conversational context to the minimum needed for follow-ups.
- Building a mock-data vertical slice before connecting physical camera feeds.

### Changes recommended

1. **Do not treat “input guardrails” as a security boundary.** Normalize and size-limit input, reject malformed requests, apply abuse controls, and optionally use a policy classifier. But prompt injection cannot be reliably removed by a filter. Keep the model away from secrets and privileged operations; pass user text as data; strictly validate structured output; ensure tools independently enforce authorization. OWASP describes prompt injection as a structural risk and recommends reducing impact through privilege minimization, validation, and controls rather than assuming a filter can eliminate it. [OWASP prompt injection guidance](https://genai.owasp.org/glossaries/prompt-injection/)

2. **Basic Auth is acceptable only as a tightly controlled prototype choice.** For deployed user sessions, use a standard identity provider/session or short-lived access tokens over TLS, with server-side authorization checks on every request. Never trust a user-supplied `user_id` or `camera_id` as proof of access. If Basic Auth is temporarily required, enforce TLS, hashed credentials, rotation, rate limiting, and avoid logging the Authorization header.

3. **Separate live video processing from chat request handling.** A request should not start an unbounded stream loop. Run camera ingestion/inference workers independently, publish normalized latest observations (and optionally events), and let the API query a recent observation or request a bounded snapshot. This makes latency, retries, camera outages, and scaling manageable.

4. **Use camera-specific agents with narrow responsibilities.** Since the desired design assigns a camera agent after classification, configure one agent per camera intent/region where its dedicated tool scope is useful. Keep the registry and Flow routing deterministic; avoid duplicating agent code for cameras with identical behavior. Model `unsupported`, `not_authorized`, `camera_unavailable`, and `stale_data` explicitly alongside clarification.

5. **Keep classifier output to one schema-constrained intent label.** Accept only `front_entrance`, `parking`, `back_entrance`, `common_lawn`, `greetings`, or `intent_clarify`. The classifier does not extract entities or select cameras/tools. Each camera agent determines the requested entity/property from the message and fills its own scoped tool arguments; application code and the tool validate access and capability.

6. **Conversation storage needs clear consistency and privacy semantics.** Redis is a short-lived cache, not the authoritative history. Store messages in MongoDB with tenant/user scope and retention rules; update the cache only after durable persistence succeeds (or use an outbox/queue if asynchronous persistence is required). Use TTL, bounded token/context budgets, encryption and access controls. Load only the last relevant turns, not blindly every last five messages—follow-ups may need a compact state summary, while unrelated messages should not influence routing.

7. **Computer-vision detections are not equivalent to state truth.** Object detection can answer “a car was detected in this region”; it cannot reliably establish “the door is closed” or “the fan is on” without a suitable state estimator, camera angle, calibration, temporal evidence, and confidence threshold. Prefer dedicated door/fan sensors where available. Expose uncertainty, model version, observation time, and evidence quality.

8. **Make tools read-only and least-privileged.** Each camera agent receives only its scoped camera tool, whose camera scope is fixed in server configuration and whose arguments are validated. Tools return only that camera’s allowed, minimal data. They must not accept arbitrary URLs, filesystem paths, Mongo queries, raw SQL, or model-selected camera identifiers.

## 4. Target architecture

```mermaid
flowchart LR
  subgraph CameraPipeline[Background camera pipeline]
    Cam[Cameras / NVR] --> Worker[Ingestion workers]
    Worker --> Vision[Vision detection and state estimation]
    Vision --> ObsDB[(🍃 MongoDB observations)]
  end

  User[Authenticated user] --> API[POST /cctv-agent]
  API --> Auth[Authenticate and authorize]
  Auth --> Context[Load conversation context]
  Context <--> Cache[(🔴 Redis recent-context cache)]
  ChatDB -. cache miss / durable history .-> Context
  API --> State[Initialize request state]
  Context --> State
  State --> Guard[Input guardrails inspect input and state]
  Guard -->|record guardrail status in state| Classifier[Classifier LLM: intent only]
  Classifier --> Update[Validate and write state.intent]
  Update --> Crew[CrewAI stateful agentic Flow]
  Crew --> Switch{Switch on state.intent}

  Switch -->|front_entrance| Front[Front Entrance Agent]
  Switch -->|parking| Park[Parking Agent]
  Switch -->|back_entrance| Back[Back Entrance Agent]
  Switch -->|common_lawn| Lawn[Common Lawn Agent]
  Switch -->|greetings| Greet[Greetings Agent]
  Switch -->|intent_clarify| Clarify[Clarification Agent]

  Front --> Tool[Camera-scoped tool]
  Park --> Tool
  Back --> Tool
  Lawn --> Tool
  Tool --> ObsSvc[Observation service]
  ObsSvc --> ObsDB
  ObsDB --> ObsSvc
  ObsSvc -->|tailored JSON| Tool
  Tool -->|tool result| Front
  Tool -->|tool result| Park
  Tool -->|tool result| Back
  Tool -->|tool result| Lawn

  Front --> Answer[Agent response]
  Park --> Answer
  Back --> Answer
  Lawn --> Answer
  Greet --> Answer
  Clarify --> Answer
  Answer --> Persist[Persist conversation turn]
  Persist --> ChatDB
  Persist -. refresh cache .-> Cache
  Persist --> Response[Validated API response]
  Response --> User
```

**MongoDB** is the durable database for conversation history and normalized camera observations. 
**Redis** is the fast, TTL-based cache for recent conversation context; on a cache miss, context is loaded from MongoDB, and durable conversation writes refresh Redis. The **Classifier LLM** only emits the intent enum. **CrewAI** is the agentic orchestration framework: its stateful Flow switches on the stored intent and invokes the corresponding LLM-backed agent; camera agents additionally call their scoped tools.

### Request path: stateful request handling, then CrewAI for every classified intent

The request has one typed state that is updated throughout the lifecycle. Input guardrails and intent classification run before CrewAI starts. **After classification, the CrewAI Flow is always invoked**, and it selects the agent using a switch on the validated `state.intent`. This applies to camera intents, `greetings`, and `intent_clarify`.

1. `POST /cctv-agent` validates the HTTP envelope (JSON schema, request id, text length, locale, and optional conversation id). Authentication middleware obtains the principal and server-side household/camera access context.
2. The application creates a typed request state containing the user message, authenticated principal reference, conversation reference, bounded recent context, request metadata, and empty guardrail, intent, and result fields.
3. **Input guardrails** inspect the user message and relevant state/context, apply deterministic validation and policy checks, and update state with `guardrail_status` and findings. Hard-blocked input is marked in state; it still proceeds through classification and CrewAI, where the safe outcome is handled without camera retrieval. Guardrails reduce risk but do not eliminate prompt injection.
4. A regular structured-output **classifier LLM call** runs outside CrewAI. Its only job is to assign one exact intent from the supported enum: `front_entrance`, `parking`, `back_entrance`, `common_lawn`, `greetings`, or `intent_clarify`. It writes the intent to state; it does not select tools, extract tool arguments, or answer the user.
5. Application code validates that the classifier returned an allowed intent and writes the trusted value to `state.intent`. If the value is malformed, use a safe fallback of `intent_clarify` and record the validation failure. Authorization for a camera is checked before that camera's tool can run; the overall CrewAI Flow still starts.
6. The application always starts the CrewAI Flow after classification and passes the typed request state (or a safe projection). A switch reads `state.intent` and selects the corresponding configured agent. Intent values map through a static registry; they are never used as dynamic imports or arbitrary tool names.
7. Camera agents receive only their camera-scoped tools. For example, `front_entrance` routes to `FrontEntranceAgent`, which reads the original user message/context from state, extracts the relevant entity/property, fills the tool arguments, and calls its `fetch_front_entrance_details(...)` tool. The classifier does not supply those tool arguments.
8. The tool validates entity/property against its allowed schema and camera capability configuration, independently enforces access and freshness, calls the observation service, and returns tailored JSON. It never accepts arbitrary camera paths or URLs. Stale/unavailable data is returned as a structured result.
9. Camera agents answer only from the tool result. The `greetings` agent responds conversationally without camera tools; the `intent_clarify` agent asks a concise follow-up and does not query cameras. A guardrail-blocked state takes a safe response path within the Flow. The outer workflow validates the result, updates state, persists conversation turns, refreshes Redis, and returns the API response.

### Suggested API contract

`POST /cctv-agent`

```json
{
  "conversation_id": "optional-opaque-id",
  "message": "Is there a car in the parking lot?",
  "locale": "en"
}
```

Response:

```json
{
  "conversation_id": "opaque-id",
  "answer": "Yes. One car was detected in the parking area at 14:03:12.",
  "status": "answered",
  "camera": {"id": "camera-parking", "label": "Parking"},
  "observed_at": "2026-10-22T08:33:12Z",
  "freshness": "fresh",
  "confidence": 0.94,
  "evidence": [{"type": "object_count", "class": "car", "count": 1}],
  "clarification": null
}
```

Use a stable response status enum such as `answered`, `clarification_required`, `guardrail_blocked`, `camera_unavailable`, `stale_data`, `not_authorized`, and `temporarily_unavailable`. Intent selection remains one of the six fixed intents; response status is separate from intent. Avoid exposing internal prompts, raw model reasoning, credentials, or unnecessary detection metadata to the client.

## 5. Intent and entity model

The classifier's output is only the intent label. It does not extract the entity, choose a tool, or answer the question. Use a strict enum so the CrewAI Flow can switch safely on `state.intent`:

```json
{
  "intent": "parking"
}
```

Allowed values and corresponding agents:

- `front_entrance` → Front Entrance Agent
- `parking` → Parking Agent
- `back_entrance` → Back Entrance Agent
- `common_lawn` → Common Lawn Agent
- `greetings` → Greetings Agent
- `intent_clarify` → Clarification Agent

The classifier selects one allowed label from the user input and relevant conversation context, then writes it into request state. If output is malformed or outside the enum, application validation writes `intent_clarify` as a safe fallback and records the error. The CrewAI Flow always starts after classification and switches on that state field. Camera agents, not the classifier, read the original message/context and determine the entity/property tool arguments. The clarification agent handles ambiguity such as “Is my door open?” and out-of-scope questions with a clarification or safe scope explanation; the greetings agent handles greetings and brief conversational openings.

## 6. Vision and observation data model

Do not expose YOLO library result objects directly to API/LLM. Convert them into a versioned, stable observation contract, for example:

```json
{
  "observation_id": "uuid",
  "tenant_id": "server-side-scope",
  "camera_id": "camera-parking",
  "captured_at": "2026-10-22T08:33:12Z",
  "processed_at": "2026-10-22T08:33:12.180Z",
  "source": "mock|snapshot|stream",
  "pipeline_version": "vision-v1",
  "health": "ok",
  "detections": [
    {"class": "car", "count": 1, "confidence": 0.94, "region": "parking_area"}
  ],
  "states": [
    {"entity": "garage_door", "state": "closed", "confidence": 0.81, "method": "visual_estimator"}
  ],
  "quality": {"occluded": false, "blurred": false}
}
```

Consider storing event summaries and latest-state snapshots separately: events support “did a person appear in the last hour?” while latest-state records support “is there a person now?”. Define time windows and event retention explicitly. Do not store images/video by default unless required; if frames are retained for debugging, restrict access and apply short retention.

## 7. CrewAI orchestration and camera-agent/tool design

The complete request lifecycle uses typed state in the application workflow. Input guardrails and intent classification run before CrewAI starts. After classification, CrewAI is always invoked and uses a switch on `state.intent` to select one of the six configured agents. No agent decides authorization, camera existence, or freshness; application code enforces those controls.

### Outer application workflow (before and after CrewAI)

1. **`initialize_state`:** create typed state with request id, user message, authenticated principal reference, conversation reference, bounded context, and empty guardrail/intent/result fields.
2. **`run_input_guardrails`:** inspect user input and relevant state/context, then write guardrail status and findings into state. A block flag does not bypass classification or the CrewAI Flow; it is carried forward so the selected agent can refuse safely and tools can deny retrieval.
3. **`classify_intent`:** make a normal structured-output LLM call outside CrewAI. Return only one enum label: `front_entrance`, `parking`, `back_entrance`, `common_lawn`, `greetings`, or `intent_clarify`. The classifier has no tools and does not answer or extract tool arguments.
4. **`validate_and_update_intent`:** validate that the label is in the allowlist and write it to `state.intent`. If malformed, set `intent_clarify` and record a classification validation error. This step does not select an agent.
5. **`run_crewai_flow`:** always pass the typed request state projection to CrewAI after classification and state update, regardless of which of the six intents was returned.
6. **`finalize_state`:** validate the returned result, update state with evidence and answer, persist conversation messages, refresh cache, and return the API response.

### CrewAI Flow (always starts after intent is classified)

1. **`load_state`:** receive request state, including `state.intent`, user message/context, and guardrail status.
2. **`switch_on_intent`:** explicit branches map all six intent values to agents: `front_entrance`, `parking`, `back_entrance`, `common_lawn`, `greetings`, and `intent_clarify`. Use a static registry; never derive Python class or tool names from free-form model output.
3. **`run_selected_agent`:** the selected agent follows its role instructions. If guardrails marked the request blocked, it returns a safe refusal and does not retrieve camera data.
4. **`camera_agent_tool_call` (camera intents only):** the camera agent reads the original message/context, determines the entity/property, fills typed tool arguments, and calls only its camera-scoped tool.
5. **`retrieve_observation` (tool/application service):** recheck access and entity/property, apply freshness policy, fetch camera data, and return tailored JSON.
6. **`answer_from_result`:** camera agents answer only from tool data; Greetings Agent responds without camera tools; Clarification Agent asks a concise follow-up without camera retrieval. All agents honor guardrail status.
7. **`return_result`:** return a typed agent result to the outer workflow for final validation and state update.

The request state is a small typed object, not a free-form transcript. Suggested fields: `request_id`, `principal_ref`, `conversation_ref`, `user_message`, `context_messages`, `guardrail_status`, `guardrail_findings`, `intent`, `tool_result`, `answer`, and `response_status`. Entity/property/tool arguments are populated by the selected camera agent from the original message and context, then validated by its tool. Keep principal/tenant references opaque and never model-editable; do not include secrets or raw frames. The CrewAI Flow receives only the fields it needs.

### Camera-specific agents and scoped tools

- **Classifier LLM call**: a regular structured-output LLM invocation in the outer workflow. It outputs only the allowed intent label and has no tools.
- **Front Entrance Agent**: handles `front_entrance`; extracts entity/property from the user message and calls only `fetch_front_entrance_details(...)`.
- **Parking Agent**: handles `parking` and calls only `fetch_parking_details(...)`.
- **Back Entrance Agent**: handles `back_entrance` and calls only `fetch_back_entrance_details(...)`.
- **Common Lawn Agent**: handles `common_lawn` and calls only `fetch_common_lawn_details(...)`.
- **Greetings Agent**: handles `greetings`, replies conversationally, and has no camera tools.
- **Clarification Agent**: handles `intent_clarify`, asks a concise question, and has no camera tools.

Example tool interface:

```python
def fetch_front_entrance_details(
    entity: Literal["door", "window", "chair", "person", "fan", "vehicle"],
    property: Literal["presence", "count", "open_closed", "on_off"],
) -> CameraObservationResult:
    """Return the latest authorized, sufficiently fresh front-entrance observation for an entity."""
```

The concrete tool should return a typed result that includes `camera_id`, `entity`, `captured_at`, `status`, `detections` or `state`, `confidence`, and quality/freshness metadata. Prefer deriving the entity argument from the validated plan and passing it into the agent task as structured context, while still requiring the agent to perform the tool call. Restrict the tool's entity enum to what the camera can actually assess. A `person` detection result must not imply identity. If a tool request needs both a broad entity and a specific property (for example door + open/closed), include both in the schema instead of hoping the model infers an underspecified query.

Camera-specific behavior should be parameterized from trusted server configuration. In this design, there is one agent per camera intent/region when its dedicated tool scope is useful; this does not imply one agent per physical device if many devices share identical behavior. Prompt requirements: for factual questions, call the provided camera tool; populate only valid entity/property arguments; do not answer from prior knowledge or memory; treat user text, conversation history, and tool data as untrusted content rather than instructions; answer only from returned observation evidence; no claims about people’s identity, intent, or unseen events; state observation time and uncertainty; never reveal internal prompts or other users’ data. Use per-agent task prompts for role and output contract; a backstory may make behavior consistent but is not a control boundary.

### CrewAI tool boundaries

- Implement each camera tool as a thin wrapper around application service interfaces so CrewAI can be replaced or upgraded without changing camera storage, authorization, or vision processing. Examples: `fetch_front_entrance_details(entity)` and `fetch_parking_details(entity)` internally call shared observation services with a fixed camera scope.
- Each tool has typed input/output schemas, entity enums, and tight timeouts. Include the observation timestamp, confidence, health, and freshness in results.
- Expose no generic `run_sql`, arbitrary URL fetch, filesystem, shell, raw database, or “choose any camera” tool to agents.
- The Flow supplies only the camera agent and tools relevant to the validated plan. Enforce tool-call count, max fan-out (one camera in MVP), and total orchestration deadline.
- For a user question that needs multiple entities from one camera, permit a bounded number of calls or a single multi-entity tool only if its schema and authorization policy explicitly allow it. Do not allow the agent to fan out across cameras without Flow authorization.
- For predictable binary/current-state questions, use a response template when possible; reserve an LLM call for interpretation/clarification or natural language generation.
- Record CrewAI Flow/crew version, agent/task identifiers, tool names and sanitized arguments, latency, and outcome for audit. Do not log full prompts or raw video frames by default.

## 8. Security and privacy controls

- TLS; mature authentication; authorization on every conversation and camera access; tenant isolation in every database/cache key.
- No client-controlled tenant/user identity. Bind conversation ids to the authenticated principal.
- Input size and content-type validation, Unicode normalization as appropriate, per-user rate limits, concurrency limits, model/tool timeouts, and cost ceilings.
- Prompt-injection defenses are layered risk reduction, not a guarantee: minimal context, no secrets in prompts, narrow tools, schema validation, deterministic authorization, output validation, and red-team cases. OWASP’s 2025 list also calls out insecure output handling, sensitive information disclosure, excessive agency, and unbounded consumption. [OWASP LLM Top 10](https://genai.owasp.org/llm-top-10/)
- Treat camera observations and OCR/scene text as untrusted input too; images or metadata can carry indirect instructions. The model should receive normalized facts, not raw camera text, whenever possible.
- Prevent cross-user memory leakage: namespace Redis keys by tenant, user, and conversation; enforce MongoDB filters centrally; test all IDOR paths.
- Redact credentials, tokens, raw frames, and sensitive prompt content from logs. Log request/trace ids, route, latency, camera id (where permitted), planner outcome, model/version, freshness, and error category.
- Set retention and deletion policies for conversation text and derived observations; support household/user deletion and key rotation.
- Keep all initial tools read-only. Add review/confirmation gates before any future actuation feature.
- Threat-model denial of service and model cost abuse, including huge prompts, repeated polling, and camera snapshot fan-out.

## 9. Execution plan

### Phase 0 — Requirements and threat model

**Deliverables:** supported question/property matrix; camera inventory/aliases; freshness and confidence policy; auth/tenant model; retention/privacy decisions; latency/cost targets; abuse cases.

**Exit criteria:** every MVP example maps to a known camera, evidence type, and an answer/clarification/unsupported behavior. Identify questions that need sensors rather than video.

### Phase 1 — Service foundation and mock vertical slice

**Deliverables:** application package structure; configuration and secrets loading; `POST /cctv-agent`; auth dependency (local dev identity first, real provider interface); typed outer request state and stateful application workflow; pre-classification input guardrails that update state; request/response schemas; health/readiness endpoints; structured logging and request ids; deterministic mock observation provider.

**Exit criteria:** all example questions run end-to-end using mock observations; camera access is enforced before mock tool invocation; no endpoint accepts arbitrary camera ids without authorization.

### Phase 2 — Query planning, clarification, and tool contracts

**Deliverables:** camera registry; direct structured-output LLM call that assigns only one of the six intents outside CrewAI; strict enum validation and state update; post-classification CrewAI Flow that always reads the intent from state and switches to the matching one of six agents; scoped camera tools backed by shared observation services; explicit response statuses; tool-result-grounded responses.

**Exit criteria:** ambiguous “my door” is routed to `intent_clarify`; a parking question is routed to `parking`; greetings are routed to `greetings`; every valid classifier output starts the CrewAI Flow; malformed intent output safely falls back to `intent_clarify`; stale observations are never represented as current.

### Phase 3 — Conversation memory and persistence

**Deliverables:** MongoDB conversation/message repository; Redis TTL cache with versioning and tenant-scoped keys; bounded context builder; durable-write/cache update ordering; retention and delete path; conversation id ownership checks.

**Exit criteria:** follow-ups resolve within a conversation; users cannot fetch another user’s history; cache misses/restarts recover from MongoDB; Redis failure degrades safely without losing acknowledged messages.

### Phase 4 — Live camera ingestion and vision normalization

**Deliverables:** independent stream/snapshot workers; camera health state; frame sampling policy; YOLO adapter converting library output into the observation schema; latest-state storage; timestamps, pipeline version, quality signals; bounded snapshot refresh path.

**Exit criteria:** an unavailable camera does not block other cameras or API workers; the API reports true capture time; stale and occluded scenes produce uncertainty; inference load is bounded and measured.

### Phase 5 — State estimation and domain quality

**Deliverables:** per-camera/region calibration; temporal smoothing/hysteresis; door/fan state estimators where video is appropriate; sensor fusion adapter for smart-home sensors; confidence thresholds and abstention behavior.

**Exit criteria:** curated validation set measures precision/recall per object/state and camera; false positives/negatives are reviewed; supported states have an explicit evidence source. Do not ship claims such as “fan is on” if the camera only detects a fan-shaped object.

### Phase 6 — Hardening, evaluation, and release

**Deliverables:** threat model review; adversarial prompt and cross-tenant test cases; planner routing/clarification golden set; vision quality report; load/cost tests; dashboards and alerts; backup/restore and retention jobs; deployment/runbook.

**Exit criteria:** routing, groundedness, freshness, authorization, latency, cost, and vision quality meet agreed thresholds; incidents can be audited without retaining unnecessary video; rollback path is documented.

## 10. Proposed complete project structure

This is the proposed target repository layout for the MVP and its first live-camera iteration. It is intentionally modular while remaining a single deployable Python application plus independently runnable vision workers. Files marked **MVP** are needed for the mocked end-to-end version; **Live** files are introduced when camera feeds are connected. The tree is a proposal for approval, not a claim that these files already exist.

```text
cctv_finder_poc/
├── README.md                         # Setup, local run, architecture overview, API examples
├── ARCHITECTURE_PLAN.md               # This architecture and execution plan
├── pyproject.toml                     # Package metadata, dependency pins/groups, lint/type config
├── requirements.txt                   # Optional exported runtime dependencies, if retained
├── .env.example                       # Variable names and safe local defaults; no secrets
├── .gitignore                         # Ignore venv, local secrets, caches, recordings, artifacts
├── Dockerfile                         # API container image
├── compose.yaml                       # Local API + Redis + Mongo + mock services
├── migrations/                        # Mongo index/bootstrap migrations or versioned setup scripts
│   └── README.md                      # Migration and index rollout procedure
├── configs/
│   ├── cameras.example.yaml           # Example camera/zone aliases and supported capabilities
│   └── vision.example.yaml            # Sampling, max-age, model and confidence settings
├── app/
│   ├── __init__.py
│   ├── main.py                        # FastAPI app factory, middleware and router registration
│   ├── dependencies.py                # Request-scoped service dependency wiring
│   ├── config.py                      # Typed settings and secret references
│   ├── errors.py                      # Stable application errors and API mapping
│   ├── workflow/
│   │   ├── __init__.py
│   │   ├── request_flow.py             # Outer stateful workflow: input -> guards -> classify -> CrewAI -> response
│   │   └── request_state.py            # Typed state shared across outer workflow stages
│   ├── api/
│   │   ├── __init__.py
│   │   ├── router.py                  # API version router
│   │   ├── schemas.py                 # Request/response Pydantic models
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── cctv_agent.py          # POST /cctv-agent
│   │       └── health.py              # Liveness/readiness endpoints
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── principal.py              # Authenticated identity and tenant types
│   │   ├── middleware.py             # Identity-provider/session integration
│   │   └── policy.py                 # Household, conversation and camera access checks
│   ├── cameras/
│   │   ├── __init__.py
│   │   ├── models.py                  # Camera, alias, capability and agent mapping types
│   │   ├── registry.py                # Server-trusted camera configuration lookup
│   │   └── resolver.py                # Zone alias/entity resolution and ambiguity handling
│   ├── planning/
│   │   ├── __init__.py
│   │   ├── models.py                  # IntentPlan and supported enum/schema definitions
│   │   ├── classifier.py              # Direct LLM intent-classification call; outside CrewAI
│   │   ├── validation.py              # Parse/validate classifier output; fail-closed behavior
│   │   └── policy.py                  # Supported query/property and confidence policy
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── flow.py                    # CrewAI Flow; begins after outer workflow validates intent
│   │   ├── state.py                   # Safe typed state projection passed into CrewAI Flow
│   │   ├── registry.py                # Static camera-intent to agent/tool mapping
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   ├── camera_base.py         # Shared camera-agent prompt/task construction
│   │   │   ├── front_entrance.py      # Front Entrance Agent and task instructions
│   │   │   ├── parking.py             # Parking Camera Agent and task instructions
│   │   │   ├── back_entrance.py       # Back Entrance Agent and task instructions
│   │   │   ├── common_lawn.py         # Common Lawn Agent and task instructions
│   │   │   ├── greetings.py           # Greetings Agent; no camera tools
│   │   │   └── clarification.py       # intent_clarify agent; no camera tools
│   │   └── tools/
│   │       ├── __init__.py
│   │       ├── front_entrance.py      # fetch_front_entrance_details(...)
│   │       ├── parking.py             # fetch_parking_details(...)
│   │       ├── back_entrance.py       # fetch_back_entrance_details(...)
│   │       └── common_lawn.py         # fetch_common_lawn_details(...)
│   ├── observations/
│   │   ├── __init__.py
│   │   ├── models.py                  # Versioned observation/tool result schemas
│   │   ├── service.py                 # Shared read/query interface for tools
│   │   ├── freshness.py               # Max-age, stale/unavailable and quality decisions
│   │   ├── repository.py              # Latest observation and event storage interface
│   │   └── providers/
│   │       ├── __init__.py
│   │       ├── mock.py                # Deterministic mock observation provider (MVP)
│   │       └── mongo.py               # Mongo-backed observation repository/provider
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── worker.py                  # Worker process entry point (Live)
│   │   ├── camera_reader.py           # RTSP/NVR/snapshot source adapter (Live)
│   │   ├── sampling.py                # Frame sampling and backpressure (Live)
│   │   ├── detector.py                # YOLO adapter and detection normalization (Live)
│   │   ├── state_estimators.py        # Door/fan or other supported state estimators (Live)
│   │   └── health.py                  # Feed health and camera status (Live)
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── models.py                  # Conversation/message records
│   │   ├── repository.py              # MongoDB durable conversation interface
│   │   ├── redis_cache.py             # Tenant/user/conversation-scoped TTL cache
│   │   ├── context_builder.py         # Relevant bounded conversation context
│   │   └── retention.py               # Expiration and user/household deletion jobs
│   ├── responses/
│   │   ├── __init__.py
│   │   ├── composer.py                # Grounded response task/template invocation
│   │   ├── validation.py              # Output/evidence association checks
│   │   └── localization.py            # Locale/tone handling and safe fallback text
│   ├── security/
│   │   ├── __init__.py
│   │   ├── input_validation.py        # Length, format and content validation
│   │   ├── input_guardrails.py        # Pre-classification checks; updates outer request state
│   │   ├── rate_limits.py             # Per-principal/IP limits and concurrency budget
│   │   ├── redaction.py               # Secret and sensitive-data log redaction
│   │   └── prompt_policy.py           # Optional risk classifier; never sole security layer
│   └── telemetry/
│       ├── __init__.py
│       ├── logging.py                 # Structured logs and correlation ids
│       ├── metrics.py                 # Routing, freshness, latency, model and vision metrics
│       └── tracing.py                 # API-to-Flow-to-tool spans
├── scripts/
│   ├── seed_mock_data.py              # Populate mock cameras/observations for local demo
│   └── validate_camera_config.py      # Check alias, capability and tool mappings
├── tests/
│   ├── conftest.py                    # Test fixtures and fake service wiring
│   ├── unit/
│   │   ├── test_intent_validation.py
│   │   ├── test_camera_resolution.py
│   │   ├── test_tool_scope.py
│   │   ├── test_freshness.py
│   │   └── test_response_grounding.py
│   ├── integration/
│   │   ├── test_ask_mock_flow.py
│   │   ├── test_conversation_isolation.py
│   │   └── test_camera_tool_observations.py
│   └── evals/
│       ├── intent_cases.yaml          # Golden routing/clarification examples
│       ├── grounding_cases.yaml       # Tool-result answer correctness cases
│       └── vision_cases/README.md     # Dataset and annotation policy; no private clips committed
└── deploy/
    ├── api.Dockerfile                 # Optional separate API image definition
    ├── worker.Dockerfile              # Vision worker image/runtime
    ├── k8s/                           # Optional deployment manifests, added when needed
    └── README.md                      # Environment-specific deployment and rollback notes
```

### Structure rules

- Camera agent modules should contain agent/task instructions and attach only their own tool. The tool module fixes the camera scope server-side and calls the shared `observations.service`; it must not duplicate camera ingestion or database logic.
- Intent classification is a regular structured-output LLM call in `planning/classifier.py`, invoked by the outer `workflow/request_flow.py` after guardrails and before CrewAI starts. It returns only one allowed intent label; `planning.validation` verifies the enum before the outer workflow writes it to `state.intent`.
- After the classifier writes intent to state, the CrewAI Flow always starts and switches among the six configured agents. Each camera agent determines the entity/property from the original message and fills its scoped tool arguments; greetings and clarification agents do not have camera tools.
- `workflow/request_state.py` holds the full request lifecycle state. `orchestration/state.py` is only the minimal validated projection passed into CrewAI after intent is resolved.
- `orchestration/registry.py` maps trusted camera ids/intents to known agent constructors. Never import a module or resolve a tool by concatenating model-generated text.
- The camera worker writes normalized observations. The API and CrewAI tools read them through `observations.service`; the API request does not own a stream-reading loop.
- Mock and live providers implement the same repository/service interfaces so the agent/tool path remains unchanged when replacing mock data with camera observations.
- Do not commit model weights, secrets, camera credentials, recordings, or private evaluation frames. Store large model weights in an approved artifact store or deployment volume.

### Current files and migration

The existing `object_detection_YOLO.py` is a local experiment, not yet an ingestion worker. Move/refactor its model invocation into `app/vision/detector.py` only after replacing the hard-coded desktop MP4 path with a configured source adapter. Keep `yolo26n.pt` out of normal source control if it is tracked as a large artifact; document how the worker obtains the approved model weights. Review `requirements.txt`: `YOLO` appears redundant or ambiguous alongside `ultralytics`, and should be checked before dependency expansion.

## 11. Evaluation and operational metrics

Measure the system at each boundary, not only whether a fluent answer was produced:

- **Routing:** correct zone/target/property; clarification precision/recall; unsupported-query rejection.
- **Grounding:** every factual answer traceable to observation ids and timestamps; unsupported claim rate.
- **Freshness:** age at answer time; stale answer rate; camera health and snapshot latency.
- **Vision:** class/state precision, recall, and confidence calibration by camera and lighting condition.
- **Security:** authorization denials; cross-tenant test results; prompt injection/tool misuse attempts; rate-limit events.
- **Reliability:** endpoint latency percentiles, inference queue depth, model/tool failures, Redis/Mongo availability.
- **Cost:** tokens per answered request, model spend per household, inference utilization.

Maintain a small, versioned golden set covering direct examples, paraphrases, follow-up references, ambiguous zones, multiple doors, camera offline, stale observations, low confidence, multilingual phrasing, malicious instructions, and requests for identity or actions. Use deterministic expected outcomes and review model outputs; do not rely solely on an LLM judge.

## 12. Immediate next steps

1. Agree on MVP question types, household/camera mapping, supported states, and freshness target.
2. Replace the hard-coded local-video script path with a configurable adapter boundary; keep the current detection demo isolated from request serving.
3. Build the mock-backed API vertical slice with a CrewAI Flow and observation schema; keep the Flow behind an orchestration interface so application services remain framework-independent.
4. Add authorization and tenant scoping before persistence or live camera integration.
5. Validate the confusing hardware claims early: car/object detection is a straightforward first target; door/fan “on/off” needs a state-specific source and evaluation data.

## References

- OWASP GenAI Security Project, [Prompt Injection](https://genai.owasp.org/glossaries/prompt-injection/)
- OWASP GenAI Security Project, [Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/)
