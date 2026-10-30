# Epic FHIR Sandbox — SMART Backend Services validation

**Status (2026-10-30): implemented and mock-tested; live run pending app
registration.** The SMART Backend Services auth module and the FHIR
client wiring are complete and covered by mock-only tests. The live
validation run against Epic's sandbox requires a registered sandbox app
(client id) — that registration is a one-time step in Raphael's Epic
developer account and has not happened yet. **No live Epic validation is
claimed until `scripts/validate_epic_sandbox.py` has run successfully
and its dated report exists under `docs/`.**

## What is implemented

- `v2/smart_backend.py` — SMART App Launch **Backend Services** flow:
  - JWT client assertion signed **RS384**: header `{alg: RS384, typ: JWT,
    kid}`; claims `iss = sub = client_id`, `aud = token URL`,
    `exp ≤ 5 minutes`, unique `jti`.
  - Token exchange: `grant_type=client_credentials`,
    `client_assertion_type=urn:ietf:params:oauth:client-assertion-type:jwt-bearer`,
    against `https://fhir.epic.com/interconnect-fhir-oauth/oauth2/token`.
  - Token caching with an expiry buffer (default 60 s).
- `v2/fhir_client.py` accepts an `auth` provider; `fetch_chart()` adds
  the bearer token to its headers, so the *same* client code that runs
  against mock/Synthea servers runs against Epic's sandbox FHIR base
  `https://fhir.epic.com/interconnect-fhir-oauth/api/FHIR/R4`.
- `scripts/validate_epic_sandbox.py` — token exchange → chart fetch for
  one sandbox synthetic patient → Layer A scoring → dated report under
  `docs/` (no tokens/keys in the report).
- Tests: `tests_v2/test_smart_backend.py` — assertion structure and
  RS384 signature verified with the public key; exchange + caching +
  refresh via a MockTransport fake token endpoint; failure surfacing.

## Scopes

Default requested scopes (configurable via `EPIC_SCOPES`):

```
system/Patient.read system/MedicationRequest.read system/MedicationDispense.read system/Encounter.read system/Observation.read system/Condition.read
```

These are exactly the resource types `v2/fhir_client.py` consumes. The
scope set that actually works is **whatever the registered sandbox app
is approved for** — requesting an unapproved scope fails the exchange.

## Registration steps (one-time, in the developer's Epic account)

1. Sign in at https://fhir.epic.com → Developer Apps → create a new app.
2. App type/audience: **Backend Systems** (system-level access, no user
   interaction), FHIR version **R4**, environment **Sandbox**
   (non-production, synthetic patients).
3. Request the scopes above.
4. Upload the app's public key as a **JWK** (Epic generates the keypair
   association from it; the matching private key stays with us and is
   never committed to this repo — `.gitignore` blocks `*.pem`/`*.key`).
   Current sandbox keypair (generated 2026-10-30, stored outside git):
   - Public key JWK `kid`: `TgqxMccYLexwu37wIK5qoCr9vL1-Dz8aRHZRtFNs2n8`
     (RFC 7638 thumbprint), RSA-2048, `alg: RS384`, `use: sig`.
5. Note the assigned **client id** — that is the only value the code
   needs from the registration, supplied via the `EPIC_CLIENT_ID`
   environment variable.

## Live run (once a client id exists)

```bash
export EPIC_CLIENT_ID=<client id from registration>
export EPIC_PRIVATE_KEY_PATH=<path to private_key.pem>   # outside git
export EPIC_PRIVATE_KEY_KID=TgqxMccYLexwu37wIK5qoCr9vL1-Dz8aRHZRtFNs2n8
python scripts/validate_epic_sandbox.py --patient <sandbox patient FHIR id>
```

Known Epic sandbox test patients are listed in Epic's documentation
once signed in; pick one with MedicationRequest/MedicationDispense data
so the chart mapping is meaningfully exercised.

## Honest limits

- Sandbox = synthetic patients only; a successful run validates
  *plumbing* (auth → FHIR → mapping → scoring), not clinical accuracy.
- Production Epic access is a different, per-health-system process
  (App Orchard / organization approval, BAA, security review) — nothing
  here claims it.
