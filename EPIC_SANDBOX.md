# Epic FHIR Sandbox — SMART Backend Services validation

**Status (2026-10-30): implemented and mock-tested; app registered
(JWK Set URL shows "Validated" in Epic's own check, app saved);
live token exchange attempted repeatedly — Epic returns
`invalid_client`, run not yet successful. UNRESOLVED.** Registration is DONE (details below, incl. the sandbox
client id and the hosted JWK Set URL). A live exchange was attempted the
same day with a correctly-formed RS384 assertion (structure and
signature verified locally against the public key; the hosted JWKS was
fetched and matches) and Epic's token endpoint answered
`HTTP 400 {"error": "invalid_client", "error_description": null}` —
Attempt log (all with correctly-formed RS384 assertions; scope
`system/Patient.read`, and the parent run also tried the full
six-resource scope set): 2026-10-30 ~19:2x UTC (2 attempts, assertion
lifetimes 300 s and 240 s), ~19:29 UTC (parent, 2 attempts, after the
JWK Set URL showed "Validated" and the app was saved), and
19:35:15 / 19:35:31 / 19:36:01 / 19:37:02 UTC (the validation script's
4 backoff retries) — every attempt answered
`HTTP 400 {"error": "invalid_client", "error_description": null}`.
Ruled out locally: assertion structure (iss=sub=client_id, aud=token
URL, exp ≤ 5 min, jti, kid header), RS384 signature (verifies against
the public key), hosted JWKS (fetched live; kid/n/e match the signing
key), client id, token URL. Remaining hypotheses, in order: (1)
sandbox provisioning / key-propagation delay after the app save;
(2) Epic-side fetching/caching of the hosted JWKS; (3) the sandbox
client's scope/API approvals not yet active. No cause is asserted —
retry the live run later; if it still fails, the next step is inside
the Epic app (re-check the JWK Set URL + client activation state) or
an Epic support/forum query with these timestamps. **No live Epic validation is claimed until
`scripts/validate_epic_sandbox.py` has run successfully and its dated
report exists under `docs/`.** Retry the live-run commands below once
the app shows the JWK Set URL saved.

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

## Registration (DONE 2026-10-30)

Registered at https://fhir.epic.com in Raphael's Epic developer account:

- **App name:** Medication Diversion Detection AI — **App ID:** 64194
- **App type:** Backend Systems, **FHIR R4**, **SMART v1**
- **Non-Production (sandbox) Client ID:**
  `f5dc34d0-2432-4297-9be8-69c9321dbc04` (a public identifier, not a
  secret; it is the validation script's default, overridable via
  `EPIC_CLIENT_ID`)
- **APIs selected (all Read + Search):** Patient, MedicationRequest,
  MedicationDispense, Encounter, Observation, Condition
- **Public key:** Epic does NOT accept pasted/uploaded keys for this —
  it requires a **hosted JWK Set URL**. The JWKS for our sandbox keypair
  (RSA-2048, `alg: RS384`, `use: sig`, `kid`
  `TgqxMccYLexwu37wIK5qoCr9vL1-Dz8aRHZRtFNs2n8` = RFC 7638 thumbprint)
  is hosted at **https://rtmalikian.github.io/epic/jwks.json**
  (verified live: HTTP 200, application/json). The private key lives
  only outside git (`.gitignore` blocks `*.pem`/`*.key`).
- A **Production** client id exists on the app but was **NOT
  activated** — do not use it or request production access.

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
