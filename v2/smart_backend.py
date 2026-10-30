"""SMART Backend Services auth for Epic's FHIR sandbox (and production).

Implements the SMART App Launch "Backend Services" authorization flow
(system-level, no user interaction) exactly as Epic documents for
fhir.epic.com apps:

1. Build a JWT client assertion:
   - header: {"alg": "RS384", "typ": "JWT", "kid": <key id>}
   - claims: iss = sub = the app's client_id, aud = the authorization
     server's token URL, exp = now + <=5 minutes, jti = random UUID.
   - sign with the app's RSA private key using RS384
     (RSASSA-PKCS1-v1_5 with SHA-384).
2. POST to the token URL with:
   - grant_type=client_credentials
   - client_assertion_type=urn:ietf:params:oauth:client-assertion-type:jwt-bearer
   - client_assertion=<signed JWT>
   - scope=<space-separated system/*.read scopes>
3. Cache the returned access token and reuse it until shortly before
   expiry (configurable buffer), then mint a fresh assertion.

Epic sandbox endpoints (non-production, synthetic patients only):
- FHIR base: https://fhir.epic.com/interconnect-fhir-oauth/api/FHIR/R4
- Token URL: https://fhir.epic.com/interconnect-fhir-oauth/oauth2/token

Configuration (environment):
- EPIC_CLIENT_ID — the registered app's client id (required for a live
  token exchange; there is deliberately no default and no credential is
  ever committed to this repo).
- EPIC_PRIVATE_KEY_PATH — path to the RSA private key PEM whose public
  key (JWK) was uploaded at app registration. The private key must
  NEVER be committed to a repo.
- EPIC_PRIVATE_KEY_KID — the kid assigned to the uploaded JWK.
- EPIC_TOKEN_URL / EPIC_FHIR_BASE_URL — endpoint overrides.
- EPIC_SCOPES — space-separated scopes. Default: system/Patient.read system/MedicationRequest.read system/MedicationDispense.read system/Encounter.read system/Observation.read system/Condition.read.
  The exact scope set that works is whatever the registered sandbox
  app is approved for; requesting unapproved scopes fails the exchange.

Requires the ``cryptography`` package (declared in requirements).
Token exchange failures raise SmartAuthError; nothing here logs the
assertion, the private key, or the access token.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid

import httpx

log = logging.getLogger(__name__)

EPIC_SANDBOX_FHIR_BASE = "https://fhir.epic.com/interconnect-fhir-oauth/api/FHIR/R4"
EPIC_SANDBOX_TOKEN_URL = "https://fhir.epic.com/interconnect-fhir-oauth/oauth2/token"
CLIENT_ASSERTION_TYPE = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
DEFAULT_SCOPES = "system/Patient.read system/MedicationRequest.read system/MedicationDispense.read system/Encounter.read system/Observation.read system/Condition.read"
_ASSERTION_LIFETIME = 300  # seconds; Epic requires exp <= 5 minutes out


class SmartAuthError(Exception):
    """SMART Backend Services token exchange failed."""


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def build_client_assertion(client_id: str, token_url: str, private_key_pem: bytes,
                           kid: str | None = None, now: int | None = None,
                           lifetime: int = _ASSERTION_LIFETIME) -> str:
    """Build + RS384-sign the JWT client assertion. Pure function of inputs."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    if lifetime > 300:
        raise ValueError("assertion lifetime must be <= 300 seconds (Epic limit)")
    now = int(time.time()) if now is None else now
    header = {"alg": "RS384", "typ": "JWT"}
    if kid:
        header["kid"] = kid
    claims = {"iss": client_id, "sub": client_id, "aud": token_url,
              "exp": now + lifetime, "iat": now, "jti": str(uuid.uuid4())}
    signing_input = ".".join([
        _b64url(json.dumps(header, separators=(",", ":")).encode()),
        _b64url(json.dumps(claims, separators=(",", ":")).encode()),
    ])
    key = serialization.load_pem_private_key(private_key_pem, password=None)
    signature = key.sign(signing_input.encode("ascii"), padding.PKCS1v15(),
                         hashes.SHA384())
    return f"{signing_input}.{_b64url(signature)}"


class SmartBackendAuth:
    """Token provider: exchanges assertions, caches the access token."""

    def __init__(self, client_id: str | None = None,
                 private_key_pem: bytes | None = None,
                 private_key_path: str | None = None,
                 kid: str | None = None,
                 token_url: str | None = None,
                 scopes: str | None = None,
                 transport: httpx.AsyncBaseTransport | None = None,
                 expiry_buffer: float = 60.0, timeout: float = 30.0):
        self.client_id = client_id or os.environ.get("EPIC_CLIENT_ID", "")
        self.token_url = (token_url or os.environ.get("EPIC_TOKEN_URL")
                          or EPIC_SANDBOX_TOKEN_URL)
        self.scopes = scopes or os.environ.get("EPIC_SCOPES") or DEFAULT_SCOPES
        self.kid = kid or os.environ.get("EPIC_PRIVATE_KEY_KID") or None
        self.transport = transport
        self.expiry_buffer = expiry_buffer
        self.timeout = timeout
        if private_key_pem is None:
            path = private_key_path or os.environ.get("EPIC_PRIVATE_KEY_PATH", "")
            if path:
                with open(path, "rb") as fh:
                    private_key_pem = fh.read()
        self._private_key_pem = private_key_pem
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self._private_key_pem)

    async def get_token(self, force_refresh: bool = False) -> str:
        """Return a valid access token, exchanging a new one if needed."""
        now = time.time()
        if (not force_refresh and self._token
                and now < self._token_expires_at - self.expiry_buffer):
            return self._token
        if not self.configured:
            raise SmartAuthError(
                "SMART Backend Services is not configured: set EPIC_CLIENT_ID "
                "and EPIC_PRIVATE_KEY_PATH (see EPIC_SANDBOX.md).")
        assertion = build_client_assertion(
            self.client_id, self.token_url, self._private_key_pem, kid=self.kid)
        form = {"grant_type": "client_credentials",
                "client_assertion_type": CLIENT_ASSERTION_TYPE,
                "client_assertion": assertion,
                "scope": self.scopes}
        try:
            async with httpx.AsyncClient(transport=self.transport,
                                         timeout=self.timeout) as client:
                resp = await client.post(
                    self.token_url, data=form,
                    headers={"Content-Type": "application/x-www-form-urlencoded"})
        except httpx.HTTPError as exc:
            raise SmartAuthError(f"token endpoint unreachable: {exc}") from exc
        if resp.status_code != 200:
            # Error bodies from the token endpoint do not contain secrets;
            # still, only the status + error code are surfaced.
            try:
                err = resp.json().get("error", "")
            except Exception:
                err = ""
            raise SmartAuthError(
                f"token exchange failed: HTTP {resp.status_code} {err}".strip())
        body = resp.json()
        token = body.get("access_token")
        if not token:
            raise SmartAuthError("token response contained no access_token")
        self._token = token
        self._token_expires_at = now + float(body.get("expires_in", 300))
        log.info("epic smart backend: token acquired (expires_in=%s, scope=%s)",
                 body.get("expires_in"), body.get("scope", self.scopes))
        return token

    async def auth_headers(self) -> dict:
        """Headers dict carrying the bearer token, for the FHIR client."""
        return {"Authorization": f"Bearer {await self.get_token()}",
                "Accept": "application/fhir+json"}
