"""SMART Backend Services auth tests — mock-only, no live network.

Generates a throwaway RSA keypair in-test, builds assertions, verifies
them with the public key, and exercises the token exchange against an
httpx MockTransport fake Epic token endpoint. No real client id, key,
or token appears anywhere in these tests.
"""
import base64
import json
import time
import urllib.parse

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from v2.smart_backend import (
    CLIENT_ASSERTION_TYPE,
    EPIC_SANDBOX_TOKEN_URL,
    SmartAuthError,
    SmartBackendAuth,
    build_client_assertion,
)


@pytest.fixture(scope="module")
def keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv = key.private_bytes(serialization.Encoding.PEM,
                             serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())
    return key, priv


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _decode(segment: str) -> dict:
    pad = "=" * (-len(segment) % 4)
    return json.loads(base64.urlsafe_b64decode(segment + pad))


def test_assertion_structure_and_signature(keypair):
    key, priv = keypair
    now = 1_700_000_000
    jwt = build_client_assertion("client-123", EPIC_SANDBOX_TOKEN_URL, priv,
                                 kid="kid-1", now=now)
    head_b64, claims_b64, sig_b64 = jwt.split(".")
    header, claims = _decode(head_b64), _decode(claims_b64)
    assert header == {"alg": "RS384", "typ": "JWT", "kid": "kid-1"}
    assert claims["iss"] == claims["sub"] == "client-123"
    assert claims["aud"] == EPIC_SANDBOX_TOKEN_URL
    assert claims["exp"] - now <= 300 and claims["exp"] > now
    assert claims["jti"]
    # signature verifies with the public key, RS384
    pad = "=" * (-len(sig_b64) % 4)
    sig = base64.urlsafe_b64decode(sig_b64 + pad)
    key.public_key().verify(sig, f"{head_b64}.{claims_b64}".encode("ascii"),
                            padding.PKCS1v15(), hashes.SHA384())


def test_assertion_lifetime_capped(keypair):
    _, priv = keypair
    with pytest.raises(ValueError):
        build_client_assertion("c", EPIC_SANDBOX_TOKEN_URL, priv, lifetime=600)


def make_token_transport(record=None, expires_in=300):
    def handler(request: httpx.Request) -> httpx.Response:
        form = dict(urllib.parse.parse_qsl(request.content.decode()))
        if record is not None:
            record.append(form)
        assert form["grant_type"] == "client_credentials"
        assert form["client_assertion_type"] == CLIENT_ASSERTION_TYPE
        assert form["client_assertion"].count(".") == 2
        assert "system/Patient.read" in form["scope"]
        return httpx.Response(200, json={"access_token": "fake-token-1",
                                         "token_type": "bearer",
                                         "expires_in": expires_in,
                                         "scope": form["scope"]})
    return httpx.MockTransport(handler)


@pytest.mark.anyio
async def test_token_exchange_and_caching(keypair):
    _, priv = keypair
    record = []
    auth = SmartBackendAuth(client_id="client-123", private_key_pem=priv,
                            transport=make_token_transport(record))
    t1 = await auth.get_token()
    t2 = await auth.get_token()
    assert t1 == t2 == "fake-token-1"
    assert len(record) == 1  # second call served from cache
    headers = await auth.auth_headers()
    assert headers["Authorization"] == "Bearer fake-token-1"


@pytest.mark.anyio
async def test_token_refresh_after_expiry(keypair):
    _, priv = keypair
    record = []
    auth = SmartBackendAuth(client_id="client-123", private_key_pem=priv,
                            transport=make_token_transport(record, expires_in=30),
                            expiry_buffer=60.0)
    await auth.get_token()
    await auth.get_token()  # within buffer window -> refresh
    assert len(record) == 2


@pytest.mark.anyio
async def test_token_exchange_failure_surfaces(keypair):
    _, priv = keypair

    def handler(request):
        return httpx.Response(401, json={"error": "invalid_client"})

    auth = SmartBackendAuth(client_id="bad", private_key_pem=priv,
                            transport=httpx.MockTransport(handler))
    with pytest.raises(SmartAuthError):
        await auth.get_token()


@pytest.mark.anyio
async def test_unconfigured_raises_clear_error():
    auth = SmartBackendAuth(client_id="", private_key_pem=None)
    auth._private_key_pem = None
    with pytest.raises(SmartAuthError):
        await auth.get_token()


def test_default_scopes_cover_fhir_client_resources():
    auth = SmartBackendAuth(client_id="x", private_key_pem=b"")
    for res in ("Patient", "MedicationRequest", "MedicationDispense",
                "Encounter", "Observation", "Condition"):
        assert f"system/{res}.read" in auth.scopes
