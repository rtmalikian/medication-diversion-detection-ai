"""open.epic MCP client tests — mock-only, no live network.

The fake MCP server is an httpx MockTransport speaking JSON-RPC 2.0,
covering: the initialize handshake + Mcp-Session-Id echo, tools/list
parsing, tools/call with JSON and SSE-framed responses, the no-PHI
guard, and graceful degradation when the server is unreachable.
"""
import json

import httpx
import pytest

import app_v2
from v2.epic_mcp import EpicMcpClient, PhiGuardError, screen_for_phi


@pytest.fixture
def anyio_backend():
    return "asyncio"

TOOLS = [
    {"name": "search_documents", "description": "Search Epic documentation",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}}},
    {"name": "get_document", "description": "Get one document by id",
     "inputSchema": {"type": "object", "properties": {"document_id": {"type": "string"}}}},
]

SEARCH_RESULT = [{"id": "api:11420", "title": "MedicationRequest.Search (R4)",
                  "content": "snippet", "url": "https://fhir.epic.com/Specifications?api=11420"}]


def make_transport(sse=False, record=None):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        if record is not None:
            record.append((body.get("method"), request.headers.get("mcp-session-id")))
        method = body.get("method")
        if method == "initialize":
            payload = {"jsonrpc": "2.0", "id": body["id"], "result": {
                "protocolVersion": "2025-03-26", "capabilities": {},
                "serverInfo": {"name": "open.epic MCP", "version": "4.0.9"}}}
            return httpx.Response(200, json=payload,
                                  headers={"Mcp-Session-Id": "sess-123"})
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "tools/list":
            payload = {"jsonrpc": "2.0", "id": body["id"], "result": {"tools": TOOLS}}
            return httpx.Response(200, json=payload)
        if method == "tools/call":
            payload = {"jsonrpc": "2.0", "id": body["id"], "result": {
                "isError": False,
                "content": [{"type": "text", "text": json.dumps(SEARCH_RESULT)}],
                "structuredContent": {"result": SEARCH_RESULT}}}
            if sse:
                return httpx.Response(
                    200, text=f"event: message\ndata: {json.dumps(payload)}\n\n",
                    headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json=payload)
        return httpx.Response(400, json={"error": "unknown method"})
    return httpx.MockTransport(handler)


def client_with(transport, **kw):
    return EpicMcpClient(base_url="https://mcp.example.test", transport=transport, **kw)


@pytest.mark.anyio
async def test_initialize_handshake_and_session_header_echo():
    record = []
    c = client_with(make_transport(record=record))
    out = await c.list_tools()
    assert out["available"] is True
    assert out["server_info"] == {"name": "open.epic MCP", "version": "4.0.9"}
    methods = [m for m, _ in record]
    assert methods[0] == "initialize"
    assert "notifications/initialized" in methods
    # the session id issued at initialize is echoed on later requests
    tools_call = [sid for m, sid in record if m == "tools/list"]
    assert tools_call == ["sess-123"]


@pytest.mark.anyio
async def test_tools_list_parsing():
    out = await client_with(make_transport()).list_tools()
    names = [t["name"] for t in out["tools"]]
    assert names == ["search_documents", "get_document"]
    assert out["tools"][0]["input_schema"]["properties"]["query"]


@pytest.mark.anyio
async def test_tools_call_json_response():
    out = await client_with(make_transport()).search_docs("MedicationRequest FHIR R4")
    assert out["available"] is True
    assert out["results"] == SEARCH_RESULT


@pytest.mark.anyio
async def test_tools_call_sse_framed_response():
    out = await client_with(make_transport(sse=True)).search_docs("MedicationDispense")
    assert out["available"] is True
    assert out["results"] == SEARCH_RESULT


@pytest.mark.anyio
async def test_get_document():
    out = await client_with(make_transport()).get_document("api:11420")
    assert out["available"] is True
    assert out["document_id"] == "api:11420"
    assert out["content_texts"]


# ------------------------------- no-PHI guard -------------------------------

@pytest.mark.parametrize("bad", [
    'score Patient/abc-123 please',
    '{"resourceType": "Patient", "id": "x"}',
    'patient ssn 123-45-6789 medication history',
    'MRN: 1234567 diversion risk',
    'DOB 1980-01-01 oxycodone',
    'what is the date of birth for this patient',
])
def test_phi_guard_patterns(bad):
    assert screen_for_phi(bad), bad


@pytest.mark.parametrize("ok", [
    "MedicationRequest FHIR R4 search parameters",
    "MedicationDispense.Read API specification",
    "Observation API Epic on FHIR documentation",
])
def test_phi_guard_allows_documentation_queries(ok):
    assert screen_for_phi(ok) == []


@pytest.mark.anyio
async def test_phi_guard_refuses_before_anything_is_sent():
    record = []
    c = client_with(make_transport(record=record))
    with pytest.raises(PhiGuardError):
        await c.search_docs("Patient/xyz-999 oxycodone history")
    assert record == []  # nothing reached the transport


# --------------------------- graceful degradation ---------------------------

@pytest.mark.anyio
async def test_unreachable_server_degrades():
    def boom(request):
        raise httpx.ConnectError("no route", request=request)
    c = client_with(httpx.MockTransport(boom))
    out = await c.list_tools()
    assert out["available"] is False and out["tools"] == []
    out = await c.search_docs("MedicationRequest")
    assert out["available"] is False and out["results"] == []


@pytest.mark.anyio
async def test_disabled_client_degrades():
    c = client_with(make_transport(), enabled=False)
    out = await c.search_docs("MedicationRequest")
    assert out["available"] is False


# ------------------------------- API endpoints ------------------------------

def test_epic_mcp_endpoints(monkeypatch, client):
    transport = make_transport()
    monkeypatch.setattr(app_v2, "epic_mcp_client",
                        lambda: client_with(transport))
    r = client.get("/epic-mcp/tools", headers={"X-API-Key": "demo-viewer"})
    assert r.status_code == 200
    assert [t["name"] for t in r.json()["tools"]] == ["search_documents", "get_document"]

    r = client.post("/epic-mcp/docs", headers={"X-API-Key": "demo-viewer"},
                    json={"query": "MedicationRequest FHIR R4"})
    assert r.status_code == 200
    assert r.json()["results"] == SEARCH_RESULT

    r = client.post("/epic-mcp/docs", headers={"X-API-Key": "demo-viewer"},
                    json={"query": "Patient/abc-123 chart"})
    assert r.status_code == 400  # no-PHI guard refusal surfaced, nothing sent

    r = client.get("/epic-mcp/tools")
    assert r.status_code == 401  # RBAC still applies
