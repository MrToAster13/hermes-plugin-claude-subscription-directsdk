"""ELI-350: /usage happy path for the Claude Subscription DirectSDK provider.

Single seam: Hermes's ``fetch_account_usage(provider)`` dispatch (agent.account_usage, which
resolves the registered plugin profile and calls its ``fetch_account_usage`` hook under a 10s
deadline) feeding ``render_account_usage_lines``. Only the HTTP response and the Claude Code
credentials read are faked; no live network.
"""
from datetime import datetime, timezone

import pytest

USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
PROVIDER_NAME = "claude-subscription-directsdk-experimental"

VALID_CREDS = {
    "accessToken": "cc-access-token",
    "refreshToken": "cc-refresh-token",
    "expiresAt": int(datetime.now(timezone.utc).timestamp() * 1000) + 3_600_000,
    "source": "claude_code_credentials_file",
}

LIMITS_PAYLOAD = {
    "limits": [
        {"kind": "session", "percent": 45, "resets_at": "2026-10-01T07:09:00Z", "is_active": True},
        {"kind": "weekly_all", "percent": 23, "resets_at": "2026-10-02T06:59:00Z", "is_active": True},
        {"kind": "weekly_model", "percent": 17, "resets_at": "2026-10-02T06:59:00Z",
         "scope": {"model": {"display_name": "Fable"}}, "is_active": True},
        {"kind": "something_new", "percent": None, "resets_at": None, "is_active": False},
    ],
    # Fields the happy path must never surface (ELI-351 fallback / excluded fields).
    "seven_day_breakdown": {"by_surface": {"cli": 1}},
    "spend": {"total_usd": 12.34},
    "extra_usage": {"is_enabled": True, "used_credits": 1.0, "monthly_limit": 10.0},
    "iguana_necktie": "codename",
}


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeClient:
    """Records every GET it receives; raises if used more than once per test."""

    instances = []

    def __init__(self, *, timeout=None):
        self.timeout = timeout
        self.requests = []
        _FakeClient.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get(self, url, headers=None):
        self.requests.append((url, headers))
        return _FakeResponse(LIMITS_PAYLOAD)


@pytest.fixture(autouse=True)
def _reset_fake_client():
    _FakeClient.instances = []
    yield
    _FakeClient.instances = []


def _never_called(name):
    def _raise(*a, **k):
        raise AssertionError(f"{name} must not be called by fetch_account_usage")
    return _raise


def _patch_transport(monkeypatch, *, creds=VALID_CREDS, client_cls=_FakeClient):
    import httpx
    import agent.anthropic_credentials as anthropic_credentials

    monkeypatch.setattr(anthropic_credentials, "read_claude_code_credentials", lambda: creds)
    # The hook must go through the read-only reader above and never refresh, write or resolve a token.
    monkeypatch.setattr(anthropic_credentials, "_refresh_oauth_token", _never_called("_refresh_oauth_token"))
    monkeypatch.setattr(anthropic_credentials, "_write_claude_code_credentials", _never_called("_write_claude_code_credentials"))
    monkeypatch.setattr(anthropic_credentials, "resolve_anthropic_token", _never_called("resolve_anthropic_token"))
    monkeypatch.setattr(httpx, "Client", client_cls)


def test_usage_snapshot_renders_one_line_per_limit_in_api_order(profile, monkeypatch):
    from agent.account_usage import fetch_account_usage, render_account_usage_lines

    _patch_transport(monkeypatch)
    snapshot = fetch_account_usage(PROVIDER_NAME)

    assert snapshot is not None
    assert snapshot.provider == "claude-subscription-directsdk-experimental"
    assert snapshot.source == "oauth_usage_api"
    assert snapshot.title == "Claude plan limits"
    assert snapshot.plan is None
    assert snapshot.details == ()

    labels = [w.label for w in snapshot.windows]
    assert labels == ["Current session", "Current week", "Fable week", "something_new"]

    used = [w.used_percent for w in snapshot.windows]
    assert used == [45.0, 23.0, 17.0, None]

    assert snapshot.windows[0].reset_at == datetime(2026, 10, 1, 7, 9, tzinfo=timezone.utc)
    assert snapshot.windows[3].reset_at is None

    lines = render_account_usage_lines(snapshot)
    assert lines[0] == "\U0001F4C8 Claude plan limits"
    assert lines[1] == "Provider: claude-subscription-directsdk-experimental"
    assert lines[2].startswith("Current session: 55% remaining (45% used) \u2022 resets ")
    assert lines[3].startswith("Current week: 77% remaining (23% used) \u2022 resets ")
    assert lines[4].startswith("Fable week: 83% remaining (17% used) \u2022 resets ")
    assert lines[5] == "something_new: unavailable"

    # No breakdown-by-surface, spend, extra-usage or codename fields leak into the rendered output.
    full_text = "\n".join(lines)
    for forbidden in ("seven_day_breakdown", "spend", "extra_usage", "iguana_necktie", "codename", "12.34"):
        assert forbidden not in full_text


def test_usage_request_shape_is_one_get_with_expected_headers_and_timeout(profile, monkeypatch):
    from agent.account_usage import fetch_account_usage

    _patch_transport(monkeypatch)
    fetch_account_usage(PROVIDER_NAME)

    assert len(_FakeClient.instances) == 1
    client = _FakeClient.instances[0]
    assert client.timeout == 8.0
    assert len(client.requests) == 1
    url, headers = client.requests[0]
    assert url == USAGE_URL
    assert headers["Authorization"] == "Bearer cc-access-token"
    assert headers["anthropic-beta"] == "oauth-2025-04-20"
    assert headers["Accept"] == "application/json"
    # Must match Hermes's built-in Anthropic OAuth usage fetcher's User-Agent exactly
    # (agent/account_usage.py), not merely be present.
    assert headers["User-Agent"] == "claude-code/2.1.0"


def test_usage_hook_never_refreshes_or_writes_credentials(profile, monkeypatch):
    """Scenario 4: the token-refresh path is never called and credentials are never written."""
    from agent.account_usage import fetch_account_usage

    _patch_transport(monkeypatch)
    fetch_account_usage(PROVIDER_NAME)  # would raise via _never_called if any forbidden path ran


def test_usage_hook_never_raises_on_a_malformed_response(profile, monkeypatch):
    """The hook must not raise on a malformed response (ticket scope note): a non-dict scope,
    a non-dict scope.model, or a top-level list body all degrade gracefully instead of dropping
    the whole /usage block (spec AC2/AC3 for the entries that ARE well-formed)."""
    from agent.account_usage import fetch_account_usage, render_account_usage_lines

    malformed_limits = {
        "limits": [
            {"kind": "session", "percent": 10, "resets_at": None},
            {"kind": "weird_scope", "percent": 20, "resets_at": None, "scope": "global"},
            {"kind": "weird_model", "percent": 30, "resets_at": None, "scope": {"model": "Fable"}},
        ],
    }

    class _MalformedClient(_FakeClient):
        def get(self, url, headers=None):
            self.requests.append((url, headers))
            return _FakeResponse(malformed_limits)

    _patch_transport(monkeypatch, client_cls=_MalformedClient)
    snapshot = fetch_account_usage(PROVIDER_NAME)

    assert snapshot is not None
    labels = [w.label for w in snapshot.windows]
    assert labels == ["Current session", "weird_scope", "weird_model"]
    lines = render_account_usage_lines(snapshot)
    assert lines  # rendered, did not drop the whole block


def test_usage_hook_never_raises_on_a_top_level_list_body(profile, monkeypatch):
    class _ListBodyClient(_FakeClient):
        def get(self, url, headers=None):
            self.requests.append((url, headers))
            return _FakeResponse([{"kind": "session", "percent": 1}])

    _patch_transport(monkeypatch, client_cls=_ListBodyClient)
    from agent.account_usage import fetch_account_usage

    snapshot = fetch_account_usage(PROVIDER_NAME)  # must not raise
    assert snapshot is None  # a list body has no `limits` key: defers like missing/empty limits
