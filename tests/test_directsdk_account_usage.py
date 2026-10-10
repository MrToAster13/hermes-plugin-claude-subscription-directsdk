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
    """H3 (ELI-351): the old assertion (``snapshot is None``) passed with or without a fix because
    the dispatch's ``except Exception: return None`` masks any crash the same way. ELI-351 gives a
    non-dict body a real, observable path instead: it degrades to an empty parsed body, which the
    new fallback renders as a present-but-empty snapshot (not None). Verified red without the
    ELI-351 fallback: on the ELI-350 code (pre-fallback) a missing/malformed ``limits`` key
    returned ``None`` unconditionally, so this assertion failed before the fallback was added.
    """
    class _ListBodyClient(_FakeClient):
        def get(self, url, headers=None):
            self.requests.append((url, headers))
            return _FakeResponse([{"kind": "session", "percent": 1}])

    _patch_transport(monkeypatch, client_cls=_ListBodyClient)
    from agent.account_usage import fetch_account_usage, render_account_usage_lines

    snapshot = fetch_account_usage(PROVIDER_NAME)  # must not raise
    assert snapshot is not None
    assert snapshot.windows == ()
    assert snapshot.details == ()
    assert snapshot.unavailable_reason is None
    lines = render_account_usage_lines(snapshot)  # must not raise either
    assert lines == ["\U0001F4C8 Claude plan limits", f"Provider: {PROVIDER_NAME}"]


# ELI-351 scenario 2 / AC 4: `limits` missing or empty -> fallback to the legacy fields of the
# SAME response body, one request only.
FALLBACK_PAYLOAD = {
    "five_hour": {"utilization": 0.42, "resets_at": "2026-10-01T07:09:00Z"},
    "seven_day": {"utilization": 0.23, "resets_at": "2026-10-02T06:59:00Z"},
    "seven_day_opus": {"utilization": 0.05, "resets_at": "2026-10-02T06:59:00Z"},
    "extra_usage": {"is_enabled": True, "used_credits": 1.5, "monthly_limit": 10.0, "currency": "USD"},
}


@pytest.mark.parametrize("limits_value", [None, [], "missing"])
def test_usage_hook_fallback_parses_legacy_fields_with_one_request(profile, monkeypatch, limits_value):
    from agent.account_usage import fetch_account_usage, render_account_usage_lines

    body = dict(FALLBACK_PAYLOAD)
    if limits_value != "missing":
        body["limits"] = limits_value

    class _FallbackClient(_FakeClient):
        def get(self, url, headers=None):
            self.requests.append((url, headers))
            return _FakeResponse(body)

    _patch_transport(monkeypatch, client_cls=_FallbackClient)
    snapshot = fetch_account_usage(PROVIDER_NAME)

    assert snapshot is not None
    assert len(_FakeClient.instances) == 1
    assert len(_FakeClient.instances[0].requests) == 1  # exactly one HTTP request, no second fetch

    labels = [w.label for w in snapshot.windows]
    assert labels == ["Current session", "Current week", "Opus week"]
    used = [w.used_percent for w in snapshot.windows]
    assert used == [42.0, 23.0, 5.0]

    lines = render_account_usage_lines(snapshot)
    assert any(line.startswith("Current session: 58% remaining (42% used)") for line in lines)
    assert any("Extra usage: 1.50 / 10.00 USD" == line for line in lines)


# ELI-351 scenario 3 / AC 5, AC 6: exact `Unavailable: <reason>` lines, no refresh, no HTTP for
# the no-login and expired-token cases.
_REASON_NO_LOGIN = "no Claude Code login found (run `claude` and log in)"
_REASON_TOKEN_EXPIRED = "token expired (run `claude` once to refresh)"
_REASON_TOKEN_REJECTED = "token rejected (run `claude` once to refresh)"
_REASON_UNREADABLE = "usage API returned an unreadable response"
_REASON_NETWORK = "could not reach the usage API"


def _assert_unavailable_line(snapshot, reason):
    from agent.account_usage import render_account_usage_lines

    assert snapshot is not None
    assert snapshot.windows == ()
    assert snapshot.unavailable_reason == reason
    lines = render_account_usage_lines(snapshot)
    assert lines.count(f"Unavailable: {reason}") == 1
    # the rest of /usage still prints: header + provider line are present alongside it.
    assert lines[0] == "\U0001F4C8 Claude plan limits"
    assert lines[-1] == f"Unavailable: {reason}"


def test_usage_hook_error_no_login(profile, monkeypatch):
    from agent.account_usage import fetch_account_usage

    _patch_transport(monkeypatch, creds=None)
    snapshot = fetch_account_usage(PROVIDER_NAME)
    _assert_unavailable_line(snapshot, _REASON_NO_LOGIN)
    assert _FakeClient.instances == []  # no HTTP request without a login


def test_usage_hook_error_expired_token_makes_no_request(profile, monkeypatch):
    from agent.account_usage import fetch_account_usage

    expired_creds = {
        **VALID_CREDS,
        "expiresAt": int(datetime.now(timezone.utc).timestamp() * 1000) + 30_000,  # within the 60s buffer
    }
    _patch_transport(monkeypatch, creds=expired_creds)
    snapshot = fetch_account_usage(PROVIDER_NAME)
    _assert_unavailable_line(snapshot, _REASON_TOKEN_EXPIRED)
    assert _FakeClient.instances == []  # expired -> no HTTP request (AC 6)


def _http_status_error(status):
    import httpx
    request = httpx.Request("GET", USAGE_URL)
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


@pytest.mark.parametrize("status", [401, 403])
def test_usage_hook_error_http_401_or_403(profile, monkeypatch, status):
    from agent.account_usage import fetch_account_usage

    class _RejectedResponse(_FakeResponse):
        def raise_for_status(self):
            raise _http_status_error(status)

    class _RejectedClient(_FakeClient):
        def get(self, url, headers=None):
            self.requests.append((url, headers))
            return _RejectedResponse(None)

    _patch_transport(monkeypatch, client_cls=_RejectedClient)
    snapshot = fetch_account_usage(PROVIDER_NAME)
    _assert_unavailable_line(snapshot, _REASON_TOKEN_REJECTED)


def test_usage_hook_error_http_500(profile, monkeypatch):
    from agent.account_usage import fetch_account_usage

    class _ServerErrorResponse(_FakeResponse):
        def raise_for_status(self):
            raise _http_status_error(500)

    class _ServerErrorClient(_FakeClient):
        def get(self, url, headers=None):
            self.requests.append((url, headers))
            return _ServerErrorResponse(None)

    _patch_transport(monkeypatch, client_cls=_ServerErrorClient)
    snapshot = fetch_account_usage(PROVIDER_NAME)
    _assert_unavailable_line(snapshot, "usage API returned HTTP 500")


def test_usage_hook_error_timeout(profile, monkeypatch):
    from agent.account_usage import fetch_account_usage

    class _TimeoutClient(_FakeClient):
        def get(self, url, headers=None):
            import httpx
            raise httpx.TimeoutException("timed out")

    _patch_transport(monkeypatch, client_cls=_TimeoutClient)
    snapshot = fetch_account_usage(PROVIDER_NAME)
    _assert_unavailable_line(snapshot, _REASON_NETWORK)


def test_usage_hook_error_unreadable_body(profile, monkeypatch):
    from agent.account_usage import fetch_account_usage

    class _UnreadableResponse(_FakeResponse):
        def json(self):
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

    class _UnreadableClient(_FakeClient):
        def get(self, url, headers=None):
            self.requests.append((url, headers))
            return _UnreadableResponse(None)

    _patch_transport(monkeypatch, client_cls=_UnreadableClient)
    snapshot = fetch_account_usage(PROVIDER_NAME)
    _assert_unavailable_line(snapshot, _REASON_UNREADABLE)


# H1 (ELI-350 review hardening): a non-string, non-hashable `kind` must not raise; valid entries
# in the same response must still render. Verified red without the fix: pre-fix, `_usage_window`
# called `_USAGE_KIND_LABELS.get(kind)` with the raw (unhashable) `kind`, raising
# `TypeError: cannot use 'list' as a dict key`, which escaped render (not the dispatch try) in the
# case driven at the bottom of this test (exercised here through fetch_account_usage/render,
# the single seam, never by calling the private `_usage_window` helper directly).
@pytest.mark.parametrize("bad_kind", [["x"], {"a": 1}])
def test_usage_hook_handles_non_string_kind_without_raising(profile, monkeypatch, bad_kind):
    from agent.account_usage import fetch_account_usage, render_account_usage_lines

    body = {"limits": [
        {"kind": "session", "percent": 1, "resets_at": None},
        {"kind": bad_kind, "percent": 2, "resets_at": None},
    ]}

    class _BadKindClient(_FakeClient):
        def get(self, url, headers=None):
            self.requests.append((url, headers))
            return _FakeResponse(body)

    _patch_transport(monkeypatch, client_cls=_BadKindClient)
    snapshot = fetch_account_usage(PROVIDER_NAME)  # must not raise

    assert snapshot is not None
    labels = [w.label for w in snapshot.windows]
    assert labels[0] == "Current session"
    assert labels[1] == str(bad_kind)  # coerced, not dropped

    lines = render_account_usage_lines(snapshot)  # must not raise
    assert any(line.startswith("Current session:") for line in lines)


# H2 (ELI-350 review hardening): a NaN/Infinity `percent` must not reach core's
# render_account_usage_lines (which calls round() and would raise outside the dispatch try); it
# must render as the standard `unavailable` line, same as a null percent. Verified red without
# the fix: pre-fix, `_usage_window` accepted any int/float (including NaN/inf) as `used_percent`,
# and `render_account_usage_lines`'s `round(100 - used)` on a NaN/inf value raises
# (`ValueError`/`OverflowError`) outside fetch_account_usage's own try/except.
@pytest.mark.parametrize("bad_percent", [float("nan"), float("inf"), float("-inf")])
def test_usage_hook_handles_non_finite_percent_without_raising(profile, monkeypatch, bad_percent):
    from agent.account_usage import fetch_account_usage, render_account_usage_lines

    body = {"limits": [
        {"kind": "session", "percent": 10, "resets_at": None},
        {"kind": "weekly_all", "percent": bad_percent, "resets_at": None},
    ]}

    class _BadPercentClient(_FakeClient):
        def get(self, url, headers=None):
            self.requests.append((url, headers))
            return _FakeResponse(body)

    _patch_transport(monkeypatch, client_cls=_BadPercentClient)
    snapshot = fetch_account_usage(PROVIDER_NAME)  # must not raise

    assert snapshot is not None
    assert snapshot.windows[0].used_percent == 10.0
    assert snapshot.windows[1].used_percent is None

    lines = render_account_usage_lines(snapshot)  # must not raise (no round() on NaN/inf)
    assert lines[2] == "Current session: 90% remaining (10% used)"


# Review round 1 findings (ELI-351): the H2 bug class and the "never raises" guarantee were only
# applied to the `limits[]` path, not the legacy fallback path added by this same ticket.


@pytest.mark.parametrize("bad_utilization", [float("nan"), float("inf"), float("-inf")])
def test_usage_hook_fallback_handles_non_finite_utilization_without_raising(profile, monkeypatch, bad_utilization):
    """Same bug class as H2, but in the fallback (legacy-field) branch: core's `_usage_windows`
    does `float(used)` with no isfinite check, so a NaN/Infinity `utilization` reaches
    render_account_usage_lines's round() and raises outside the dispatch try. Verified red
    without the fix: pre-fix this raised ValueError/OverflowError from render_account_usage_lines.
    """
    from agent.account_usage import fetch_account_usage, render_account_usage_lines

    body = {
        "five_hour": {"utilization": bad_utilization, "resets_at": None},
        "seven_day": {"utilization": 0.2, "resets_at": None},
    }

    class _BadUtilizationClient(_FakeClient):
        def get(self, url, headers=None):
            self.requests.append((url, headers))
            return _FakeResponse(body)

    _patch_transport(monkeypatch, client_cls=_BadUtilizationClient)
    snapshot = fetch_account_usage(PROVIDER_NAME)  # must not raise

    assert snapshot is not None
    labels = [w.label for w in snapshot.windows]
    assert labels == ["Current session", "Current week"]
    used = [w.used_percent for w in snapshot.windows]
    assert used == [None, 20.0]

    lines = render_account_usage_lines(snapshot)  # must not raise (no round() on NaN/inf)
    assert lines[2] == "Current session: unavailable"
    assert lines[3] == "Current week: 80% remaining (20% used)"


@pytest.mark.parametrize("bad_window", ["oops", [1, 2]])
def test_usage_hook_fallback_handles_non_dict_window_without_raising(profile, monkeypatch, bad_window):
    """The hook must not raise when a legacy window's value isn't a dict (e.g. `five_hour` is a
    string or a list). Verified red without the fix: pre-fix `_usage_windows`'s `window.get(...)`
    raised `AttributeError` ('str'/'list' object has no attribute 'get').
    """
    from agent.account_usage import fetch_account_usage, render_account_usage_lines

    body = {"five_hour": bad_window, "seven_day": {"utilization": 0.3, "resets_at": None}}

    class _BadWindowClient(_FakeClient):
        def get(self, url, headers=None):
            self.requests.append((url, headers))
            return _FakeResponse(body)

    _patch_transport(monkeypatch, client_cls=_BadWindowClient)
    snapshot = fetch_account_usage(PROVIDER_NAME)  # must not raise

    assert snapshot is not None
    labels = [w.label for w in snapshot.windows]
    assert labels == ["Current week"]  # the malformed "five_hour" window is dropped, not fabricated
    assert snapshot.windows[0].used_percent == 30.0

    lines = render_account_usage_lines(snapshot)  # must not raise
    assert any(line.startswith("Current week:") for line in lines)


@pytest.mark.parametrize("bad_utilization", ["abc", [1]])
def test_usage_hook_fallback_handles_non_numeric_utilization_without_raising(profile, monkeypatch, bad_utilization):
    """The hook must not raise when `utilization` itself is a non-numeric type (a string that
    isn't float-able, or a list). Verified red without the fix: pre-fix `float(used)` raised
    `ValueError`/`TypeError`.
    """
    from agent.account_usage import fetch_account_usage, render_account_usage_lines

    body = {
        "five_hour": {"utilization": bad_utilization, "resets_at": None},
        "seven_day": {"utilization": 0.3, "resets_at": None},
    }

    class _BadNumericClient(_FakeClient):
        def get(self, url, headers=None):
            self.requests.append((url, headers))
            return _FakeResponse(body)

    _patch_transport(monkeypatch, client_cls=_BadNumericClient)
    snapshot = fetch_account_usage(PROVIDER_NAME)  # must not raise

    assert snapshot is not None
    labels = [w.label for w in snapshot.windows]
    assert labels == ["Current week"]  # the malformed "five_hour" window is dropped, not fabricated
    assert snapshot.windows[0].used_percent == 30.0

    lines = render_account_usage_lines(snapshot)  # must not raise
    assert any(line.startswith("Current week:") for line in lines)


# ELI-402: a non-string legacy `resets_at` reaches core's `_parse_dt`, which raises
# (ValueError for epoch-ms, NaN; OverflowError for huge floats; TypeError for unhashable
# list/dict). The dispatch's `except Exception: return None` masked it and the whole /usage
# block disappeared. A malformed reset time must drop only that reset time, keeping the
# window's percentage and the rest of the block. Verified red on 07e3fe0 (pre-fix): each of
# these raised from inside `agent.account_usage._parse_dt`/`_usage_windows`, escaping the
# plugin's own try/except in `_usage_windows`'s caller and hitting the dispatch's masking
# `except Exception: return None`, so `snapshot is None` on 07e3fe0 and this test fails there.
@pytest.mark.parametrize("bad_resets_at", [1790913290000, 1e20, float("nan"), ["x"], {"a": 1}, True, False])
def test_usage_hook_fallback_handles_malformed_resets_at_without_raising(profile, monkeypatch, bad_resets_at):
    from agent.account_usage import fetch_account_usage, render_account_usage_lines

    body = {
        "five_hour": {"utilization": 0.1, "resets_at": bad_resets_at},
        "seven_day": {"utilization": 0.2, "resets_at": "2026-10-02T06:59:00Z"},
    }

    class _BadResetsAtClient(_FakeClient):
        def get(self, url, headers=None):
            self.requests.append((url, headers))
            return _FakeResponse(body)

    _patch_transport(monkeypatch, client_cls=_BadResetsAtClient)
    snapshot = fetch_account_usage(PROVIDER_NAME)  # must not raise

    assert snapshot is not None
    labels = [w.label for w in snapshot.windows]
    assert labels == ["Current session", "Current week"]  # block stays, both windows present
    used = [w.used_percent for w in snapshot.windows]
    assert used == [10.0, 20.0]  # percentage kept even though the reset time is dropped
    assert snapshot.windows[0].reset_at is None  # only the malformed reset time is dropped
    assert snapshot.windows[1].reset_at is not None  # the well-formed window's reset time survives

    lines = render_account_usage_lines(snapshot)  # must not raise
    assert lines[2] == "Current session: 90% remaining (10% used)"
    assert lines[3].startswith("Current week: 80% remaining (20% used) \u2022 resets ")


# ELI-402: a NaN/Infinity extra-usage value must render no "Extra usage" line instead of
# "Extra usage: nan" / "Extra usage: inf". Verified red on 07e3fe0 (pre-fix): `_is_num` accepts
# NaN/Infinity (they are floats), so the f-string formatted them and "Extra usage: nan" appeared.
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("bad_field", ["used_credits", "monthly_limit"])
def test_usage_hook_fallback_hides_extra_usage_line_on_non_finite_value(profile, monkeypatch, bad_field, bad_value):
    from agent.account_usage import fetch_account_usage, render_account_usage_lines

    extra_usage = {"is_enabled": True, "used_credits": 1.5, "monthly_limit": 10.0, "currency": "USD"}
    extra_usage[bad_field] = bad_value
    body = {"five_hour": {"utilization": 0.1, "resets_at": None}, "extra_usage": extra_usage}

    class _BadExtraUsageClient(_FakeClient):
        def get(self, url, headers=None):
            self.requests.append((url, headers))
            return _FakeResponse(body)

    _patch_transport(monkeypatch, client_cls=_BadExtraUsageClient)
    snapshot = fetch_account_usage(PROVIDER_NAME)  # must not raise

    assert snapshot is not None
    assert not any(d.startswith("Extra usage") for d in snapshot.details)

    lines = render_account_usage_lines(snapshot)  # must not raise
    assert not any(line.startswith("Extra usage") for line in lines)


def test_usage_hook_error_non_numeric_expires_at_treated_as_expired(profile, monkeypatch):
    """core's `is_claude_code_token_valid` does `expires_at - 60_000`, which raises `TypeError`
    on a non-numeric `expiresAt`. The hook must guard this and treat it as an expired token (no
    HTTP request, no refresh, no write), not raise. Verified red without the fix: pre-fix this
    raised TypeError from is_claude_code_token_valid, masked to None by the dispatch.
    """
    from agent.account_usage import fetch_account_usage

    bad_creds = {**VALID_CREDS, "expiresAt": "soon"}
    _patch_transport(monkeypatch, creds=bad_creds)
    snapshot = fetch_account_usage(PROVIDER_NAME)
    _assert_unavailable_line(snapshot, _REASON_TOKEN_EXPIRED)
    assert _FakeClient.instances == []  # malformed expiresAt -> no HTTP request
