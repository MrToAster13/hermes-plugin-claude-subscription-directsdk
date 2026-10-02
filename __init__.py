"""Claude Subscription DirectSDK (Experimental) — standalone Hermes model-provider registration."""
import logging
import os

from providers import register_provider
from providers.base import ProviderProfile

# Dual import: the Hermes loader imports this directory as a package; the flat test path does not.
try:
    from .model_catalog import ALIASES, MODEL_METADATA, native_model
    from .directsdk_setup import INSTALL_HINT, _resolve
except ImportError:
    from model_catalog import ALIASES, MODEL_METADATA, native_model
    from directsdk_setup import INSTALL_HINT, _resolve

logger = logging.getLogger(__name__)

# kind -> fixed label (ELI-349 Implementation Decisions). A model-scoped entry (scope.model.display_name
# set) overrides this with "<display_name> week"; anything else keeps the raw kind string.
_USAGE_KIND_LABELS = {'session': 'Current session', 'weekly_all': 'Current week'}


def _utc_now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)


def _parse_usage_reset_at(value):
    """ISO-8601 ``resets_at`` -> aware UTC datetime, or None (null/invalid)."""
    from datetime import datetime, timezone
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _usage_window(entry):
    """One ``limits[]`` entry -> an ``AccountUsageWindow`` (ELI-349 Implementation Decisions).

    Entries from the API are not fully trusted: a malformed ``scope`` or ``scope.model`` (e.g. a
    string instead of an object) must fall back to the raw kind label, never raise, per the
    ticket's "do not make the hook raise" instruction.

    H1 (ELI-351 hardening): ``kind`` can be a non-string, non-hashable value (a list or a dict),
    which would raise ``TypeError: unhashable type`` on a bare ``dict.get(kind)`` lookup. Only a
    string ``kind`` is used as the label-table key; anything else falls back to ``str(kind)``.
    H2 (ELI-351 hardening): ``percent`` can be NaN or Infinity (``json.loads`` accepts both); core's
    ``render_account_usage_lines`` calls ``round()`` on ``used_percent`` and raises on a non-finite
    float, so a non-finite percent must degrade to ``None`` (renders as "unavailable") here.
    """
    import math
    from agent.account_usage import AccountUsageWindow
    kind = entry.get('kind')
    kind_label = kind if isinstance(kind, str) else None
    scope = entry.get('scope')
    scope = scope if isinstance(scope, dict) else {}
    model = scope.get('model')
    model = model if isinstance(model, dict) else {}
    model_name = model.get('display_name')
    model_name = model_name if isinstance(model_name, str) and model_name else None
    label = _USAGE_KIND_LABELS.get(kind_label) if kind_label else None
    if label is None:
        if model_name:
            label = f'{model_name} week'
        elif kind_label:
            label = kind_label
        elif kind is not None:
            label = str(kind)
        else:
            label = 'unknown'
    percent = entry.get('percent')
    is_number = isinstance(percent, (int, float)) and not isinstance(percent, bool)
    used_percent = float(percent) if is_number and math.isfinite(percent) else None
    return AccountUsageWindow(label=label, used_percent=used_percent, reset_at=_parse_usage_reset_at(entry.get('resets_at')))


def _sanitize_legacy_resets_at(value):
    """Drop a legacy ``resets_at`` value that core's ``_parse_dt`` would raise on, or silently
    mis-render (ELI-402 review round 2 finding; ELI-352 S3 advisory finding (b)): a list/dict is
    unhashable and raises ``TypeError`` from ``_parse_dt``'s own ``value in {None, ""}`` check; a
    non-finite or out-of-range numeric value (NaN, 1e20, epoch-milliseconds such as 1790913290000
    which is year 58721) raises ``ValueError`` or ``OverflowError`` from
    ``datetime.fromtimestamp``; a bool (``True``/``False``) is a legacy-field type error too — it
    is not a raise (``bool`` is an ``int`` subclass, so ``_parse_dt`` happily parses it as epoch
    0 or 1) but a silently wrong reset time, so it is dropped the same way. Anything else
    (``None``, a string, a normal epoch number) is passed through unchanged for core to
    parse/validate itself.
    """
    from datetime import datetime, timezone
    if isinstance(value, (list, dict, bool)):
        return None
    if isinstance(value, (int, float)):
        try:
            datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            return None
    return value


def _sanitize_legacy_source(payload, keys):
    """Pre-validate the legacy ``five_hour``/``seven_day``/... fields before handing them to
    core's ``_usage_windows`` (ELI-351 review round 1 finding 2): that helper assumes
    ``source[key]`` is a dict and ``source[key][used_key]`` is float-able, and raises
    (``AttributeError``/``ValueError``/``TypeError``) on a malformed shape, which the dispatch
    then masks to ``None`` and the whole /usage block disappears. A malformed window (wrong type,
    or a non-numeric ``utilization``) is dropped here, same as a genuinely absent window: the
    other, well-formed windows still render. ELI-402: a malformed ``resets_at`` (see
    ``_sanitize_legacy_resets_at``) is dropped the same way, but only the reset time — the
    window's ``utilization`` and the rest of the block survive.
    """
    cleaned = {}
    for key in keys:
        window = payload.get(key)
        if not isinstance(window, dict):
            continue
        used = window.get('utilization')
        if used is not None and (isinstance(used, bool) or not isinstance(used, (int, float))):
            window = {k: v for k, v in window.items() if k != 'utilization'}
        if 'resets_at' in window and _sanitize_legacy_resets_at(window.get('resets_at')) is None:
            window = {k: v for k, v in window.items() if k != 'resets_at'}
        cleaned[key] = window
    return cleaned


def _finite_or_none(window):
    """A NaN/Infinity ``used_percent`` (ELI-351 review round 1 finding 1 — the H2 bug class,
    reopened in the legacy fallback path) must degrade to ``None`` (renders as ``unavailable``),
    same as H2 does for ``limits[].percent``: core's ``_usage_windows`` floats the raw value with
    no isfinite check, and render_account_usage_lines's round() raises on a non-finite float.
    """
    import math
    from dataclasses import replace
    if window.used_percent is not None and not math.isfinite(window.used_percent):
        return replace(window, used_percent=None)
    return window


class ClaudeOAuthDirectSDKProfile(ProviderProfile):
    model_metadata = MODEL_METADATA

    def get_model_context_length(self, model):
        route = native_model(model)
        pinned = self.model_metadata.get(route, {}).get('context_window')
        # Unpinned: behind the relay native Claude Code runs a plain id within its 200K default, and
        # Hermes' own family guess (claude-opus-5-5 -> 1M before it was pinned) would outgrow that.
        # A [1m] id stays unreported: no Hermes guess exceeds the 1M native budget, nor is it promised.
        return pinned or (None if route.endswith('[1m]') else 200_000)

    def get_usage_cost(self, model, usage):
        from decimal import Decimal, InvalidOperation
        from agent.usage_pricing import CostResult, format_cost_label

        native = (usage.raw_usage or {}).get('native_cost') or {}
        unknown = CostResult(amount_usd=None, status='unknown', source='none', label='n/a',
                             notes=('native final list-price accounting unavailable; subscription invoice unknown',))
        amount = native.get('total_cost_usd')
        models = native.get('modelUsage') or {}
        if isinstance(amount, bool) or not models or any(row.get('costBasis') != 'list' for row in models.values()):
            return unknown
        try:
            amount = Decimal(str(amount))
        except InvalidOperation:
            return unknown
        if not amount.is_finite() or amount < 0:
            return unknown
        return CostResult(amount_usd=amount, status='estimated', source='provider_cost_api',
                          label=format_cost_label(amount), notes=('native API list-price equivalent; not subscription invoice; extra usage unknown',))

    def create_client(self, **client_kwargs):
        try:
            from .directsdk import Client
        except ImportError:
            from directsdk import Client
        return Client(**client_kwargs)

    def fetch_models(self, **kwargs):
        # No HTTP /models endpoint: the account's own picker (CLI `initialize` handshake) is the
        # live list for /model, the Desktop picker and `hermes model`; None degrades to the catalog.
        rows = self.discover_models(**kwargs)
        return [row["id"] for row in rows] if rows else None

    def setup_status(self, **kwargs):
        try:
            from .directsdk_setup import setup_status
        except ImportError:
            from directsdk_setup import setup_status
        return setup_status(**kwargs)

    def discover_models(self, **kwargs):
        try:
            from .directsdk_setup import discover_models
        except ImportError:
            from directsdk_setup import discover_models
        return discover_models(**kwargs)

    def build_api_kwargs_extras(self, *, reasoning_config=None, **_):
        return ({'reasoning': dict(reasoning_config)} if reasoning_config else {}), {}

    def _usage_unavailable(self, reason):
        from agent.account_usage import AccountUsageSnapshot
        return AccountUsageSnapshot(
            provider=self.name, source='oauth_usage_api', fetched_at=_utc_now(),
            title='Claude plan limits', unavailable_reason=reason,
        )

    def fetch_account_usage(self, *, base_url=None, api_key=None):
        # Read-only: the Claude Code credential reader below never refreshes or writes, and this
        # hook must never call resolve_anthropic_token (that path can refresh/rotate the token).
        # ELI-351: on any failure this returns a snapshot with an unavailable_reason (never None,
        # never raises) so /usage prints "Unavailable: <reason>" and the rest of the block still
        # renders. The exact reason strings are ELI-349's (Implementation Decisions, Errors).
        import math
        import httpx
        from agent.account_usage import AccountUsageSnapshot, _is_num, _usage_windows
        from agent.anthropic_credentials import is_claude_code_token_valid, read_claude_code_credentials

        creds = read_claude_code_credentials()
        if not creds or not creds.get('accessToken'):
            return self._usage_unavailable('no Claude Code login found (run `claude` and log in)')
        # expiresAt within 60s (is_claude_code_token_valid's own buffer): no HTTP request at all.
        # A non-numeric expiresAt makes core's own check raise (`'str' - int`, ELI-351 review
        # round 1 finding 2); treat that the same as an expired token rather than let it raise.
        try:
            token_valid = is_claude_code_token_valid(creds)
        except (TypeError, ValueError):
            token_valid = False
        if not token_valid:
            return self._usage_unavailable('token expired (run `claude` once to refresh)')
        token = creds['accessToken']
        headers = {
            'Authorization': f'Bearer {token}',
            'anthropic-beta': 'oauth-2025-04-20',
            'Accept': 'application/json',
            # Mirrors Hermes's built-in Anthropic OAuth usage fetcher (agent/account_usage.py).
            'User-Agent': 'claude-code/2.1.0',
        }
        try:
            with httpx.Client(timeout=8.0) as client:
                response = client.get('https://api.anthropic.com/api/oauth/usage', headers=headers)
        except httpx.HTTPError:
            return self._usage_unavailable('could not reach the usage API')
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in (401, 403):
                return self._usage_unavailable('token rejected (run `claude` once to refresh)')
            return self._usage_unavailable(f'usage API returned HTTP {status}')
        try:
            payload = response.json()
        except ValueError:
            return self._usage_unavailable('usage API returned an unreadable response')

        payload = payload if isinstance(payload, dict) else {}
        limits = payload.get('limits')
        details = []
        if isinstance(limits, list) and limits:
            windows = tuple(_usage_window(entry) for entry in limits if isinstance(entry, dict))
        else:
            # Fallback (ELI-351 AC4): no second request, no call into Hermes's built-in Anthropic
            # fetcher (it refreshes tokens) — parse the SAME body's legacy fields instead, reusing
            # core's own window-building helper on the already-fetched payload. The extra-usage
            # details line is part of the legacy-field fallback only (ELI-349); the `limits` happy
            # path (ELI-350) never surfaces it.
            mapping = (('five_hour', 'Current session'), ('seven_day', 'Current week'),
                       ('seven_day_opus', 'Opus week'), ('seven_day_sonnet', 'Sonnet week'))
            # ELI-351 review round 1 findings 1 & 2: sanitize malformed window shapes/types before
            # core's _usage_windows (which assumes a dict with a float-able utilization), then
            # degrade any non-finite percent that survives float() to None (same as H2).
            cleaned = _sanitize_legacy_source(payload, [key for key, _label in mapping])
            windows = tuple(_finite_or_none(w) for w in _usage_windows(
                cleaned, mapping, 'utilization', 'resets_at', fraction=True,
            ))
            extra = payload.get('extra_usage')
            extra = extra if isinstance(extra, dict) else {}
            used_credits, monthly_limit = extra.get('used_credits'), extra.get('monthly_limit')
            # ELI-402: _is_num accepts NaN/Infinity (they are floats); isfinite() keeps a NaN or
            # infinite value from rendering as "Extra usage: nan" (same bug class as H2/finding 1).
            if (extra.get('is_enabled') and _is_num(used_credits) and _is_num(monthly_limit)
                    and math.isfinite(used_credits) and math.isfinite(monthly_limit)):
                details.append(f'Extra usage: {used_credits:.2f} / {monthly_limit:.2f} {extra.get("currency") or "USD"}')
        return AccountUsageSnapshot(
            provider=self.name, source='oauth_usage_api', fetched_at=_utc_now(),
            title='Claude plan limits', windows=windows, details=tuple(details),
        )


profile = ClaudeOAuthDirectSDKProfile(
    name='claude-subscription-directsdk-experimental',
    display_name='Claude Subscription DirectSDK (Experimental)',
    description='Claude Subscription DirectSDK (Experimental) (Claude Pro/Max subscription via your Claude Code login; Hermes owns tools)',
    api_mode='chat_completions',
    auth_type='external_process',
    supports_health_check=False,
    native_reasoning_details_type='claude-subscription-directsdk-experimental.native_assistant',
    env_vars=(),
    base_url='process://claude-subscription-directsdk-experimental',
    process_command='claude',
    process_args=(),
    process_command_env_vars=('CLAUDE_SUBSCRIPTION_DIRECTSDK_COMMAND',),
    default_aux_model='claude-sonnet-5[1m]',
    fallback_models=tuple(MODEL_METADATA),
    model_aliases={alias: native_model(alias) for alias in ALIASES},
)
register_provider(profile)

# The provider stays registered when Claude Code is missing so `hermes model` can show the
# install hint; the request path (`directsdk.Client`) refuses with the same message.
if _resolve(None, os.environ) is None:
    logger.warning("%s: %s", profile.display_name, INSTALL_HINT)
