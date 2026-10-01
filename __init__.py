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
    """
    from agent.account_usage import AccountUsageWindow
    kind = entry.get('kind')
    scope = entry.get('scope')
    scope = scope if isinstance(scope, dict) else {}
    model = scope.get('model')
    model = model if isinstance(model, dict) else {}
    model_name = model.get('display_name')
    model_name = model_name if isinstance(model_name, str) and model_name else None
    label = _USAGE_KIND_LABELS.get(kind) or (f'{model_name} week' if model_name else kind) or 'unknown'
    percent = entry.get('percent')
    used_percent = float(percent) if isinstance(percent, (int, float)) and not isinstance(percent, bool) else None
    return AccountUsageWindow(label=label, used_percent=used_percent, reset_at=_parse_usage_reset_at(entry.get('resets_at')))


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

    def fetch_account_usage(self, *, base_url=None, api_key=None):
        # Read-only: the Claude Code credential reader below never refreshes or writes, and this
        # hook must never call resolve_anthropic_token (that path can refresh/rotate the token).
        import httpx
        from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
        from agent.anthropic_credentials import is_claude_code_token_valid, read_claude_code_credentials

        creds = read_claude_code_credentials()
        if not creds or not is_claude_code_token_valid(creds):
            return None
        token = creds.get('accessToken')
        if not token:
            return None
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
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return None
        payload = payload if isinstance(payload, dict) else {}
        limits = payload.get('limits')
        if not isinstance(limits, list) or not limits:
            return None  # empty/missing limits: ELI-351 fallback scope, not this ticket
        windows = tuple(_usage_window(entry) for entry in limits if isinstance(entry, dict))
        return AccountUsageSnapshot(
            provider=self.name, source='oauth_usage_api', fetched_at=_utc_now(),
            title='Claude plan limits', windows=windows,
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
