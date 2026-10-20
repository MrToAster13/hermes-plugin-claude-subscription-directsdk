"""Pinned native routes; a loopback gateway needs explicit long-context selection."""
CONTEXT_WINDOWS = {
    'claude-sonnet-5': 1_000_000,
    'claude-haiku-5-5': 1_000_000,
    'claude-haiku-4-5-20251001': 200_000,
    'claude-opus-5-5': 1_000_000,
    'claude-opus-5': 1_000_000,
    'claude-opus-4-8': 1_000_000,
    'claude-fable-5-1': 1_000_000,
}
# Families that cannot turn thinking off. Fable 400s on ``thinking: {"type": "disabled"}`` (the
# same contract Hermes core keeps in agent/anthropic_adapter.py). The Claude Code model-config
# docs say Haiku 5.5 cannot turn thinking off either. A 2026-10-20 probe did not reject the
# disable, but it did not record its CLI version, and title calls that day recorded CLI 2.1.282
# (ELI-564 postpatch_title.log), below the documented 2.1.293 minimum for Haiku 5.5, so it
# is listed here. A spurious entry only leaves thinking on; a missing one risks a 400 on every
# call that sends the disable, which Hermes title generation always does.
# A caller's disable is omitted for these families: thinking stays on at the model's default.
MANDATORY_THINKING = ('claude-fable', 'claude-haiku-5-5')
ALIASES = {
    'sonnet': 'claude-sonnet-5',
    'haiku': 'claude-haiku-5-5',
    'claude-haiku-4-5': 'claude-haiku-4-5-20251001',
    'opus': 'claude-opus-5-5',
    'fable': 'claude-fable-5-1',
}
# Haiku 4.5 rejects `thinking: {'type': 'adaptive'}` with a 400 upstream; its effort signal
# still applies. Unknown routes keep adaptive so future models are not silently downgraded.
NO_ADAPTIVE_THINKING = frozenset({'claude-haiku-4-5-20251001'})


def native_model(model):
    base = model.removesuffix('[1m]')
    canonical = ALIASES.get(base, base)
    window = CONTEXT_WINDOWS.get(canonical)
    if window == 1_000_000:
        return canonical + '[1m]'
    if window == 200_000:
        if model.endswith('[1m]'):
            raise ValueError('Haiku 4.5 does not support a 1M context window')
        return canonical
    return model


def accepts_thinking_disable(model):
    # Same tolerance as supports_adaptive_thinking: an absent model still reaches `model is required`.
    base = model.removesuffix('[1m]') if isinstance(model, str) else ''
    return not ALIASES.get(base, base).startswith(MANDATORY_THINKING)


def supports_adaptive_thinking(model):
    # Body assembly precedes model validation, so an absent or invalid route answers True
    # and still reaches the existing `model is required` error.
    base = model.removesuffix('[1m]') if isinstance(model, str) else ''
    return ALIASES.get(base, base) not in NO_ADAPTIVE_THINKING


MODEL_METADATA = {
    native_model(model): {'canonical_model': model, 'context_window': window}
    for model, window in CONTEXT_WINDOWS.items()
}
