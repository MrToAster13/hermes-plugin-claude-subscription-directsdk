## Problem

`/usage` shows no Claude plan limits when the provider is this plugin
(`claude-subscription-directsdk-experimental`). Hermes core has its own built-in Anthropic
usage fetcher, but it only covers overall account usage — it misses the per-model limits
Anthropic's OAuth usage API reports, such as the separate weekly limit for Fable. Users of
this plugin get no limits block at all in `/usage`, in the CLI, the TUI or on Telegram.

This PR adds a `fetch_account_usage` hook to the plugin's provider profile. It calls
`GET https://api.anthropic.com/api/oauth/usage` once per `/usage` invocation and renders a
`📈 Claude plan limits` block through Hermes core's existing single render seam
(`agent/account_usage.py`), so the block appears on every surface that already calls that
seam (CLI, TUI, gateway/Telegram) with no core changes.

## Read-only credential decision

The hook never refreshes or writes Claude Code credentials. It only reads the existing OAuth
access token from disk. If the token is expired, `/usage` reports "token expired (run `claude`
once to refresh)" and makes no HTTP request at all — it does not attempt to refresh the token
itself.

This is deliberate: Claude's OAuth refresh tokens are single-use. If this plugin refreshed the
token on its own, it would race Claude Code's own refresh cycle — whichever process refreshes
second gets a token Claude Code (or this plugin) has already invalidated, breaking the user's
native `claude` login. Reading only, and deferring entirely to `claude`'s own refresh, avoids
that failure mode.

## What's covered

- One line per `limits` entry from the API, in API order, labelled `Current session`,
  `Current week`, and `<model> week` (so Fable's weekly limit shows as `Fable week`); unknown
  `limits` kinds fall back to their raw kind string.
- When `limits` is missing or empty, the block falls back to the legacy fields in the same
  response, with no second request.
- Exact `Unavailable: <reason>` lines for every error case: no login, expired token, HTTP
  401/403, other non-2xx, timeout/network, and unreadable JSON. The rest of `/usage` keeps
  rendering around any of these.
- An 8-second timeout on the request; the hook never raises.
- No breakdown-by-surface, spend, plan name, or codename fields in the output.

## Tests

New tests in `tests/test_directsdk_account_usage.py` cover `fetch_account_usage` plus the
render seam for: the happy path (ordered labels, percentages, reset times), an unknown
`limits` kind, the legacy-fallback path, every `Unavailable: <reason>` error case (no login,
expired, 401, 403, other non-2xx, timeout, bad JSON), the expired-token case making no HTTP
request, the 8-second timeout, the forbidden-fields check, and malformed legacy `resets_at`
values (including a bare `True`/`False`, which previously silently rendered as epoch 0/1
instead of being dropped).

Full plugin suite at this branch's head: **81 passed, 1 failed**. The 1 failure is
`tests/test_directsdk_replay.py::test_host_history_edits_replay_canonical_visible_blocks`
(`AttributeError: ContextCompressor has no attribute _truncate_tool_call_args_at`), which is a
pre-existing Hermes-core API drift unrelated to this plugin — it fails the same way on
`fleet/hermes-infra` before this branch's changes.

## Scope

This diff touches only this plugin repo (provider code, tests, README). Hermes core is
unchanged.
