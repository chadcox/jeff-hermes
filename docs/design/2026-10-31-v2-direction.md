# jeff_router v2: direction after the shadow run

Strategy plan with go/no-go gates. Each phase gets its own task-level plan once the previous gate passes.

## What the shadow run showed (840 ok records, 2026-10-31 16:01 to 2026-11-01 00:04 UTC)

- Plumbing is fine: 99.6% ok, Jeff p50 358 ms, p95 606 ms.
- The generic `tools` adapter is not usable as a router on Hermes traffic. Tool agreement with Strata is 12.6%
  overall and 25.6% at tool.p ≥ 0.8, while always guessing `terminal` scores 53.6%. Full `read_file` calls at
  t ≥ 0.8 match Strata 0 times out of 41.
- Little traffic could go to a step menu at all: of 756 Strata tool calls, 10.8% used a read-only tool and 2.4%
  exactly repeated an earlier call in the same session. Most calls are one-off `ssh` / `pct exec` / `python3 -c`
  commands.

## What Jeff-Code does differently (github.com/firelex/jeff-code)

- It uses adapters trained on the same agent and model it runs with (Pi + Qwen3.8-27B). We used a generic adapter
  with no training on Hermes or Strata.
- Code builds a small menu of concrete next steps (with arguments), and Jeff only picks from it. Jeff never chose
  among 25 tool names.
- Jeff takes only safe information steps; edits stay with the big model.
- Most of the speed-up comes from the second adapter, `code-router`, which turns thinking off on about 3/4 of turns.
  Pass rate stayed within noise (62.4% vs 62.8%, 1,242 paired tasks). Their README reports 0.68× time per task.

## Why the thinking router is the better bet here

- Strata runs with thinking on: `agent.reasoning_effort: high`, with `xhigh` overrides for `ternary-bonsai-2-27b`
  and the `Qwen3.8-27B-*` models (`~/.hermes/config.yaml`). `qwen3.8-flash-next` is overridden to `none`, so the
  router does nothing for it.
- `code-router` is already in `models/adapters/` and already served by jeff-serve (`/health` lists it, 4 options:
  off / low / medium / xhigh).
- The middleware receives the full provider kwargs before `next_call`, so it can rewrite the effort per call without
  patching Hermes (`docs/research/01-hook-api.md`). Effort is sent as top-level `reasoning_effort` or `extra_body.reasoning`,
  depending on the provider profile (`agent/transports/chat_completions.py`, `agent/auxiliary_reasoning_floor.py`).
- Caveats: `code-router` was trained on Qwen3.8-27B coding turns in Pi, which suits neither our models nor our
  admin traffic. Its offline accuracy was only 68% even in-domain. The card says to serve it on a Q8_0 base;
  ours is the HF bf16 checkpoint on GPU, which is fine.

## Phase 0: measure what thinking costs (no behaviour change)

1. Confirm the exact field Strata's profile uses for effort, and that `usage` reports reasoning tokens.
   Verify with one captured request and response from a probe.
2. Extend each shadow record with Strata's wall time (`strata_ms`, timed around `next_call`), the effort sent,
   `usage` (prompt, completion, reasoning tokens) and `finish_reason`. Log schema becomes `v: 2`; the report
   accepts both versions.
3. Run for 1–2 days of normal use.

**Gate 0:** reasoning time is a large share of Strata's time on the models we use. Rough bar: on thinking models,
the median turn spends ≥ 30% of its completion tokens on reasoning. Otherwise stop: no router can win much.

## Phase 1: shadow the thinking router

1. On calls to thinking models, also ask `code-router` (separate request, same pool) and log P(off / low / medium /
   xhigh). Leave the request unchanged.
2. Report: the share of turns the router would set to off at thresholds 0.5–0.8, and the reasoning tokens and
   `strata_ms` those turns actually used (the time a router could save).

**Gate 1:** at P(xhigh) < 0.6 the router would turn thinking off on ≥ 30% of thinking-model turns, and those turns
cost ≥ 15% of total Strata time. Below that the saving is too small to be worth the quality risk.

## Phase 2: label our own turns offline (Jeff-Code's labelling recipe)

1. Capture full provider requests for a sample of thinking-model turns (owner-only, size-capped, separate file,
   off by default). The current log keeps only a summary, which is not enough to replay.
2. Offline replay script: re-send each captured request with effort off, then low, then medium (`max_tokens`
   capped, no tool execution) and compare the first action with the original. Code judges equivalence: same text
   vs tool, same tool name, normalised arguments. Leave out turns whose action involves a model judge, as in
   v1's undecidable records. The label is the cheapest effort whose action matched, or xhigh.
3. Score `code-router` against those labels: how often it says off when off would have changed the action.
   That figure is the quality risk.

Cost: about 3 extra Strata calls per sampled turn. Start with 200 turns.

**Gate 2:** among turns where the router says off at the chosen threshold, ≥ 90% kept the same action with thinking
off. If `code-router` misses that but the labels show many turns don't need thinking, go to Phase 2b.

### Phase 2b (only if needed): train a Hermes router adapter

- Rows from the Phase 2 labels in `jeff-kit` row format; family = session (or cron job), so the test set holds out
  whole sessions.
- Run `jeff-kit check-rows`, `split`, `leak-check`, `shortcut-report`, then `jeff-train` and `jeff-evaluate`
  (all in `vendor/jeff`).
- Needs far more than 200 labels, so likely weeks of capture. Re-check Gate 2 on the held-out split.

## Phase 3: live, opt-in

1. Setting `mode: route_thinking` (default `shadow`). Lower effort only on thinking models and only when the router
   is confident. Never raise effort. Fall back to the configured effort on any Jeff error or timeout (≤ 0.6 s
   budget, so it never adds a full Jeff round trip).
2. Log the effort chosen in every record so the report can compare quality and time by routed vs not routed.
3. Run interactively for a week, reviewing sessions where routed turns were followed by retries or user
   corrections.

**Gate 3:** no noticeable quality drop in review, and median Strata time per thinking turn down ≥ 15%.

## Step menu (Jeff-Code's `code` adapter idea): parked

The data shows little room for it (10.8% read-only, 2.4% repeats), and building it means a menu builder plus a
trained adapter. Revisit only if traffic shifts towards coding in local repos. Then the first step is measuring how
many Strata calls a code-built menu would have contained, before building anything.

## Housekeeping now

- Keep v1 shadow logging on: Phase 0 extends the same log.
- The `tools` adapter request can be dropped from the shadow path once Phase 1 lands; it costs about 0.36 s of GPU
  per call and the result is clear.
