# AAPL SEC/Gemma online risk overlay v2.2 preflight rejection

## Status

The preregistered v2.2 approach is rejected before any official SEC or Yahoo
request, Gemma generation, prediction, trade, or performance calculation.

The implementation commit is
`efbc481c57e480d48303763163676e64e87df49d`. It remains preserved on
`codex/aapl-sec-gemma-online-risk-overlay-v2-2` together with its parent
preregistration commit `a849b9d704ffd98547e570735a221b2b75f7db86`.

## Checks that passed

- The committed branch was clean, pushed, and identical to its live remote.
- The live implementation manifest was
  `a779c0b669ff7fa323a81fd6a6ef7467ad3333caf35ef452d9fde2195d4d3804`.
- The pinned dependency closure was
  `d37bb45d1d792efa88cb834e1b6a0cabde5ea3e435466a00905002b29152e1bd`.
- The local runtime probe asked Ollama only for `/api/version` and
  `/api/show`; it made no `/api/chat` request and generated no model output.
  Its payload hash was
  `c26dfd47a62115e8d71340d31d386086bcd90253c6a82a92e0ada2d166068f1e`.
- The focused v2.2 verification set passed 518 checks.
- No benchmark, Python worker, Gemma generation, or llama runner remained
  active after the preflight.

These checks show that the implementation is reproducible and locally ready.
They do not repair the runtime-contract failure below.

## Rejection reason

The user's limit applies to the complete approach test. V2.2 instead gives a
fresh 3,600-second clock to each of four separately executed commands:

1. development acquisition;
2. development scoring;
3. confirmation scoring; and
4. final scoring.

Publication recovery may then use additional 300-second invocations. The
contract explicitly reports cumulative multi-stage and recovery time
separately and does not claim that the complete lifecycle is sub-hour.
Consequently, v2.2 could legally consume several hours even though every
individual command stayed below one hour.

An outside timer could kill the process tree after one hour, but that would
only prove that an incomplete run was stopped. It would not prove that the
complete approach finished inside the limit. V2.2 also has no public
all-stages entry point and no immutable receipt binding every stage to one
shared start and deadline.

This is a hard preregistered requirement failure, not a trading-performance
failure. Running the model despite it would waste time and would not satisfy
the goal.

## Required successor

A successor may keep the same trading thesis, chronology, long/cash rules,
costs, and success gates, but it must preregister:

- one monotonic 3,600-second deadline shared by acquisition, development,
  confirmation, final scoring, publication, and cleanup;
- admission checks that start a later stage only while enough shared time
  remains;
- one supported all-stages coordinator;
- one immutable lifecycle timing receipt; and
- fail-closed treatment of any unfinished publication or recovery at the
  shared deadline.

The missing private SEC contact remains an execution prerequisite for that
successor, but it is not the reason v2.2 is rejected.

## Result

V2.2 has no return, no comparison with buy-and-hold, and no evidence about
whether the filing signal predicts AAPL. It is preserved solely as a complete
implementation and a no-effect runtime preflight rejection.
