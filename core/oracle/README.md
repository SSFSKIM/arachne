# core/oracle

The truth check for the core: an independent judge of whether the core decided right when it drew a
frame from an XF stream.

## What this directory holds

- `CONTRACT.md`, the truth contract: what "decided right" means at each of a pixel's 16 sub-samples
  (which fragments cover it, in what order they composite) and how those decisions become the pixel's
  bytes. Every rule beyond plain depth is named and versioned (`name@1`).
- Later, the `arachne-oracle` crate: the oracle, which recomputes coverage, depth, the tie band and
  the admissible orders from the XF stream by brute force, and the recompute, which turns the core's
  recorded decisions and colours into bytes by the contract's resolve text.
- Later, `read.py`, which reads the oracle's verdicts beside the core's recorded decisions, and the
  contract tests, which hold the core to the contract's text.

Status: only the contract exists. No code in this directory yet.

## The independence rule

The oracle and the recompute are written from `CONTRACT.md` alone, never from the core's source.
The crate depends on the core for one thing, the XF reader (`arachne_core::xf`), so that both read
the same stream the same way; a check that lands with the crate (`core/ci/oracle_check.py`) refuses
any other use of the core. The one
exception is the contract tests, which call into the core because their job is to compare it with
the contract.

The one shared component is kurbo (the version the contract names): the outline's flattening and
the stroker are contract terms, and both the core and the oracle use the library for them. Cairo,
which flattens and strokes on its own, witnesses that component on 2D scenes.

When the core changes what it does, the contract changes first, as a new version of the rule it
moves; the oracle follows the text.
