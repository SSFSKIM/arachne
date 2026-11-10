# core/oracle

The truth check for the core: an independent judge of whether the core decided right when it drew
a frame from an XF stream.

## What this directory holds

- `CONTRACT.md`, the truth contract. It defines what "decided right" means at each of a pixel's 16
  sub-samples: which fragments cover the sub-sample, and in what order they composite. It also
  defines how those decisions become the pixel's bytes. Every rule beyond plain depth is named and
  versioned (`name@1`).
- The `arachne-oracle` crate, to come. It holds two parts:
  - the oracle, which computes from the XF stream the coverage, depth, tie band and admissible
    orders the contract allows;
  - the recompute, which turns the core's recorded decisions and colours into bytes by the
    contract's resolve text.
- `read.py`, to come. It reads the oracle's verdicts beside the core's recorded decisions.
- The contract tests, to come. They hold the core to the contract's text: each conservative set
  must contain the core's decisions, and each named mutation must be rejected.

Status: only the contract exists. No code in this directory yet.

## The independence rule

The oracle and the recompute are written from `CONTRACT.md` alone, never from the core's source.

The crate depends on the core for one thing, the XF reader (`arachne_core::xf`), so that both read
the same stream the same way. A check that lands with the crate (`core/ci/oracle_check.py`)
refuses any other use of the core. The contract tests are the one exception: they call into the
core, because their job is to compare it with the contract.

The other shared components are named, with their versions, in the contract:

- **kurbo.** The outline's flattening and the stroker are contract terms, and both the core and the
  oracle use the library for them.
- **libm.** It supplies the transcendental functions. The projection is written to the last bit, so
  both sides hand kurbo the same input.

Cairo, which flattens and strokes on its own, witnesses the shared component on 2D scenes. Cairo
1.18 with pixman 0.42 is also the contract's pinned definition of a linear gradient.

When the core changes what it does, the contract changes first, as a new version of the rule it
moves. The oracle follows the text.
