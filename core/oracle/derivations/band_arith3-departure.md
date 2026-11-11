# `band_arith@3`: the rounding counts with a `Nearest` face's departure `O` inside `(1 + 64u)`

This note supports `CONTRACT.md` §6 (`band_arith@3`); §5 and §6 below are that file's.

A `Nearest` face's `R` is `(1 + 64u) ((M + ρ + O) + ν)`: its outline's departure `O` is a term of
the sum the factor covers, not an addition after it, as `band_arith@2` had it. This note derives
the counts §6 states for that sum: `O` 22 deep where it enters, 28 roundings in the order the
formulas give, 27 in any other order, and `E_k = γ_{2N+66}` with `n = N + 33`.

The counting rule is §6's own: an input counts 0; a sum counts one more than its larger operand;
a product or quotient counts its operands' counts plus one; a square root (and `hypot`, correctly
rounded) counts one more than its operand. View terms (the rotation's entries `R_ij`, `c`, `d`,
`z0`, `xx`, `yy`, `x0`, `y0`) and the rounded plane's grid values (`n`, `e`) are inputs. A read's
shown position `(x, y, z)` is an input.

## 1. `O`'s depth

The rounded plane's screen key, in §5's order (`m_i = (R_i0 n_0 + R_i1 n_1) + R_i2 n_2`,
`D = (e − n·c) − m_2 d`, `g_x = m_0 / (z0 xx)`, `b = −g_x / D`, `c_k = −g_y / D`,
`a = −(((−g_x x0) − g_y y0) − m_2 d) / D`):

`c_k` is §5's coefficient `c`, written apart from the camera point `c`.

| quantity | count | why |
|---|---|---|
| `m_i` | 3 | three products (1), two sums (2, 3) |
| `n·c` | 3 | as `m_i` |
| `e − n·c` | 4 | sum |
| `m_2 d` | 4 | 3 + 0 + 1 |
| `D` | 5 | sum |
| `z0 xx` | 1 | product |
| `g_x`, `g_y` | 5 | 3 + 1 + 1 |
| `b`, `c_k` | 11 | 5 + 5 + 1 |
| `g_x x0`, `g_y y0` | 6 | 5 + 0 + 1 |
| `(−g_x x0) − g_y y0` | 7 | sum |
| `… − m_2 d` | 8 | sum |
| `a` | 14 | 8 + 5 + 1 |

A point `q`'s terms:

| quantity | count | why |
|---|---|---|
| `map(q)` (`X = xx x + x0`) | 2 | product, sum |
| `b X`, `c_k Y` | 14 | 11 + 2 + 1 |
| `a + b X` | 15 | sum |
| `plane(map(q))` | 16 | sum |
| `f = d / (d − z)` | 2 | sum (1), quotient 0 + 1 + 1 |
| `k_q = −f` | 2 | negation exact |
| `gap_q` | 17 | sum, `\|·\|` exact |
| `ρ(q) = 4u (\|a\| + \|b X\| + \|c_k Y\|)` | 17 | sums 15, 16; product 16 + 0 + 1 |
| `r = \|z\| + hypot(x, y) / \|f z0\| + \|c\|` | 7 | `hypot` 1; `f z0` 3; quotient 5; sums 6, 7 |
| `4 f² / \|d\| · r` | 15 | `f²` 5; `4 f²` 6; `/\|d\|` 7; `· r` 7 + 7 + 1 |
| `key_error(q) = u (2\|f\| + 4 f² / \|d\| · r)` | 17 | `2\|f\|` 3; sum 16; product 17 |
| `4u \|k_q\|` | 3 | product |
| `u gap_q` | 18 | product |
| `((((gap_q + ρ) + key_error) + 4u\|k_q\|) + u gap_q)` | 21 | sums 18, 19, 20, 21 |
| `max` over `q` | 21 | exact |
| `hypot(b, c_k) · 0.1` | 13 | `hypot` 12; product with the constant (an input) 12 + 0 + 1 |
| `O` | 22 | sum |

So `O` is 22 deep where it enters `R`'s sum, under the 25 that `M + ρ` reaches (below).

## 2. `R`'s own sum

Without `O`, every term reaches `(1 + 64u) ((M + ρ) + ν)` through at most 27 roundings, the
factor included, the longest through `|eye|, dD, f dD, A, 3A, M, ρ, ν`. That makes `(M + ρ)` at
most 25, `((M + ρ) + ν)` at most 26, and the factor 27; in any other order, at most 26.

With `R = (1 + 64u) ((M + ρ + O) + ν)`, read left to right:
- `(M + ρ) + O`: `max(25, 22) + 1 = 26`;
- `+ ν`: 27;
- the factor: 28.

Every path through `M + ρ` gains exactly one rounding; `O`'s own paths reach `R` through at most
`22 + 3 = 25`. So: **at most 28** in the order the formulas give.

A conforming computation of the same terms in another order: without `O` it is at most 26, one
under the 27 that includes the factor. Adding the one operand `O` to the sum adds at most one
rounding to every path through it, and `O`'s five-term sum taken in its worst order (the deepest
operand, `u gap_q` at 18, first) is `18 + 4 = 22`, then 23 with the flattening term, then at most
`23 + 3 = 26` into `R` without the factor. So: **at most 27**.

## 3. The inequality

Needed: `(1 + 64u)(1 − γ_28) >= 1 + γ_27`. With `γ_j = j u / (1 − j u)`,
`1 − γ_28 = (1 − 56u) / (1 − 28u)` and `1 + γ_27 = 1 / (1 − 27u)`, so the inequality is
`(1 + 64u)(1 − 56u)(1 − 27u) >= 1 − 28u`. The left side is
`1 − 19u − 3800u² + 96768u³`; the difference is `u (9 − 3800u + 96768u²)`. The quadratic has
real roots near `2.53e-3` and `3.67e-2` and is negative between them, so the difference is
positive below the smaller root (sufficient: `u < 9/3800`, where `9 − 3800u > 0` and the `u²` term
only adds). The inequality holds for u ≤ 2^−24, which covers binary32 and binary64 (at both the
quadratic is `≈ 9`). The rewriting also needs `1 − 56u`, `1 − 28u` and `1 − 27u` positive, true
there, so `γ_27` and `γ_28` are finite and the cross-multiplication keeps its direction.

## 4. The `E_k` path

Without `O`, every path from a point's coordinate through `E_k` to `R` passes at most `N + 32`
roundings, the longest through `E_k`'s first sum, `δe`, `C`'s product, `M`, `ρ`, `ν` and the
factor. With `O` between `ρ` and `ν` that path passes `N + 33`.

`E_k = γ_{2N+66}`, `n = N + 33`. The identity `γ_{2n} (1 − γ_n) = 2 γ_n` holds exactly
(`γ_{2n}(1 − γ_n) = (2nu / (1 − 2nu)) · ((1 − 2nu) / (1 − nu)) = 2 γ_n`, for `2nu < 1`, the
domain below). `δe` as computed counts for at least `4 γ_{N+33} G`, and that covers the offsets'
difference `(2 γ_{N+2} + 8u (1 + γ_{N+2})) G = ((2N + 12) u / (1 − (N + 2) u)) G`, since
`4 γ_{N+33} = (4N + 132) u / (1 − (N + 33) u)` has the larger numerator and the smaller
denominator, both denominators positive inside the domain. The domain is `(2N + 66) u < 1`; there
`(N + 33) u < 1/2`, so `1 − (N + 33) u > 0` and `1 − (N + 2) u > 0`, and outside it `δe`, and so
`R`, is infinite. Against `γ_{2N+64}`, `R` grows by a relative `≈ 2u` on a `Nearest` face's `δe`:
looser, never tighter.

Keeping `γ_{2N+64}` with `n = N + 33` would break the identity: `γ_{2N+64} (1 − γ_{N+33})` is at
least `2 γ_{N+32} (1 − 2u / (1 − (2N + 64) u))` inside the domain, so the cover holds to first
order (`4N + 128` against `2N + 12`) but its margin vanishes as `(2N + 64) u → 1`, and the domain
would need a further bound to state.

## What this rests on

- §5's order for the rounded plane's screen coefficients as written above (`a`, `b`, `c_k` from
  `m`, `D`, `g`); if §5 pinned another order the table would move, and only `O <= 25` matters.
- The counts without `O` (27, 26 and `N + 32`), as `band_arith@2` stated them: this note adds
  one rounding to each and derives `O`'s depth fresh.
