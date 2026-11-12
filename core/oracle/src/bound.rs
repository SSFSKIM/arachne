//! Folds over bound terms in which a term that is not a number never drops out (contract §6,
//! "Unbounded keys": where `S` or `R` is infinite the oracle states no bound). `f64::max` and
//! `f64::min` return the other operand when one is NaN, so a largest bound term that is NaN
//! would leave the bound at its finite terms. Here it states no bound instead.

/// The margin `η = 2^-20` (contract §6, "The margin `η`"), taken relative to the magnitude of a
/// decision's inputs.
pub const ETA: f64 = 1.0 / 1_048_576.0;

/// An upper-bound quotient `num / den` under the margin: infinite where `den` is within `η` of
/// zero at the inputs' scale `scale` (or below it), else the quotient times `(1 + 2η)`.
pub fn quotient(num: f64, den: f64, scale: f64) -> f64 {
    if den > ETA * scale {
        num / den * (1.0 + 2.0 * ETA)
    } else {
        f64::INFINITY
    }
}

/// The larger of `a` and `b`, infinite where either is NaN: a bound term that is not a number
/// bounds nothing, so the bound it joins states none.
pub fn max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::INFINITY
    } else {
        a.max(b)
    }
}

/// The least of `terms`, or `on_nan` where any term is NaN: the caller names the end a least
/// that is not known must take (`∞` for a distance, 0 for a width, `−∞` for a key's low end).
pub fn least(terms: impl IntoIterator<Item = f64>, on_nan: f64) -> f64 {
    let mut out = f64::INFINITY;
    for t in terms {
        if t.is_nan() {
            return on_nan;
        }
        out = out.min(t);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_nan_operand_makes_the_largest_unbounded() {
        assert_eq!(max(1.0, f64::NAN), f64::INFINITY);
        assert_eq!(max(f64::NAN, 1.0), f64::INFINITY);
        assert_eq!(
            [0.5, f64::NAN, 2.0].into_iter().fold(0.0, max),
            f64::INFINITY
        );
        assert_eq!([0.5, 2.0].into_iter().fold(0.0, max), 2.0);
    }

    #[test]
    fn a_quotient_within_its_margin_is_unbounded_and_beyond_it_is_widened() {
        assert_eq!(quotient(1.0, ETA, 1.0), f64::INFINITY);
        assert_eq!(quotient(1.0, 0.0, 0.0), f64::INFINITY);
        assert_eq!(quotient(1.0, f64::NAN, 1.0), f64::INFINITY);
        assert_eq!(quotient(1.0, 2.0, 2.0), 0.5 * (1.0 + 2.0 * ETA));
    }

    #[test]
    fn a_nan_term_makes_the_least_its_named_end() {
        assert_eq!(least([3.0, f64::NAN, 1.0], f64::INFINITY), f64::INFINITY);
        assert_eq!(least([3.0, 1.0, f64::NAN], 0.0), 0.0);
        assert_eq!(least([3.0, 1.0], 0.0), 1.0);
        assert_eq!(least([], 0.0), f64::INFINITY);
    }
}
