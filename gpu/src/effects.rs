//! The effects' numbers: the Gaussian a layer is blurred by, the integer kernels a blur and a
//! feather draw with, and how far an effect spreads a layer.
//!
//! A kernel's weights are computed here, on the CPU, in 64-bit floating point, and rounded once
//! to integers, so every adapter is handed the same weights. Pixels are then filtered on integers,
//! so filtering and rounding are the same on every adapter. A kernel's weights are fixed point
//! with 16 fractional bits and sum to exactly [`ONE`]. A blur runs along rows and then along
//! columns (it is separable); between the two passes values are kept in 32-bit integers with 8
//! fractional bits, so the second pass reads no 8-bit rounding of the first. A soft mask's
//! alpha is feathered the same way. The layer is rounded back to 8 bits once, at the end, after
//! the mask has multiplied it.

use crate::frame::Focus;

/// The sum of a kernel's weights: 1.0 with 16 fractional bits.
pub(crate) const ONE: u32 = 1 << 16;

/// The Gaussian a layer is blurred by: its own blur and, when it has a depth under a focus,
/// `aperture * |depth - focus.depth|`, combined as the root of the sum of their squares.
pub(crate) fn resolved_sigma(blur: Option<f64>, depth: Option<f64>, focus: Option<Focus>) -> f64 {
    let own = blur.unwrap_or(0.0);
    let defocus = match (depth, focus) {
        (Some(depth), Some(focus)) => focus.aperture * (depth - focus.depth).abs(),
        _ => 0.0,
    };
    own.hypot(defocus)
}

/// How far a blur of `sigma` reaches, in whole pixels: three sigma, rounded up. The object map
/// grows a footprint by `3 * sigma`, and a grown box is taken to the next whole pixel.
pub(crate) fn blur_reach(sigma: f64) -> f64 {
    (3.0 * sigma).ceil()
}

/// How far a feather reaches, in whole pixels: the taps within it, so that a pixel whose centre is
/// farther than `feather` from the mask's edge sees one side of the edge only. The feather is a
/// Gaussian of sigma `feather / 3`.
pub(crate) fn feather_reach(feather: f64) -> f64 {
    feather.floor()
}

/// The weights of a Gaussian of `sigma` at offsets 0 to `reach`, for a kernel running from
/// `-reach` to `reach`: symmetric, non-increasing, and summing over the whole kernel to exactly
/// [`ONE`], so a constant picture stays constant. Each tail weight is its exact value rounded to
/// the nearest unit while what flooring left over allows; the centre takes the rest. A sigma or
/// reach of 0 is the kernel that changes nothing.
pub(crate) fn kernel(sigma: f64, reach: u32) -> Vec<u32> {
    if reach == 0 || sigma <= 0.0 {
        return vec![ONE];
    }
    let gaussian: Vec<f64> = (0..=reach)
        .map(|k| {
            let k = f64::from(k);
            (-k * k / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let total = gaussian[0] + 2.0 * gaussian[1..].iter().sum::<f64>();
    let exact: Vec<f64> = gaussian
        .iter()
        .map(|g| g / total * f64::from(ONE))
        .collect();
    let mut weights: Vec<u32> = exact.iter().map(|e| e.floor() as u32).collect();
    let floored = weights[0] + 2 * weights[1..].iter().sum::<u32>();
    let mut left = ONE - floored;
    // A tail weight taken up by one unit costs two, one on each side: largest remainders first.
    let remainder = |k: usize| exact[k] - exact[k].floor();
    let mut tails: Vec<usize> = (1..weights.len()).collect();
    tails.sort_by(|&a, &b| remainder(b).total_cmp(&remainder(a)).then(a.cmp(&b)));
    for k in tails {
        if left < 2 || remainder(k) < 0.5 {
            break;
        }
        weights[k] += 1;
        left -= 2;
    }
    weights[0] += left;
    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total(weights: &[u32]) -> u32 {
        weights[0] + 2 * weights[1..].iter().sum::<u32>()
    }

    #[test]
    fn a_kernel_sums_to_one_exactly_and_falls_away_from_its_centre() {
        for (sigma, reach) in [
            (0.3, 1),
            (1.0, 3),
            (2.0, 6),
            (6.0, 18),
            (12.0, 36),
            (4.0 / 3.0, 4),
            (160.0, 480),
        ] {
            let weights = kernel(sigma, reach);
            assert_eq!(weights.len(), reach as usize + 1);
            assert_eq!(total(&weights), ONE, "sigma {sigma}");
            assert!(
                weights.windows(2).all(|w| w[0] >= w[1]),
                "sigma {sigma}: {weights:?}"
            );
        }
    }

    #[test]
    fn a_kernels_spread_is_its_sigma() {
        // Cut at three sigma, a Gaussian keeps 97.3% of its variance.
        for sigma in [1.0, 2.0, 6.0, 12.0] {
            let reach = blur_reach(sigma) as u32;
            let weights = kernel(sigma, reach);
            let variance: f64 = (1..=reach)
                .map(|k| 2.0 * f64::from(k * k) * f64::from(weights[k as usize]))
                .sum::<f64>()
                / f64::from(ONE);
            let ratio = variance / (sigma * sigma);
            assert!((0.95..=1.01).contains(&ratio), "sigma {sigma}: {ratio}");
        }
    }

    #[test]
    fn no_sigma_or_no_reach_is_the_kernel_that_changes_nothing() {
        assert_eq!(kernel(0.0, 5), vec![ONE]);
        assert_eq!(kernel(3.0, 0), vec![ONE]);
        assert_eq!(feather_reach(0.9), 0.0);
        assert_eq!(blur_reach(0.0), 0.0);
    }

    #[test]
    fn a_layers_sigma_combines_its_blur_and_its_defocus() {
        let focus = Some(Focus {
            depth: 2.0,
            aperture: 1.5,
        });
        assert_eq!(resolved_sigma(Some(3.0), None, focus), 3.0);
        assert_eq!(resolved_sigma(None, Some(2.0), focus), 0.0);
        assert_eq!(resolved_sigma(None, Some(6.0), focus), 6.0);
        assert_eq!(resolved_sigma(None, Some(-2.0), focus), 6.0);
        let wide = Some(Focus {
            depth: 1.0,
            aperture: 2.0,
        });
        assert!((resolved_sigma(Some(3.0), Some(3.0), wide) - 5.0).abs() < 1e-12);
        // A depth with no focus, or a focus with no depth, blurs nothing.
        assert_eq!(resolved_sigma(None, Some(6.0), None), 0.0);
        assert_eq!(resolved_sigma(Some(2.5), None, None), 2.5);
    }

    #[test]
    fn a_blur_reaches_three_sigma_rounded_up_and_a_feather_no_further_than_itself() {
        assert_eq!(blur_reach(2.0), 6.0);
        assert_eq!(blur_reach(2.1), 7.0);
        assert_eq!(feather_reach(10.0), 10.0);
        assert_eq!(feather_reach(10.7), 10.0);
    }
}
