//! Lighting: CE's per-face model against values Arachne's own ThreeDCamera computed
//! (`tests/data/shading.txt`, written by `tests/data/gen_shading.py`), and the soft diffuse model,
//! which lights in linear light.

use arachne_core::shade::{SoftDiffuse, ce_face_stops};

/// NumPy takes `** 3` and the norms from the platform and BLAS; the core multiplies and takes the
/// square root itself, so the last bits may differ.
const REL: f64 = 1e-12;

fn nums(tokens: &[&str]) -> Vec<f64> {
    tokens
        .iter()
        .map(|t| t.parse::<f64>().unwrap_or_else(|_| panic!("number: {t}")))
        .collect()
}

#[test]
fn ces_two_stops_match_arachnes_three_d_camera() {
    let text = include_str!("data/shading.txt");
    let mut faces = 0;
    for line in text.lines().filter(|l| l.starts_with("face ")) {
        let t: Vec<&str> = line.split_whitespace().collect();
        let n: usize = t[1].parse().unwrap();
        let points: Vec<[f64; 3]> = nums(&t[2..2 + 3 * n])
            .chunks(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let k: usize = t[2 + 3 * n].parse().unwrap();
        let at = 3 + 3 * n;
        let rgbas: Vec<[f64; 4]> = nums(&t[at..at + 4 * k])
            .chunks(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect();
        let light = nums(&t[at + 4 * k..at + 4 * k + 3]);
        assert_eq!(t[at + 4 * k + 3], "->");
        let want = nums(&t[at + 4 * k + 4..]);
        assert_eq!(want.len(), 8, "{line}");
        let got = ce_face_stops(&points, &rgbas, [light[0], light[1], light[2]]);
        for (g, w) in got.iter().flatten().zip(&want) {
            let err = (g - w).abs() / w.abs().max(1.0);
            assert!(err <= REL, "face {faces}: got {got:?}, want {want:?}");
        }
        faces += 1;
    }
    assert!(faces >= 48, "{faces} faces");
}

#[test]
fn soft_diffuse_lights_in_linear_light() {
    let soft = SoftDiffuse {
        ambient: 0.2,
        wrap: 0.5,
    };
    let point = [0.0, 0.0, 0.0];
    let eye = [0.0, 0.0, 10.0];
    let up = [0.0, 0.0, 1.0];
    let base = [0.5, 0.25, 0.1];
    // Facing the light: full light.
    let lit = soft.lit(base, point, up, [0.0, 0.0, 100.0], eye);
    for k in 0..3 {
        assert!((lit[k] - base[k]).abs() < 1e-15);
    }
    // At the terminator the light wraps: (0 + wrap) / (1 + wrap) of the light past the ambient.
    let side = soft.lit(base, point, up, [100.0, 0.0, 0.0], eye);
    let share = 0.2 + 0.8 * (0.5 / 1.5);
    for k in 0..3 {
        assert!((side[k] - base[k] * share).abs() < 1e-15, "{side:?}");
    }
    // Facing away from the light: the ambient floor.
    let away = soft.lit(base, point, up, [0.0, 0.0, -100.0], eye);
    for k in 0..3 {
        assert!((away[k] - base[k] * 0.2).abs() < 1e-15, "{away:?}");
    }
    // A normal turned away from the eye is lit as the side the eye sees.
    let down = [0.0, 0.0, -1.0];
    assert_eq!(soft.lit(base, point, down, [0.0, 0.0, 100.0], eye), lit);
}
