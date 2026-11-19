// A layer's effects, on integers, so every adapter rounds them alike (see effects.rs).
//
// `blur_pass` runs a kernel along rows or along columns. Its weights are fixed point with 16
// fractional bits and sum to 1 << 16; the clamped index continues the picture's edge outward. Run
// on an 8-bit texture it keeps 8 fractional bits (shift 8); run on its own output, wide values with
// 8 fractional bits, it keeps 8 again (shift 16). Every sum fits in 32 bits: at most
// (1 << 16) * (255 << 8), plus the half that rounds it.
//
// `finish_pass` multiplies the blurred layer by its feathered mask's alpha, both with 8 fractional
// bits, and rounds once, half up, to 8 bits.

@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

struct Blur {
    // (1, 0) along rows, (0, 1) along columns.
    direction: vec2<i32>,
    // The kernel runs from -radius to radius.
    radius: i32,
    // 8 from an 8-bit texture, 16 from a wide one.
    shift: u32,
}

@group(0) @binding(0) var<uniform> blur: Blur;
// The weights at offsets 0 to radius.
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var blur_source: texture_2d<u32>;

@fragment
fn blur_pass(@builtin(position) position: vec4<f32>) -> @location(0) vec4<u32> {
    let pixel = vec2<i32>(floor(position.xy));
    let last = vec2<i32>(textureDimensions(blur_source)) - vec2<i32>(1);
    var sum = vec4<u32>(0u);
    for (var k = -blur.radius; k <= blur.radius; k++) {
        let at = clamp(pixel + blur.direction * k, vec2<i32>(0), last);
        sum += weights[u32(abs(k))] * textureLoad(blur_source, at, 0);
    }
    let half = vec4<u32>(1u << (blur.shift - 1u));
    return (sum + half) >> vec4<u32>(blur.shift);
}

struct Finish {
    // 1 when the layer has a mask.
    masked: u32,
    // 1 when the mask keeps what it would remove.
    invert: u32,
    padding: vec2<u32>,
}

// 255 with 8 fractional bits: a mask that keeps everything.
const FULL: u32 = 65280u;
// The product of two values with 8 fractional bits, over 255 and both fractions: 255 << 16.
const SCALE: u32 = 16711680u;

@group(0) @binding(3) var<uniform> finish: Finish;
@group(0) @binding(4) var blurred: texture_2d<u32>;
@group(0) @binding(5) var feathered: texture_2d<u32>;

@fragment
fn finish_pass(@builtin(position) position: vec4<f32>) -> @location(0) vec4<u32> {
    let pixel = vec2<i32>(floor(position.xy));
    let value = textureLoad(blurred, pixel, 0);
    if finish.masked == 0u {
        return (value + vec4<u32>(128u)) >> vec4<u32>(8u);
    }
    var alpha = textureLoad(feathered, pixel, 0).a;
    if finish.invert == 1u {
        alpha = FULL - alpha;
    }
    // At most 65280 * 65280 + SCALE / 2, which fits in 32 bits.
    return (value * alpha + vec4<u32>(SCALE / 2u)) / vec4<u32>(SCALE);
}
