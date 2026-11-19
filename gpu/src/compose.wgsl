// One layer stacked over what is below it, in Cairo's 8-bit arithmetic on painted values.
//
// Every texture is Rgba8Uint: values are read and written as the integers they are, so the
// stacking itself rounds the same on every adapter. A quad's placement is floating point: a pixel
// centre within rounding of a texel's or the quad's edge may take another texel on another
// adapter. A full-frame triangle covers every pixel; a pixel the layer does not cover takes the
// value below unchanged.

struct Layer {
    // The inverse of the quad's placement: a pixel position (x, y, 1) to texel coordinates
    // (u * w, v * h, w), rows of a 3x3 matrix; w > 0 inside the quad.
    row0: vec4<f32>,
    row1: vec4<f32>,
    row2: vec4<f32>,
    // The colour of a fill (kind 2).
    colour: vec4<u32>,
    // The source texture's size in texels.
    size: vec2<u32>,
    // 0: a picture of the frame's size; 1: a quad, nearest resampling; 2: a fill of `colour`.
    kind: u32,
    // 0..=255.
    opacity: u32,
}

@group(0) @binding(0) var<uniform> layer: Layer;
@group(0) @binding(1) var below: texture_2d<u32>;
@group(0) @binding(2) var source: texture_2d<u32>;

@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

// round(a * b / 255) for 8-bit a and b, exactly: pixman's MUL_UN8.
fn mul_un8(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    let t = a * b + vec4<u32>(128u);
    return ((t >> vec4<u32>(8u)) + t) >> vec4<u32>(8u);
}

@fragment
fn fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<u32> {
    if layer.kind == 2u {
        return layer.colour;
    }
    // The fragment's position is its pixel's centre, (x + 0.5, y + 0.5).
    let pixel = vec2<i32>(floor(position.xy));
    let d = textureLoad(below, pixel, 0);
    var s = vec4<u32>(0u);
    if layer.kind == 0u {
        s = textureLoad(source, pixel, 0);
    } else {
        let p = vec3<f32>(position.x, position.y, 1.0);
        let w = dot(layer.row2.xyz, p);
        if w > 0.0 {
            let texel = vec2<f32>(dot(layer.row0.xyz, p), dot(layer.row1.xyz, p)) / w;
            let size = vec2<f32>(layer.size);
            if all(texel >= vec2<f32>(0.0)) && all(texel < size) {
                let at = min(vec2<u32>(floor(texel)), layer.size - vec2<u32>(1u));
                s = textureLoad(source, vec2<i32>(at), 0);
            }
        }
    }
    s = mul_un8(s, vec4<u32>(layer.opacity));
    // Painted "over", saturating as pixman's is.
    return min(s + mul_un8(d, vec4<u32>(255u - s.a)), vec4<u32>(255u));
}
