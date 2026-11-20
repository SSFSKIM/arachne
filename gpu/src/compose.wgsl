// One layer stacked over what is below it, in Cairo's 8-bit arithmetic on painted values; or,
// for an effect, placed alone; or, for the object map, marked as one bit of coverage.
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
    // 0: a picture of the frame's size; 1: a quad; 2: a fill of `colour`.
    kind: u32,
    // 0..=255.
    opacity: u32,
    // 0: stacked over what is below; 1: bit `bit` of a coverage frame set where the layer puts any
    // alpha (32 bits a pixel, bit b in channel b / 8); 2: placed alone, over nothing.
    mode: u32,
    bit: u32,
    // A quad's resampling: 0 nearest; 1 bilinear (a sized window at an axis-aligned rectangle),
    // in fixed point, scaled by the pixel's coverage of the rectangle.
    resampling: u32,
    // For bilinear: the outline, an axis-aligned rectangle (x0, y0, x1, y1) in pixels.
    rect: vec4<f32>,
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

// The source's texel at `at`, the nearest edge texel past its edge.
fn texel_clamped(at: vec2<i32>) -> vec4<u32> {
    return textureLoad(source, clamp(at, vec2<i32>(0), vec2<i32>(layer.size) - vec2<i32>(1)), 0);
}

// How much of the pixel at `position` the outline covers, in 8 fractional bits (0 to 256): the
// area of the pixel's cell inside the rectangle, exactly. A cell that does not touch it gets 0,
// and a cell wholly inside gets 256.
fn coverage(position: vec2<f32>) -> u32 {
    let cell = floor(position);
    let near = max(cell, layer.rect.xy);
    let far = min(cell + vec2<f32>(1.0), layer.rect.zw);
    let inside = max(far - near, vec2<f32>(0.0));
    return u32(floor(inside.x * inside.y * 256.0 + 0.5));
}

// The source at texel coordinates `texel`, bilinear, scaled by `cover` (0 to 256): the four
// texels round it (texel centres at i + 0.5, the edge texel continued past the edge), weighted by
// its offset from them in 8 fractional bits a side, times the cover, rounded half up to 8 bits
// once. At a texel's centre the offset is 0, and wholly inside the cover is 256, so the texel
// comes back exactly. Every sum fits in 32 bits: at most (255 << 16) * 256 + (1 << 23).
fn bilinear(texel: vec2<f32>, cover: u32) -> vec4<u32> {
    let at = texel - vec2<f32>(0.5);
    let base = floor(at);
    let f = vec2<u32>(floor((at - base) * 256.0 + vec2<f32>(0.5)));
    let i = vec2<i32>(base);
    let wx = f.x;
    let wy = f.y;
    let top = texel_clamped(i) * (256u - wx) + texel_clamped(i + vec2<i32>(1, 0)) * wx;
    let bottom = texel_clamped(i + vec2<i32>(0, 1)) * (256u - wx)
        + texel_clamped(i + vec2<i32>(1, 1)) * wx;
    let sum = top * (256u - wy) + bottom * wy;
    return (sum * cover + vec4<u32>(1u << 23u)) >> vec4<u32>(24u);
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
            if layer.resampling == 0u {
                if all(texel >= vec2<f32>(0.0)) && all(texel < size) {
                    let at = min(vec2<u32>(floor(texel)), layer.size - vec2<u32>(1u));
                    s = textureLoad(source, vec2<i32>(at), 0);
                }
            } else {
                // Bounded by the pixel's own coverage of the rectangle, in pixels: nothing lands
                // in a pixel the rectangle does not touch, whatever the texel's size.
                let cover = coverage(position.xy);
                if cover > 0u {
                    s = bilinear(texel, cover);
                }
            }
        }
    }
    s = mul_un8(s, vec4<u32>(layer.opacity));
    if layer.mode == 1u {
        let channel = vec4<u32>(0u, 1u, 2u, 3u) == vec4<u32>(layer.bit / 8u);
        let flag = select(vec4<u32>(0u), vec4<u32>(1u << (layer.bit % 8u)), channel);
        return select(d, d | flag, s.a > 0u);
    }
    if layer.mode == 2u {
        return s;
    }
    // Painted "over", saturating as pixman's is.
    return min(s + mul_un8(d, vec4<u32>(255u - s.a)), vec4<u32>(255u));
}
