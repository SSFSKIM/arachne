//! The compositor: textures of two kinds, a stack of layers, and the frame read back.

use std::collections::HashMap;
use std::fmt;

use wgpu::util::DeviceExt;

use crate::device::{self, AdapterSummary, Door, NoGpu, Pin};
use crate::frame::{Frame, NEAREST, Source};
use crate::keys::{Key, image_key, painted_from_straight, painted_key};
use crate::placement::{Placement, place};

/// Every texture and target: 8-bit values read and written as integers.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Uint;

/// Why a frame was not drawn or a texture not taken. Nothing is drawn when a frame is refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DrawError {
    /// A layer names a key no texture is held under.
    UnknownKey(Key),
    /// The frame asks for something this build does not draw (`render.unsupported`).
    Unsupported(String),
    /// Malformed input: a key that is not its bytes' key, a size that does not fit.
    Invalid(String),
    /// The device failed.
    Device(String),
}

impl fmt::Display for DrawError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DrawError::UnknownKey(key) => {
                write!(f, "no texture is held under the key {}", hex(key))
            }
            DrawError::Unsupported(what) => write!(f, "render.unsupported: {what}"),
            DrawError::Invalid(what) => f.write_str(what),
            DrawError::Device(what) => write!(f, "the device failed: {what}"),
        }
    }
}

impl std::error::Error for DrawError {}

/// Why no compositor was made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpenError {
    /// No adapter this door may draw on, or the chosen one could not be opened.
    NoGpu(NoGpu),
    /// The frame size does not fit the device: no side may be 0 or above the device's texture
    /// limit, and a frame must fit its read-back buffer.
    InvalidSize(String),
}

impl fmt::Display for OpenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpenError::NoGpu(no_gpu) => no_gpu.fmt(f),
            OpenError::InvalidSize(what) => f.write_str(what),
        }
    }
}

impl std::error::Error for OpenError {}

impl From<NoGpu> for OpenError {
    fn from(no_gpu: NoGpu) -> Self {
        OpenError::NoGpu(no_gpu)
    }
}

fn hex(key: &Key) -> String {
    key.iter().map(|b| format!("{b:02x}")).collect()
}

struct Texture {
    view: wgpu::TextureView,
    texture: wgpu::Texture,
    width: u32,
    height: u32,
}

/// One layer pass's parameters, laid out as `compose.wgsl`'s `Layer`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LayerUniform {
    rows: [[f32; 4]; 3],
    colour: [u32; 4],
    size: [u32; 2],
    kind: u32,
    opacity: u32,
}

const PICTURE: u32 = 0;
const QUAD: u32 = 1;
const FILL: u32 = 2;

/// A Vulkan device with no window, the textures it holds, and the frame size it draws.
pub struct Compositor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    software: bool,
    width: u32,
    height: u32,
    max_dimension: u32,
    images: HashMap<Key, Texture>,
    painted: HashMap<Key, Texture>,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    accumulators: [Texture; 2],
    readback: wgpu::Buffer,
    padded_row: u32,
}

impl Compositor {
    /// A compositor drawing `width` x `height` frames on a GPU: a discrete, integrated or virtual
    /// adapter, never a software one. Fails with [`OpenError::NoGpu`], naming every adapter it
    /// saw, or [`OpenError::InvalidSize`] for a frame size the device cannot draw.
    pub fn new(width: u32, height: u32) -> Result<Self, OpenError> {
        Self::open(width, height, Door::Ordinary)
    }

    pub(crate) fn open(width: u32, height: u32, door: Door) -> Result<Self, OpenError> {
        if width == 0 || height == 0 {
            return Err(OpenError::InvalidSize(format!(
                "a frame of {width} x {height} has no pixels"
            )));
        }
        let opened = device::open(door)?;
        // Both limits are checked before any resource exists.
        let (padded_row, readback_size) = frame_size(width, height, &opened.limits)?;
        let max_dimension = opened.limits.max_texture_dimension_2d;
        let device = opened.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compose"),
            source: wgpu::ShaderSource::Wgsl(include_str!("compose.wgsl").into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("compose"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let layout = pipeline.get_bind_group_layout(0);
        let accumulator = || {
            texture(
                &device,
                width,
                height,
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            )
        };
        let accumulators = [accumulator(), accumulator()];
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: readback_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Ok(Compositor {
            device,
            queue: opened.queue,
            info: opened.info,
            software: door == Door::Software,
            width,
            height,
            max_dimension,
            images: HashMap::new(),
            painted: HashMap::new(),
            pipeline,
            layout,
            accumulators,
            readback,
            padded_row,
        })
    }

    /// The device this compositor draws on.
    pub fn pin(&self) -> Pin {
        Pin {
            adapter: self.info.name.clone(),
            backend: format!("{:?}", self.info.backend),
            driver: self.info.driver_info.clone(),
            wgpu: crate::WGPU_VERSION.to_string(),
        }
    }

    /// Everything the adapter it draws on reports.
    pub fn adapter(&self) -> AdapterSummary {
        AdapterSummary::of(&self.info)
    }

    /// True only for a compositor from [`crate::testing::software_compositor`].
    pub fn software(&self) -> bool {
        self.software
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Takes an image: straight RGBA, `key` its Frame IR image-shape key. Its values are converted
    /// to painted ones here, once. Holding the key already, it does nothing.
    pub fn upload_image(
        &mut self,
        key: Key,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> Result<(), DrawError> {
        self.check_size(width, height, rgba)?;
        if key != image_key(width, height, rgba) {
            return Err(DrawError::Invalid(
                "the key is not the Frame IR image-shape key of these straight RGBA bytes".into(),
            ));
        }
        if !self.images.contains_key(&key) {
            let texture = self.texture_of(width, height, &painted_from_straight(rgba));
            self.images.insert(key, texture);
        }
        Ok(())
    }

    /// Takes a painted picture: painted RGBA, `key` its [`painted_key`]. Holding the key already,
    /// it does nothing.
    pub fn upload_painted(
        &mut self,
        key: Key,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> Result<(), DrawError> {
        self.check_size(width, height, rgba)?;
        if key != painted_key(width, height, rgba) {
            return Err(DrawError::Invalid(
                "the key is not the painted key of these painted RGBA bytes".into(),
            ));
        }
        if !self.painted.contains_key(&key) {
            let texture = self.texture_of(width, height, rgba);
            self.painted.insert(key, texture);
        }
        Ok(())
    }

    /// Lets go of the texture held under `key`, of either kind; false if none was.
    pub fn release(&mut self, key: &Key) -> bool {
        let image = self.images.remove(key);
        let painted = self.painted.remove(key);
        image.is_some() || painted.is_some()
    }

    fn check_size(&self, width: u32, height: u32, rgba: &[u8]) -> Result<(), DrawError> {
        let max = self.max_dimension;
        if !(1..=max).contains(&width) || !(1..=max).contains(&height) {
            return Err(DrawError::Invalid(format!(
                "a texture of {width} x {height} does not fit this device (1 to {max} a side)"
            )));
        }
        if rgba.len() as u64 != u64::from(width) * u64::from(height) * 4 {
            return Err(DrawError::Invalid(format!(
                "{} bytes are not a {width} x {height} RGBA texture",
                rgba.len()
            )));
        }
        Ok(())
    }

    fn texture_of(&self, width: u32, height: u32, rgba: &[u8]) -> Texture {
        let texture = texture(
            &self.device,
            width,
            height,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            extent(width, height),
        );
        texture
    }

    fn held(&self, key: &Key) -> Result<&Texture, DrawError> {
        self.images
            .get(key)
            .or_else(|| self.painted.get(key))
            .ok_or(DrawError::UnknownKey(*key))
    }

    /// Draws the frame: tight RGBA8 rows, top first, painted. The whole frame is checked before
    /// anything is drawn; a refused frame draws nothing. A draw holds the compositor exclusively
    /// from its first pass to the read-back buffer's unmapping: its accumulators and that buffer
    /// are one frame's at a time.
    pub fn draw(&mut self, frame: &Frame) -> Result<Vec<u8>, DrawError> {
        let mut passes = Vec::with_capacity(frame.layers.len());
        for (index, layer) in frame.layers.iter().enumerate() {
            let texture = self.held(layer.source.key())?;
            if !(0.0..=1.0).contains(&layer.opacity) {
                return Err(DrawError::Invalid(format!(
                    "layer {index}: opacity {} is not in 0..=1",
                    layer.opacity
                )));
            }
            let mut uniform = LayerUniform {
                rows: [[0.0; 4]; 3],
                colour: [0; 4],
                size: [texture.width, texture.height],
                kind: PICTURE,
                opacity: (layer.opacity * 255.0).round() as u32,
            };
            match &layer.source {
                Source::Picture { .. } => {
                    if (texture.width, texture.height) != (self.width, self.height) {
                        return Err(DrawError::Invalid(format!(
                            "layer {index}: a picture of {} x {} in a frame of {} x {}",
                            texture.width, texture.height, self.width, self.height
                        )));
                    }
                }
                Source::Quad {
                    corners,
                    resampling,
                    ..
                } => {
                    if *resampling != NEAREST {
                        return Err(DrawError::Unsupported(format!(
                            "layer {index}: resampling {resampling} (PIL's value); this build \
                             resamples a quad by nearest ({NEAREST}) only"
                        )));
                    }
                    if corners.iter().flatten().any(|c| !c.is_finite()) {
                        return Err(DrawError::Invalid(format!(
                            "layer {index}: a quad's corners are not finite"
                        )));
                    }
                    match place(corners, texture.width, texture.height) {
                        Placement::Mapped(rows) => {
                            uniform.kind = QUAD;
                            uniform.rows = rows;
                        }
                        Placement::Empty => continue,
                        Placement::Refused(why) => {
                            return Err(DrawError::Unsupported(format!("layer {index}: {why}")));
                        }
                    }
                }
            }
            passes.push((texture, uniform));
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let fill = LayerUniform {
            rows: [[0.0; 4]; 3],
            colour: frame.background.map(u32::from),
            size: [0; 2],
            kind: FILL,
            opacity: 255,
        };
        let [first, second] = &self.accumulators;
        self.pass(&mut encoder, fill, second, second, first);
        let mut current = 0;
        for (texture, uniform) in passes {
            let below = &self.accumulators[current];
            let target = &self.accumulators[1 - current];
            self.pass(&mut encoder, uniform, below, texture, target);
            current = 1 - current;
        }
        self.read_back(encoder, &self.accumulators[current])
    }

    fn pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        uniform: LayerUniform,
        below: &Texture,
        source: &Texture,
        target: &Texture,
    ) {
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&below.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&source.view),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Copies the frame out: to a staging buffer, mapped, waited for, rows copied out.
    fn read_back(
        &self,
        mut encoder: wgpu::CommandEncoder,
        source: &Texture,
    ) -> Result<Vec<u8>, DrawError> {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &source.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_row),
                    rows_per_image: Some(self.height),
                },
            },
            extent(self.width, self.height),
        );
        self.queue.submit(Some(encoder.finish()));
        let slice = self.readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| DrawError::Device(e.to_string()))?;
        receiver
            .recv()
            .map_err(|e| DrawError::Device(e.to_string()))?
            .map_err(|e| DrawError::Device(e.to_string()))?;
        let row = (self.width * 4) as usize;
        let mut rgba = Vec::with_capacity(row * self.height as usize);
        {
            let view = slice
                .get_mapped_range()
                .map_err(|e| DrawError::Device(e.to_string()))?;
            for y in 0..self.height as usize {
                let start = y * self.padded_row as usize;
                rgba.extend_from_slice(&view[start..start + row]);
            }
        }
        self.readback.unmap();
        Ok(rgba)
    }
}

/// A frame's read-back row (padded to the copy alignment) and buffer size, once its sides are
/// known to fit `limits`: none may be 0 or above the texture limit, and the read-back buffer may
/// not pass the buffer limit.
fn frame_size(width: u32, height: u32, limits: &wgpu::Limits) -> Result<(u32, u64), OpenError> {
    let max_dimension = limits.max_texture_dimension_2d;
    if width == 0 || height == 0 || width > max_dimension || height > max_dimension {
        return Err(OpenError::InvalidSize(format!(
            "a frame of {width} x {height} does not fit this device (1 to {max_dimension} a side)"
        )));
    }
    let align = u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let padded_row = (u64::from(width) * 4).div_ceil(align) * align;
    let readback_size = padded_row * u64::from(height);
    if readback_size > limits.max_buffer_size {
        return Err(OpenError::InvalidSize(format!(
            "a frame of {width} x {height} needs a {readback_size}-byte read-back buffer; this \
             device allows {}",
            limits.max_buffer_size
        )));
    }
    Ok((padded_row as u32, readback_size))
}

fn extent(width: u32, height: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    }
}

fn texture(device: &wgpu::Device, width: u32, height: u32, usage: wgpu::TextureUsages) -> Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: extent(width, height),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Texture {
        view,
        texture,
        width,
        height,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(max_dimension: u32, max_buffer_size: u64) -> wgpu::Limits {
        wgpu::Limits {
            max_texture_dimension_2d: max_dimension,
            max_buffer_size,
            ..wgpu::Limits::downlevel_defaults()
        }
    }

    #[test]
    fn a_frame_must_fit_the_texture_limit_and_the_read_back_buffer() {
        let device = limits(1000, 1_000_000);
        // 480 * 4 = 1920 bytes a row, padded to 2048.
        assert_eq!(frame_size(480, 270, &device), Ok((2048, 2048 * 270)));
        for (width, height) in [(0, 10), (10, 0), (1001, 10), (10, 1001)] {
            assert!(matches!(
                frame_size(width, height, &device),
                Err(OpenError::InvalidSize(_))
            ));
        }
        // Within the texture limit, past the buffer limit: 1000 rows of 4096 bytes.
        match frame_size(1000, 1000, &device) {
            Err(OpenError::InvalidSize(what)) => assert!(what.contains("read-back"), "{what}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(frame_size(1000, 244, &device), Ok((4096, 4096 * 244)));
    }
}
