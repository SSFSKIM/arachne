//! The compositor: textures of two kinds, a stack of layers with their effects, the frame read
//! back, and the object map.

use std::collections::HashMap;
use std::fmt;

use wgpu::util::DeviceExt;

use crate::device::{self, AdapterSummary, Door, ForkedProcess, NoGpu, Opened, Pin};
use crate::frame::{BILINEAR, Frame};
use crate::keys::{Key, image_key, painted_from_straight, painted_key};
use crate::object_map::{Footprint, from_coverage};
use crate::plan::{Canvas, Effect, Placed, Planned, Stack, plan};

/// Every texture and target a layer is drawn to: 8-bit values read and written as integers.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Uint;
/// The working textures between an effect's passes: 32-bit integers, 8 fractional bits.
const WIDE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Uint;
/// How many layers one coverage frame marks: a bit each, 32 bits a pixel.
const COVERAGE_BITS: usize = 32;

/// Why a frame was not drawn or a texture not taken. Nothing is drawn when a frame is refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DrawError {
    /// A layer names a key no texture is held under.
    UnknownKey(Key),
    /// The frame asks for something this build does not draw (`render.unsupported`).
    Unsupported(String),
    /// Malformed input: a key that is not its bytes' key, a size that does not fit.
    Invalid(String),
    /// A picture or a mask that is not of the size of the frame it is in: the outermost frame's
    /// size, or a window's own.
    SizeMismatch(String),
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
            DrawError::Invalid(what) | DrawError::SizeMismatch(what) => f.write_str(what),
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
    /// This process was forked after its parent first used this build: nothing was opened.
    ForkedProcess(ForkedProcess),
}

impl fmt::Display for OpenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpenError::NoGpu(no_gpu) => no_gpu.fmt(f),
            OpenError::InvalidSize(what) => f.write_str(what),
            OpenError::ForkedProcess(forked) => forked.fmt(f),
        }
    }
}

impl std::error::Error for OpenError {}

impl From<NoGpu> for OpenError {
    fn from(no_gpu: NoGpu) -> Self {
        OpenError::NoGpu(no_gpu)
    }
}

impl From<ForkedProcess> for OpenError {
    fn from(forked: ForkedProcess) -> Self {
        OpenError::ForkedProcess(forked)
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
    mode: u32,
    bit: u32,
    resampling: u32,
    padding: u32,
    /// A bilinear window's outline: top, bottom, left and right lines (`placement::edges`).
    edges: [[f32; 4]; 4],
}

// A layer pass's kinds.
const PICTURE: u32 = 0;
const QUAD: u32 = 1;
const FILL: u32 = 2;
// What a layer pass writes: the layer stacked over what is below, one bit of coverage, or the
// layer placed alone.
const OVER: u32 = 0;
const COVERAGE: u32 = 1;
const PLACE: u32 = 2;

impl LayerUniform {
    fn new(kind: u32, rows: [[f32; 4]; 3], size: [u32; 2]) -> Self {
        LayerUniform {
            rows,
            colour: [0; 4],
            size,
            kind,
            opacity: 255,
            mode: OVER,
            bit: 0,
            resampling: 0,
            padding: 0,
            edges: [[0.0; 4]; 4],
        }
    }

    fn fill(colour: [u8; 4]) -> Self {
        LayerUniform {
            colour: colour.map(u32::from),
            ..LayerUniform::new(FILL, [[0.0; 4]; 3], [0; 2])
        }
    }
}

/// One blur pass's parameters, laid out as `effects.wgsl`'s `Blur`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurUniform {
    direction: [i32; 2],
    radius: i32,
    shift: u32,
}

/// The finishing pass's parameters, laid out as `effects.wgsl`'s `Finish`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FinishUniform {
    masked: u32,
    invert: u32,
    padding: [u32; 2],
}

/// The textures the stacks of one level of windows and one size draw with: two accumulators a
/// stack ping-pongs between, and, when a layer of such a stack carries an effect, that effect's
/// working textures.
struct Level {
    accumulators: [Texture; 2],
    effects: Option<Scratch>,
}

/// An effect's working textures: the layer placed alone, and later finished, in 8 bits; and three
/// wide ones between the blur's and the feather's passes.
struct Scratch {
    layer: Texture,
    wide: [Texture; 3],
}

/// Frames of one size drawn on a Vulkan device with no window, and the textures it holds. Every
/// compositor of one door in a process draws on that door's one device (`device.rs`); its
/// textures, pipelines, working textures and read-back buffer are its own.
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
    compose: Pipeline,
    blur: Pipeline,
    finish: Pipeline,
    /// The working textures, by level and size: level 0 draws the frame at its size; level n + 1
    /// the windows of level n, each at its own. A draw keeps exactly those its frame uses.
    levels: HashMap<(usize, [u32; 2]), Level>,
    /// The most working textures held at once since the compositor opened.
    peak_working: usize,
    readback: wgpu::Buffer,
    padded_row: u32,
}

struct Pipeline {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
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
        Self::on(device::open(door)?, width, height, door == Door::Software)
    }

    /// A compositor drawing on `opened`, a device of the door `software` names.
    fn on(opened: Opened, width: u32, height: u32, software: bool) -> Result<Self, OpenError> {
        // Both limits are checked before any resource exists.
        let (padded_row, readback_size) = frame_size(width, height, &opened.limits)?;
        let max_dimension = opened.limits.max_texture_dimension_2d;
        let device = opened.device;
        let compose = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compose"),
            source: wgpu::ShaderSource::Wgsl(include_str!("compose.wgsl").into()),
        });
        let effects = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("effects"),
            source: wgpu::ShaderSource::Wgsl(include_str!("effects.wgsl").into()),
        });
        let compose = pipeline(&device, &compose, "fs", FORMAT);
        let blur = pipeline(&device, &effects, "blur_pass", WIDE);
        let finish = pipeline(&device, &effects, "finish_pass", FORMAT);
        let level = Level::new(&device, [width, height]);
        let peak_working = level.textures();
        let readback = readback_buffer(&device, readback_size);
        Ok(Compositor {
            device,
            queue: opened.queue,
            info: opened.info,
            software,
            width,
            height,
            max_dimension,
            images: HashMap::new(),
            painted: HashMap::new(),
            compose,
            blur,
            finish,
            levels: HashMap::from([((0, [width, height]), level)]),
            peak_working,
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

    /// The texture held under `key`, of either kind.
    fn held(&self, key: &Key) -> Option<&Texture> {
        self.images.get(key).or_else(|| self.painted.get(key))
    }

    /// The texture under a key the frame's plan found held: no texture is released while a frame
    /// is drawn.
    fn planned_texture(&self, key: &Key) -> &Texture {
        self.held(key).expect("a key the plan found is held")
    }

    /// The frame checked whole and resolved, and its working textures made ready: those of a
    /// level and size it does not use are freed first, then those it lacks are made. So the
    /// working textures held are always those of one frame, the last one planned; a refused
    /// frame changes none.
    fn prepare(&mut self, frame: &Frame) -> Result<Stack, DrawError> {
        let size_of = |key: &Key| self.held(key).map(|t| (t.width, t.height));
        let canvas = Canvas {
            width: self.width,
            height: self.height,
            max_dimension: self.max_dimension,
            size_of: &size_of,
        };
        let stack = plan(frame, &canvas, 0)?;
        let mut used = HashMap::new();
        uses(&stack, 0, [self.width, self.height], &mut used);
        self.levels.retain(|at, level| match used.get(at) {
            Some(&effects) => {
                if !effects {
                    level.effects = None;
                }
                true
            }
            None => false,
        });
        let device = &self.device;
        for (&(nesting, size), &effects) in &used {
            let level = self
                .levels
                .entry((nesting, size))
                .or_insert_with(|| Level::new(device, size));
            if effects && level.effects.is_none() {
                level.effects = Some(Scratch::new(device, size));
            }
            let held: usize = self.levels.values().map(Level::textures).sum();
            self.peak_working = self.peak_working.max(held);
        }
        Ok(stack)
    }

    /// The working textures held now, and the most held at once since the compositor opened
    /// (`testing::working_textures`).
    pub(crate) fn working_textures(&self) -> (usize, usize) {
        let held: usize = self.levels.values().map(Level::textures).sum();
        (held, self.peak_working)
    }

    /// The working textures of `level` at `size`, which the frame being drawn was prepared with.
    fn level(&self, level: usize, size: [u32; 2]) -> &Level {
        self.levels
            .get(&(level, size))
            .expect("a frame is prepared before it is drawn")
    }

    /// Draws the frame: tight RGBA8 rows, top first, painted. The whole frame is checked before
    /// anything is drawn; a refused frame draws nothing. A draw holds the compositor exclusively
    /// from its first pass to the read-back buffer's unmapping: its working textures and that
    /// buffer are one frame's at a time.
    pub fn draw(&mut self, frame: &Frame) -> Result<Vec<u8>, DrawError> {
        let stack = self.prepare(frame)?;
        let submission = self.submit_stack(&stack, &self.readback);
        self.collect(&self.readback, submission)
    }

    /// Checks and submits the frame as [`Compositor::draw`] does, copying it into `buffer` (one
    /// of [`Compositor::readback_buffer`]'s) without waiting: [`Compositor::collect`] takes it
    /// out. For the read-back benchmark, which copies one frame out while the next draws; the
    /// public drawing call stays `draw`.
    #[cfg(any(test, feature = "bench"))]
    pub(crate) fn submit(
        &mut self,
        frame: &Frame,
        buffer: &wgpu::Buffer,
    ) -> Result<wgpu::SubmissionIndex, DrawError> {
        let stack = self.prepare(frame)?;
        Ok(self.submit_stack(&stack, buffer))
    }

    /// A buffer one frame can be copied into and read back from, apart from the compositor's own.
    #[cfg(any(test, feature = "bench"))]
    pub(crate) fn readback_buffer(&self) -> wgpu::Buffer {
        readback_buffer(
            &self.device,
            u64::from(self.padded_row) * u64::from(self.height),
        )
    }

    /// Records a planned stack's passes and the copy of its frame into `buffer`, and submits them.
    fn submit_stack(&self, stack: &Stack, buffer: &wgpu::Buffer) -> wgpu::SubmissionIndex {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let drawn = self.encode_stack(&mut encoder, stack, 0, [self.width, self.height]);
        self.copy_out(&mut encoder, drawn, buffer);
        self.queue.submit(Some(encoder.finish()))
    }

    /// One footprint per layer that names an object, in the frame's order: the pixels the layer
    /// covers drawn with no effect, and how far its effects spread it. The frame is checked as
    /// [`Compositor::draw`] checks it. Coverage is drawn on the device, one bit a layer, 32 layers
    /// to a read-back; a window's stack is composed for it first.
    pub fn object_map(&mut self, frame: &Frame) -> Result<Vec<Footprint>, DrawError> {
        let stack = self.prepare(frame)?;
        let named: Vec<(usize, &Planned, &str)> = stack
            .layers
            .iter()
            .enumerate()
            .filter_map(|(index, layer)| Some((index, layer, layer.object_id.as_deref()?)))
            .collect();
        let mut footprints = Vec::with_capacity(named.len());
        for batch in named.chunks(COVERAGE_BITS) {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let frame_size = [self.width, self.height];
            let accumulators = &self.level(0, frame_size).accumulators;
            let [first, second] = accumulators;
            self.layer_pass(
                &mut encoder,
                LayerUniform::fill([0; 4]),
                second,
                second,
                first,
            );
            let mut current = 0;
            for (bit, (_, layer, _)) in batch.iter().enumerate() {
                let Some((source, mut uniform)) =
                    self.placed(&mut encoder, &layer.placed, 0, frame_size)
                else {
                    continue;
                };
                uniform.mode = COVERAGE;
                uniform.bit = bit as u32;
                uniform.opacity = layer.opacity;
                let (below, target) = (&accumulators[current], &accumulators[1 - current]);
                self.layer_pass(&mut encoder, uniform, below, source, target);
                current = 1 - current;
            }
            let coverage = self.read_back(encoder, &accumulators[current])?;
            for (bit, (index, layer, object_id)) in batch.iter().enumerate() {
                footprints.push(from_coverage(
                    &coverage,
                    self.width,
                    bit,
                    object_id.to_string(),
                    *index,
                    layer.grow_px,
                ));
            }
        }
        Ok(footprints)
    }

    /// Records the passes of a stack of `size` with the accumulators of `level` at that size, and
    /// returns the one it ends in.
    fn encode_stack(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        stack: &Stack,
        level: usize,
        size: [u32; 2],
    ) -> &Texture {
        let accumulators = &self.level(level, size).accumulators;
        let [first, second] = accumulators;
        self.layer_pass(
            encoder,
            LayerUniform::fill(stack.background),
            second,
            second,
            first,
        );
        let mut current = 0;
        for layer in &stack.layers {
            let Some((placed, uniform)) = self.placed(encoder, &layer.placed, level, size) else {
                continue;
            };
            let (source, mut uniform) = match &layer.effect {
                Some(effect) => {
                    let finished =
                        self.encode_effect(encoder, level, size, placed, uniform, effect);
                    (finished, LayerUniform::new(PICTURE, [[0.0; 4]; 3], size))
                }
                None => (placed, uniform),
            };
            uniform.opacity = layer.opacity;
            let (below, target) = (&accumulators[current], &accumulators[1 - current]);
            self.layer_pass(encoder, uniform, below, source, target);
            current = 1 - current;
        }
        &accumulators[current]
    }

    /// The texture a layer of a stack of `frame_size` at `level` draws from and the uniform that
    /// places it; `None` when it places nothing. A window's stack is composed first, one level
    /// down, at the window's size.
    fn placed(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        placed: &Placed,
        level: usize,
        frame_size: [u32; 2],
    ) -> Option<(&Texture, LayerUniform)> {
        match placed {
            Placed::Nothing => None,
            Placed::Picture(key) => Some((
                self.planned_texture(key),
                LayerUniform::new(PICTURE, [[0.0; 4]; 3], frame_size),
            )),
            Placed::Quad { key, rows } => {
                let texture = self.planned_texture(key);
                let size = [texture.width, texture.height];
                Some((texture, LayerUniform::new(QUAD, *rows, size)))
            }
            Placed::Window {
                stack,
                rows,
                size,
                resampling,
                edges,
            } => {
                let composed = self.encode_stack(encoder, stack, level + 1, *size);
                let mut uniform = LayerUniform::new(QUAD, *rows, *size);
                uniform.resampling = u32::from(*resampling == BILINEAR);
                uniform.edges = *edges;
                Some((composed, uniform))
            }
        }
    }

    /// Records one layer's effects with the working textures of `level` at `size`, the size of
    /// its stack, and returns the finished layer, a picture of that size in 8 bits. The layer is
    /// placed alone, blurred along rows then columns into wide values, multiplied by its mask's
    /// alpha (feathered the same way), and rounded to 8 bits once (`effects.wgsl`).
    fn encode_effect(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        level: usize,
        size: [u32; 2],
        source: &Texture,
        mut place: LayerUniform,
        effect: &Effect,
    ) -> &Texture {
        let scratch = self
            .level(level, size)
            .effects
            .as_ref()
            .expect("room is made for every effect before it is drawn");
        let [rows, blurred, feathered] = &scratch.wide;
        place.mode = PLACE;
        self.layer_pass(encoder, place, source, source, &scratch.layer);
        self.blur_pass(encoder, &effect.blur, [1, 0], 8, &scratch.layer, rows);
        self.blur_pass(encoder, &effect.blur, [0, 1], 16, rows, blurred);
        let mut finish = FinishUniform {
            masked: 0,
            invert: 0,
            padding: [0; 2],
        };
        if let Some(mask) = &effect.mask {
            let alpha = self.planned_texture(&mask.key);
            self.blur_pass(encoder, &mask.feather, [1, 0], 8, alpha, rows);
            self.blur_pass(encoder, &mask.feather, [0, 1], 16, rows, feathered);
            finish.masked = 1;
            finish.invert = u32::from(mask.invert);
        }
        let mask = if finish.masked == 1 {
            feathered
        } else {
            blurred
        };
        let bind_group = self.bind_group(
            &self.finish,
            [
                (
                    3,
                    self.uniform(bytemuck::bytes_of(&finish))
                        .as_entire_binding(),
                ),
                (4, wgpu::BindingResource::TextureView(&blurred.view)),
                (5, wgpu::BindingResource::TextureView(&mask.view)),
            ],
        );
        self.run(encoder, &self.finish, &bind_group, &scratch.layer);
        &scratch.layer
    }

    /// One pass of `effects.wgsl`'s blur: `weights` along `direction`, keeping 8 fractional bits
    /// over a source with `shift - 8`.
    fn blur_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        weights: &[u32],
        direction: [i32; 2],
        shift: u32,
        source: &Texture,
        target: &Texture,
    ) {
        let uniform = BlurUniform {
            direction,
            radius: (weights.len() - 1) as i32,
            shift,
        };
        let uniform = self.uniform(bytemuck::bytes_of(&uniform));
        let weights = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(weights),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let bind_group = self.bind_group(
            &self.blur,
            [
                (0, uniform.as_entire_binding()),
                (1, weights.as_entire_binding()),
                (2, wgpu::BindingResource::TextureView(&source.view)),
            ],
        );
        self.run(encoder, &self.blur, &bind_group, target);
    }

    /// One pass of `compose.wgsl`: `source` placed by `uniform`, written to `target` as
    /// `uniform.mode` says, over `below`.
    fn layer_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        uniform: LayerUniform,
        below: &Texture,
        source: &Texture,
        target: &Texture,
    ) {
        let uniform = self.uniform(bytemuck::bytes_of(&uniform));
        let bind_group = self.bind_group(
            &self.compose,
            [
                (0, uniform.as_entire_binding()),
                (1, wgpu::BindingResource::TextureView(&below.view)),
                (2, wgpu::BindingResource::TextureView(&source.view)),
            ],
        );
        self.run(encoder, &self.compose, &bind_group, target);
    }

    fn uniform(&self, contents: &[u8]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents,
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    fn bind_group<'a>(
        &self,
        pipeline: &Pipeline,
        entries: impl IntoIterator<Item = (u32, wgpu::BindingResource<'a>)>,
    ) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry<'a>> = entries
            .into_iter()
            .map(|(binding, resource)| wgpu::BindGroupEntry { binding, resource })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.layout,
            entries: &entries,
        })
    }

    /// Records one full-frame pass of `pipeline` writing `target`.
    fn run(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &Pipeline,
        bind_group: &wgpu::BindGroup,
        target: &Texture,
    ) {
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
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Copies the frame out: to the read-back buffer, mapped, waited for, rows copied out.
    fn read_back(
        &self,
        mut encoder: wgpu::CommandEncoder,
        source: &Texture,
    ) -> Result<Vec<u8>, DrawError> {
        self.copy_out(&mut encoder, source, &self.readback);
        let submission = self.queue.submit(Some(encoder.finish()));
        self.collect(&self.readback, submission)
    }

    /// Records the copy of a frame-sized texture into a read-back buffer.
    fn copy_out(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &Texture,
        buffer: &wgpu::Buffer,
    ) {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &source.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_row),
                    rows_per_image: Some(self.height),
                },
            },
            extent(self.width, self.height),
        );
    }

    /// Waits for `submission`, which copied a frame into `buffer`, then maps the buffer and copies
    /// the frame's rows out: tight RGBA8 rows, top first.
    pub(crate) fn collect(
        &self,
        buffer: &wgpu::Buffer,
        submission: wgpu::SubmissionIndex,
    ) -> Result<Vec<u8>, DrawError> {
        let slice = buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
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
        buffer.unmap();
        Ok(rgba)
    }
}

/// A buffer of `size` bytes a frame is copied into and mapped for reading.
fn readback_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
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
    texture_of_format(device, width, height, usage, FORMAT)
}

/// A texture a pass writes and a later pass reads.
fn working(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> Texture {
    let usage = wgpu::TextureUsages::RENDER_ATTACHMENT
        | wgpu::TextureUsages::TEXTURE_BINDING
        | wgpu::TextureUsages::COPY_SRC;
    texture_of_format(device, width, height, usage, format)
}

impl Level {
    fn new(device: &wgpu::Device, [width, height]: [u32; 2]) -> Self {
        Level {
            accumulators: [
                working(device, width, height, FORMAT),
                working(device, width, height, FORMAT),
            ],
            effects: None,
        }
    }

    /// How many textures it holds: two accumulators, and four for an effect.
    fn textures(&self) -> usize {
        2 + if self.effects.is_some() { 4 } else { 0 }
    }
}

/// The levels and sizes whose working textures `stack`, drawn at `level` and `size`, uses, each
/// with whether a layer of a stack there carries an effect.
fn uses(stack: &Stack, level: usize, size: [u32; 2], used: &mut HashMap<(usize, [u32; 2]), bool>) {
    let effects = used.entry((level, size)).or_insert(false);
    *effects |= stack.layers.iter().any(|l| l.effect.is_some());
    for layer in &stack.layers {
        if let Placed::Window { stack, size, .. } = &layer.placed {
            uses(stack, level + 1, *size, used);
        }
    }
}

impl Scratch {
    fn new(device: &wgpu::Device, [width, height]: [u32; 2]) -> Self {
        Scratch {
            layer: working(device, width, height, FORMAT),
            wide: [
                working(device, width, height, WIDE),
                working(device, width, height, WIDE),
                working(device, width, height, WIDE),
            ],
        }
    }
}

/// A full-frame pipeline of `module`'s vertex stage `vs` and fragment stage `fragment`, writing
/// `format` with no blending; its bind group's layout is derived from what the stages use.
fn pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    fragment: &str,
    format: wgpu::TextureFormat,
) -> Pipeline {
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(fragment),
        layout: None,
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fragment),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
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
    Pipeline { pipeline, layout }
}

fn texture_of_format(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    usage: wgpu::TextureUsages,
    format: wgpu::TextureFormat,
) -> Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: extent(width, height),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
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
