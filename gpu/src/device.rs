//! Finding and opening a device: Vulkan only, no window, no surface.
//!
//! A process has one Vulkan instance, and opens one device for each door, which every compositor
//! opened through that door shares; a compositor holds its own textures, pipelines and read-back
//! buffer on it. A Vulkan driver's cost per instance and per device is not ours to bound: on one
//! card's driver, a process that made an instance and a device for each compositor saw the card
//! no more after a few dozen, dropped or kept.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

/// What an adapter reports about itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterSummary {
    pub name: String,
    pub backend: String,
    /// `DiscreteGpu`, `IntegratedGpu`, `VirtualGpu`, `Cpu` (a software adapter) or `Other`.
    pub device_type: String,
    pub driver: String,
    pub driver_info: String,
    pub vendor: u32,
    pub device: u32,
}

impl AdapterSummary {
    pub(crate) fn of(info: &wgpu::AdapterInfo) -> Self {
        AdapterSummary {
            name: info.name.clone(),
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
            driver: info.driver.clone(),
            driver_info: info.driver_info.clone(),
            vendor: info.vendor,
            device: info.device,
        }
    }
}

impl fmt::Display for AdapterSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "name=\"{}\" backend={} device_type={} vendor=0x{:04x} device=0x{:04x} driver=\"{}\" driver_info=\"{}\"",
            self.name,
            self.backend,
            self.device_type,
            self.vendor,
            self.device,
            self.driver,
            self.driver_info
        )
    }
}

/// The device a compositor draws on, as every GPU result records it: the adapter's name, the
/// backend, the driver version string the adapter reports, and the locked wgpu version.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pin {
    pub adapter: String,
    pub backend: String,
    pub driver: String,
    pub wgpu: String,
}

/// No adapter this build may draw on. Lists every adapter it saw.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoGpu {
    /// What was wanted: "a GPU" for the ordinary door, "a software adapter" for the testing door.
    pub wanted: &'static str,
    pub seen: Vec<AdapterSummary>,
    /// Why the chosen adapter could not be opened, when one was chosen.
    pub failure: Option<String>,
}

impl fmt::Display for NoGpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.failure {
            Some(failure) => write!(f, "{} could not be opened: {failure}", self.wanted)?,
            None => write!(f, "no {} on Vulkan", self.wanted)?,
        }
        if self.seen.is_empty() {
            write!(f, "; no Vulkan adapter was found")
        } else {
            write!(f, "; adapters seen:")?;
            for adapter in &self.seen {
                write!(f, " [{adapter}]")?;
            }
            Ok(())
        }
    }
}

impl std::error::Error for NoGpu {}

/// Which adapters a constructor accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Door {
    /// Discrete, integrated or virtual GPUs, in that order of preference.
    Ordinary,
    /// A software (CPU-type) adapter only: the testing door.
    Software,
}

impl Door {
    fn rank(self, device_type: wgpu::DeviceType) -> Option<u8> {
        use wgpu::DeviceType::*;
        match (self, device_type) {
            (Door::Ordinary, DiscreteGpu) => Some(0),
            (Door::Ordinary, IntegratedGpu) => Some(1),
            (Door::Ordinary, VirtualGpu) => Some(2),
            (Door::Software, Cpu) => Some(0),
            _ => None,
        }
    }

    fn wanted(self) -> &'static str {
        match self {
            Door::Ordinary => "GPU (a discrete, integrated or virtual adapter)",
            Door::Software => "software adapter",
        }
    }
}

/// The process's one instance: the Vulkan backend alone, with no display handle. Nothing in the
/// environment chooses another backend or the fallback adapter.
fn instance() -> &'static wgpu::Instance {
    static INSTANCE: OnceLock<wgpu::Instance> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::VULKAN;
        wgpu::Instance::new(descriptor)
    })
}

fn enumerate() -> Vec<wgpu::Adapter> {
    pollster::block_on(instance().enumerate_adapters(wgpu::Backends::VULKAN))
}

/// Every adapter this machine offers on Vulkan, of any device type.
pub fn adapters() -> Vec<AdapterSummary> {
    enumerate()
        .iter()
        .map(|adapter| AdapterSummary::of(&adapter.get_info()))
        .collect()
}

/// An opened device and the adapter it is on, shared by every compositor of one door.
#[derive(Clone)]
pub(crate) struct Opened {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    /// Set when the device is lost: the next open opens another.
    lost: Arc<AtomicBool>,
}

impl Door {
    /// The device this door has opened in this process, if any.
    fn opened(self) -> &'static Mutex<Option<Opened>> {
        static ORDINARY: Mutex<Option<Opened>> = Mutex::new(None);
        static SOFTWARE: Mutex<Option<Opened>> = Mutex::new(None);
        match self {
            Door::Ordinary => &ORDINARY,
            Door::Software => &SOFTWARE,
        }
    }
}

/// The device of the most preferred adapter the door accepts: the one this door opened earlier in
/// the process, unless it was lost, and otherwise a new one, kept for the next open.
pub(crate) fn open(door: Door) -> Result<Opened, NoGpu> {
    let mut slot = door.opened().lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(held) = slot.as_ref()
        && !held.lost.load(Ordering::Acquire)
    {
        return Ok(held.clone());
    }
    let fresh = request(door)?;
    *slot = Some(fresh.clone());
    Ok(fresh)
}

/// Opens a new device on the most preferred adapter the door accepts.
fn request(door: Door) -> Result<Opened, NoGpu> {
    let all = enumerate();
    let seen: Vec<AdapterSummary> = all
        .iter()
        .map(|adapter| AdapterSummary::of(&adapter.get_info()))
        .collect();
    let chosen = all
        .into_iter()
        .filter_map(|adapter| {
            door.rank(adapter.get_info().device_type)
                .map(|rank| (rank, adapter))
        })
        .min_by_key(|(rank, _)| *rank);
    let Some((_, adapter)) = chosen else {
        return Err(NoGpu {
            wanted: door.wanted(),
            seen,
            failure: None,
        });
    };
    let limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());
    let requested = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("arachne-gpu"),
        required_features: wgpu::Features::empty(),
        required_limits: limits.clone(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }));
    match requested {
        Ok((device, queue)) => {
            let lost = Arc::new(AtomicBool::new(false));
            let flag = Arc::clone(&lost);
            device.set_device_lost_callback(move |_, _| flag.store(true, Ordering::Release));
            Ok(Opened {
                device,
                queue,
                info: adapter.get_info(),
                limits,
                lost,
            })
        }
        Err(err) => Err(NoGpu {
            wanted: door.wanted(),
            seen,
            failure: Some(err.to_string()),
        }),
    }
}
