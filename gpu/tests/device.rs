//! Which adapters draw, what the pin says, and how the launch mode is read.

mod common;

use arachne_gpu::OpenError;
use arachne_gpu::{Compositor, WGPU_VERSION};
use common::{compositor, is_child, mode_from, no_gpu, open, run_child, software_icd};

/// The machines below are made by running this binary again with the Vulkan loader shown only a
/// software adapter, or with no loader it can load.
fn assert_child_passed(test: &str, env: &[(&str, &str)]) {
    let output = run_child(test, env);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && stdout.contains("1 passed"),
        "the child {test} failed:\n{stdout}\n{stderr}"
    );
}

#[test]
fn the_ordinary_door_refuses_a_machine_whose_only_adapter_is_software() {
    if !is_child() {
        let icd = software_icd();
        assert_child_passed(
            "the_ordinary_door_refuses_a_machine_whose_only_adapter_is_software",
            &[("VK_DRIVER_FILES", &icd), ("VK_ICD_FILENAMES", &icd)],
        );
        return;
    }
    let seen = arachne_gpu::adapters();
    assert!(
        !seen.is_empty() && seen.iter().all(|a| a.device_type == "Cpu"),
        "the child sees {seen:?}"
    );
    let refused = no_gpu(Compositor::new(64, 64));
    assert_eq!(refused.seen, seen);
    let message = refused.to_string();
    for adapter in &seen {
        assert!(
            message.contains(&adapter.name),
            "{message} does not name {}",
            adapter.name
        );
    }
    // The testing door draws there.
    let software = arachne_gpu::testing::software_compositor(64, 64).unwrap();
    assert!(software.software());
    assert_eq!(software.adapter().device_type, "Cpu");
}

#[test]
fn the_ordinary_door_refuses_a_machine_with_no_vulkan_loader() {
    if !is_child() {
        // A directory whose libvulkan.so.1 is no library: the loader cannot be loaded.
        let dir =
            std::env::temp_dir().join(format!("arachne-gpu-no-loader-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("libvulkan.so.1"), b"").unwrap();
        let dir = dir.display().to_string();
        assert_child_passed(
            "the_ordinary_door_refuses_a_machine_with_no_vulkan_loader",
            &[("LD_LIBRARY_PATH", &dir)],
        );
        return;
    }
    assert_eq!(arachne_gpu::adapters(), vec![]);
    let refused = no_gpu(Compositor::new(64, 64));
    assert!(refused.seen.is_empty());
    assert!(refused.to_string().contains("no Vulkan adapter was found"));
    assert!(arachne_gpu::testing::software_compositor(64, 64).is_err());
}

#[test]
fn the_pin_names_the_device_and_the_locked_wgpu() {
    let c = compositor(32, 32);
    let pin = c.pin();
    let adapter = c.adapter();
    assert_eq!(pin.adapter, adapter.name);
    assert_eq!(pin.backend, "Vulkan");
    assert_eq!(pin.driver, adapter.driver_info);
    // Read from the lock file here, apart from how the build reads it.
    let lock = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.lock")).unwrap();
    let packages: Vec<&str> = lock.split("[[package]]").collect();
    let wgpu: Vec<String> = packages
        .iter()
        .filter(|p| p.contains("\nname = \"wgpu\"\n"))
        .filter_map(|p| p.split("\nversion = \"").nth(1))
        .filter_map(|v| v.split('"').next().map(String::from))
        .collect();
    assert_eq!(wgpu, vec![WGPU_VERSION.to_string()]);
    assert_eq!(pin.wgpu, WGPU_VERSION);
    assert_eq!(WGPU_VERSION, "30.0.2");
}

#[test]
fn with_no_launch_mode_the_drawing_tests_fail() {
    assert!(mode_from(None, None).is_err());
    assert!(mode_from(Some("gpu"), None).is_err());
    assert!(mode_from(Some("hardware"), None).is_err());
    assert_eq!(
        mode_from(Some("software"), None),
        Ok(common::Mode::Software)
    );
    let pin = r#"{"adapter": "A", "backend": "Vulkan", "driver": "1.2", "wgpu": "30.0.2"}"#;
    assert!(matches!(
        mode_from(Some("hardware"), Some(pin)),
        Ok(common::Mode::Hardware(_))
    ));
}

#[test]
fn a_frame_size_the_device_cannot_draw_is_refused_as_invalid() {
    // No device has a side of a million pixels; the read-back limit is probed with set limits in
    // the crate's own unit tests, so this runs on any adapter.
    for (width, height) in [(0, 10), (10, 0), (1_000_000, 10)] {
        assert!(
            matches!(open(width, height), Err(OpenError::InvalidSize(_))),
            "{width} x {height}"
        );
    }
}

#[test]
fn a_compositor_moves_to_another_thread() {
    let mut c = compositor(16, 16);
    let drawn = std::thread::spawn(move || {
        c.draw(&arachne_gpu::Frame {
            background: [1, 2, 3, 4],
            layers: vec![],
        })
        .unwrap()
    })
    .join()
    .unwrap();
    assert!(drawn.as_chunks::<4>().0.iter().all(|p| *p == [1, 2, 3, 4]));
}
