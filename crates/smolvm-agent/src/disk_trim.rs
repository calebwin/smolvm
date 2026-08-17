//! Periodic in-guest `fstrim` so a machine's disk footprint tracks the data
//! actually live inside it.
//!
//! The guest's storage and rootfs-overlay disks are sparse files on the host.
//! They grow as the guest writes, but deleting files inside the guest only
//! frees blocks in the guest filesystem — the host sparse file stays at its
//! high-water mark until the freed blocks are discarded. virtio-blk passes
//! `FITRIM` through to the host, punching holes back into the sparse file, so a
//! periodic `fstrim` makes disk usage shrink symmetrically with the guest's
//! own usage — the disk counterpart to the host-side idle memory reclaim.
//!
//! `fstrim -a` trims every mounted filesystem that supports discard (the ext4
//! storage disk at `/storage` and the ext4 rootfs overlay) and skips the rest
//! (virtiofs, overlay, tmpfs) without error, so it needs no mount enumeration.
//!
//! Best-effort throughout: a failed trim is logged and retried next tick; it
//! never touches the workload. Tunable via `SMOLVM_DISK_TRIM`:
//!   - unset            → default interval
//!   - `off` / `0`      → disabled
//!   - `<minutes>`      → custom interval (floored at 1)

use std::time::Duration;

const DEFAULT_TRIM_MINUTES: u64 = 10;

/// Minutes between trims, or `None` when disabled.
fn trim_minutes() -> Option<u64> {
    match std::env::var("SMOLVM_DISK_TRIM") {
        Ok(v) => {
            let v = v.trim();
            if v.eq_ignore_ascii_case("off") || v == "0" {
                None
            } else {
                Some(v.parse::<u64>().unwrap_or(DEFAULT_TRIM_MINUTES).max(1))
            }
        }
        Err(_) => Some(DEFAULT_TRIM_MINUTES),
    }
}

/// Spawn the background trim thread. No-op when disabled.
pub fn spawn() {
    let Some(minutes) = trim_minutes() else {
        return;
    };
    let _ = std::thread::Builder::new()
        .name("disk-trim".into())
        .spawn(move || run(Duration::from_secs(minutes * 60)));
}

fn run(interval: Duration) {
    loop {
        std::thread::sleep(interval);
        trim_once();
    }
}

/// Run one `fstrim -a`. Logged best-effort; never propagates failure.
fn trim_once() {
    match std::process::Command::new("fstrim").arg("-a").output() {
        Ok(out) if out.status.success() => {
            tracing::debug!("disk trim: fstrim -a completed");
        }
        Ok(out) => {
            tracing::debug!(
                stderr = %String::from_utf8_lossy(&out.stderr).trim(),
                "disk trim: fstrim -a returned non-zero"
            );
        }
        Err(e) => {
            tracing::debug!(error = %e, "disk trim: fstrim not run");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{trim_minutes, DEFAULT_TRIM_MINUTES};

    fn with_env<T>(val: Option<&str>, f: impl FnOnce() -> T) -> T {
        match val {
            Some(v) => std::env::set_var("SMOLVM_DISK_TRIM", v),
            None => std::env::remove_var("SMOLVM_DISK_TRIM"),
        }
        let out = f();
        std::env::remove_var("SMOLVM_DISK_TRIM");
        out
    }

    #[test]
    fn trim_interval_parsing() {
        assert_eq!(with_env(None, trim_minutes), Some(DEFAULT_TRIM_MINUTES));
        assert_eq!(with_env(Some("off"), trim_minutes), None);
        assert_eq!(with_env(Some("0"), trim_minutes), None);
        assert_eq!(with_env(Some("30"), trim_minutes), Some(30));
        // A garbage value falls back to the default rather than disabling.
        assert_eq!(
            with_env(Some("nonsense"), trim_minutes),
            Some(DEFAULT_TRIM_MINUTES)
        );
        // Sub-1 is floored to 1 (a value of 1 stays 1).
        assert_eq!(with_env(Some("1"), trim_minutes), Some(1));
    }
}
