//! Mount S3 buckets inside the workload container's mount namespace.
//!
//! A remote volume has to appear in the namespace the workload actually lives
//! in — the same reasoning as [`crate::nsfile`]: a mount made in the agent's
//! namespace is invisible to the container, and to `exec`/`shell` sessions that
//! join it.
//!
//! `setns(CLONE_NEWMNT)` is refused for a multithreaded caller and the agent
//! runs a threaded async runtime, so each mount runs in a fresh single-threaded
//! re-exec of this binary (`smolvm-agent s3-mount …`), dispatched at the top of
//! `main` before any thread starts. The helper then *stays alive* serving FUSE
//! for as long as the mount exists — unlike the ns-file helper, which does one
//! operation and exits.
//!
//! Nothing is required of the container image: no rclone, no fuse3, no
//! `fusermount3`. The helper opens `/dev/fuse` (creating the node if the image
//! lacks one) and calls `mount(2)` directly as root, so a bucket can be mounted
//! into a distroless or scratch image that could not install a helper at all.

use std::time::Duration;

use smolvm_s3fs::{s3, sigv4, MountOptions};

const HELPER_ARG: &str = "s3-mount";

/// Everything the helper needs, passed as one JSON argv entry so credentials
/// never appear in a separate process's environment or on a shared command
/// line more than once.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct MountSpec {
    pub endpoint: String,
    pub region: String,
    pub bucket: String,
    #[serde(default)]
    pub prefix: String,
    pub mountpoint: String,
    #[serde(default)]
    pub read_only: bool,
    #[serde(default)]
    pub access_key_id: Option<String>,
    #[serde(default)]
    pub secret_access_key: Option<String>,
    #[serde(default)]
    pub session_token: Option<String>,
}

impl MountSpec {
    fn credentials(&self) -> Option<sigv4::Credentials> {
        match (&self.access_key_id, &self.secret_access_key) {
            // Both halves or neither: a half-configured key would sign
            // requests that always fail, which is harder to diagnose than
            // an explicit anonymous request.
            (Some(k), Some(s)) if !k.is_empty() && !s.is_empty() => Some(sigv4::Credentials {
                access_key_id: k.clone(),
                secret_access_key: s.clone(),
                session_token: self.session_token.clone(),
            }),
            _ => None,
        }
    }
}

/// Whether this process was re-exec'd as the mount helper.
///
/// Checked before the async runtime starts, because `setns(CLONE_NEWMNT)`
/// requires a single-threaded caller.
pub fn helper_requested() -> bool {
    std::env::args().nth(1).as_deref() == Some(HELPER_ARG)
}

/// Entry point for `smolvm-agent s3-mount <pid> <spec-json>`.
///
/// Runs until the mount is torn down; the caller supervises it as a child.
pub fn run_helper() -> i32 {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("s3-mount: usage: s3-mount <container-pid> <spec-json>");
        return 2;
    }
    let Ok(pid) = args[2].parse::<u32>() else {
        eprintln!("s3-mount: bad pid");
        return 2;
    };
    let spec: MountSpec = match serde_json::from_str(&args[3]) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("s3-mount: bad spec: {e}");
            return 2;
        }
    };

    // pid 0 means "already in the right namespace" (used by tests and by the
    // no-container case); anything else joins the workload's namespace first.
    if pid != 0 {
        if let Err(e) = enter_mount_namespace(pid) {
            eprintln!("s3-mount: {e}");
            return 1;
        }
    }

    let cfg = s3::Config {
        endpoint: spec.endpoint.clone(),
        region: spec.region.clone(),
        bucket: spec.bucket.clone(),
        prefix: spec.prefix.clone(),
        credentials: spec.credentials(),
        // Path-style works against AWS and every S3-compatible server; virtual
        // host style would need per-provider DNS assumptions.
        path_style: true,
        timeout: Duration::from_secs(60),
    };
    let opts = MountOptions {
        mountpoint: spec.mountpoint.clone(),
        read_only: spec.read_only,
        allow_other: true,
        // Staging lives on the container's own writable layer so a large write
        // is bounded by the machine's disk, not by RAM.
        scratch_dir: std::path::PathBuf::from("/var/tmp/smolvm-s3fs"),
    };

    eprintln!(
        "s3-mount: mounting s3://{}/{} at {}",
        spec.bucket, spec.prefix, spec.mountpoint
    );
    #[cfg(target_os = "linux")]
    {
        match smolvm_s3fs::mount(cfg, opts) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("s3-mount: mount failed: {e}");
                1
            }
        }
    }
    // The agent only ever runs in a Linux guest; this keeps host-side unit
    // tests (which build the crate on macOS) compiling.
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (cfg, opts);
        eprintln!("s3-mount: mounting is Linux-only");
        1
    }
}

#[cfg(target_os = "linux")]
fn enter_mount_namespace(pid: u32) -> Result<(), String> {
    use std::os::unix::io::AsRawFd;
    let ns = format!("/proc/{pid}/ns/mnt");
    let file = std::fs::File::open(&ns).map_err(|e| format!("open {ns}: {e}"))?;
    // SAFETY: `fd` is a live descriptor for a mount-namespace file; setns only
    // reads it and this process is single-threaded (re-exec'd for that reason).
    let rc = unsafe { libc::setns(file.as_raw_fd(), libc::CLONE_NEWNS) };
    if rc != 0 {
        return Err(format!("setns({ns}): {}", std::io::Error::last_os_error()));
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn enter_mount_namespace(_pid: u32) -> Result<(), String> {
    Err("mount namespaces are Linux-only".to_string())
}

/// Spawn a mount helper for `spec` against the container at `pid`.
///
/// Returns the child so the caller can supervise it; the mount lives exactly as
/// long as this process does.
pub fn spawn(pid: u32, spec: &MountSpec) -> Result<std::process::Child, String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let json = serde_json::to_string(spec).map_err(|e| format!("encode spec: {e}"))?;
    std::process::Command::new(exe)
        .arg(HELPER_ARG)
        .arg(pid.to_string())
        .arg(json)
        .stdin(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("spawn s3-mount helper: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> MountSpec {
        MountSpec {
            endpoint: "http://127.0.0.1:9000".into(),
            region: "us-east-1".into(),
            bucket: "b".into(),
            prefix: "p".into(),
            mountpoint: "/mnt/x".into(),
            read_only: false,
            access_key_id: Some("k".into()),
            secret_access_key: Some("s".into()),
            session_token: None,
        }
    }

    #[test]
    fn a_spec_survives_the_argv_round_trip() {
        let s = spec();
        let json = serde_json::to_string(&s).unwrap();
        let back: MountSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.bucket, "b");
        assert_eq!(back.mountpoint, "/mnt/x");
        assert_eq!(back.access_key_id.as_deref(), Some("k"));
    }

    // A half-supplied key pair would sign every request with a broken
    // credential; anonymous is both correct and far easier to diagnose.
    #[test]
    fn credentials_need_both_halves_or_none() {
        assert!(spec().credentials().is_some());

        let mut half = spec();
        half.secret_access_key = None;
        assert!(half.credentials().is_none());

        let mut empty = spec();
        empty.access_key_id = Some(String::new());
        assert!(empty.credentials().is_none());

        let mut anon = spec();
        anon.access_key_id = None;
        anon.secret_access_key = None;
        assert!(anon.credentials().is_none());
    }

    #[test]
    fn the_helper_is_only_requested_by_its_own_argv() {
        // Guard the dispatch predicate: a stray match would make the agent exit
        // instead of booting the machine.
        assert_eq!(HELPER_ARG, "s3-mount");
    }

    #[test]
    fn optional_fields_may_be_omitted_entirely() {
        let json = r#"{"endpoint":"http://e","region":"r","bucket":"b","mountpoint":"/m"}"#;
        let s: MountSpec = serde_json::from_str(json).unwrap();
        assert_eq!(s.prefix, "");
        assert!(!s.read_only);
        assert!(s.credentials().is_none());
    }
}
