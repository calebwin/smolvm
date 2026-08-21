//! Remote volumes: mount object stores and network filesystems into a
//! machine with the same `-v SOURCE:GUEST[:ro]` flag as directory mounts.
//!
//! Two source forms are recognized; everything else stays a host directory
//! mount handled by `HostMount`:
//!
//! - `s3://bucket[/prefix]` — sugar for an S3 mount. Credentials come from
//!   the machine's `--env` (`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`,
//!   optional `AWS_ENDPOINT_URL` for R2/MinIO); without credentials the
//!   bucket is accessed anonymously, which covers public datasets.
//! - `:backend,opt=value,...:path` — a raw rclone connection string passed
//!   through verbatim, so every rclone-supported filesystem (sftp, gcs,
//!   azure, webdav, http, b2, ...) works without smolvm knowing about it.
//!
//! The mounts run inside the machine's workload container, whose command is
//! wrapped with the mount script on every start: the guest kernel ships
//! FUSE, machine containers have the capabilities to `mknod /dev/fuse`, and
//! `exec`/`shell` sessions join the workload container's namespaces — so the
//! workload container is the one place a mount is visible everywhere and
//! lives exactly as long as the workload. The image must provide `rclone`
//! and `fusermount3` (installable via `--init` on first boot, which runs
//! before the workload launches).

use serde::{Deserialize, Serialize};

/// One remote volume attached to a machine, stored on its record verbatim so
/// `status` can echo what the user wrote and the mount command can evolve
/// without migrating persisted state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteVolume {
    /// The user-supplied source (`s3://bucket/prefix` or `:backend,...:path`).
    pub source: String,
    /// Absolute guest mount point.
    pub target: String,
    /// Mount read-only.
    pub read_only: bool,
}

/// Split raw `-v` specs into host-directory specs (returned untouched for
/// `HostMount::parse`, preserving its validation and error messages) and
/// parsed remote volumes.
pub fn split_specs(specs: &[String]) -> crate::Result<(Vec<String>, Vec<RemoteVolume>)> {
    let mut host = Vec::new();
    let mut remote = Vec::new();
    for spec in specs {
        match parse_spec(spec)? {
            Some(volume) => remote.push(volume),
            None => host.push(spec.clone()),
        }
    }
    // Remote targets must not collide with each other; the caller checks
    // them against host mount targets once those are parsed.
    let mut seen = std::collections::HashSet::new();
    for volume in &remote {
        if !seen.insert(&volume.target) {
            return Err(crate::Error::config(
                "parse remote volume",
                format!(
                    "duplicate mount target: {} is specified more than once",
                    volume.target
                ),
            ));
        }
    }
    Ok((host, remote))
}

/// Parse one `-v` spec; `Ok(None)` means "not a remote source" and the spec
/// belongs to the host-directory path. Mirrors `HostMount`'s right-anchored
/// parse so sources may contain colons (rclone connection strings do).
fn parse_spec(spec: &str) -> crate::Result<Option<RemoteVolume>> {
    let (rest, read_only) = match spec.rsplit_once(':') {
        Some((head, "ro")) => (head, true),
        Some((head, "rw")) => (head, false),
        _ => (spec, false),
    };
    let Some((source, target)) = rest.rsplit_once(':') else {
        return Ok(None);
    };
    let is_remote = source.starts_with("s3://") || source.starts_with(':');
    if !is_remote {
        return Ok(None);
    }
    if !target.starts_with('/') {
        return Err(crate::Error::config(
            "parse remote volume",
            format!("remote volume guest path must be absolute: '{spec}'"),
        ));
    }
    if target.contains(' ') {
        return Err(crate::Error::config(
            "parse remote volume",
            format!("remote volume guest path must not contain spaces: '{spec}'"),
        ));
    }
    // Both values end up inside a single-quoted shell word.
    if source.contains('\'') || target.contains('\'') {
        return Err(crate::Error::config(
            "parse remote volume",
            format!("remote volume spec must not contain single quotes: '{spec}'"),
        ));
    }
    if let Some(bucket) = source.strip_prefix("s3://") {
        if bucket.trim_matches('/').is_empty() {
            return Err(crate::Error::config(
                "parse remote volume",
                format!("s3 volume needs a bucket name: '{spec}'"),
            ));
        }
    } else if !raw_remote_has_path_colon(source) {
        // An rclone connection string is `:backend,opts:path` — the path colon
        // is part of the remote. With an empty remote path the user must write
        // it explicitly, which makes a double colon before the guest path:
        // `:http,url="https://host"::/mnt/data`.
        return Err(crate::Error::config(
            "parse remote volume",
            format!(
                "rclone remote is missing its ':path' part in '{spec}' \
                 (an empty remote path is written '::' before the guest path, \
                 e.g. ':http,url=\"https://host\"::/mnt/data')"
            ),
        ));
    }
    Ok(Some(RemoteVolume {
        source: source.to_string(),
        target: target.to_string(),
        read_only,
    }))
}

/// Whether a structured mount source (the API's `MountSpec.source`) denotes a
/// remote volume rather than a host directory.
pub fn is_remote_source(source: &str) -> bool {
    source.starts_with("s3://") || source.starts_with(':')
}

/// Build a remote volume from already-split parts (the API's structured mount
/// spec), reusing the colon-spec parser so both entry points validate
/// identically.
pub fn from_parts(source: &str, target: &str, read_only: bool) -> crate::Result<RemoteVolume> {
    // The parser is right-anchored, so a colon in the target would mis-split
    // the reassembled spec; targets never legitimately contain one.
    if target.contains(':') {
        return Err(crate::Error::config(
            "parse remote volume",
            format!("remote volume guest path must not contain ':': '{target}'"),
        ));
    }
    let spec = format!("{source}:{target}{}", if read_only { ":ro" } else { "" });
    parse_spec(&spec)?.ok_or_else(|| {
        crate::Error::config(
            "parse remote volume",
            format!("not a remote volume source: '{source}'"),
        )
    })
}

/// Whether a raw rclone connection string still has its remote-terminating
/// `:path` colon. Colons inside double-quoted option values (URLs, mostly)
/// don't count — rclone's own parser treats those as part of the value.
fn raw_remote_has_path_colon(source: &str) -> bool {
    let mut in_quotes = false;
    for ch in source.chars().skip(1) {
        match ch {
            '"' => in_quotes = !in_quotes,
            ':' if !in_quotes => return true,
            _ => {}
        }
    }
    false
}

impl RemoteVolume {
    /// The rclone remote for this volume. Raw connection strings pass
    /// through; `s3://` sugar expands using the machine env: credentials in
    /// the env switch rclone to env-based auth (the mount command runs with
    /// the machine env applied), and `AWS_ENDPOINT_URL` points the S3 API at
    /// R2/MinIO/other S3-compatible stores.
    fn rclone_remote(&self, env: &[(String, String)]) -> crate::Result<String> {
        let Some(bucket) = self.source.strip_prefix("s3://") else {
            return Ok(self.source.clone());
        };
        let has = |key: &str| env.iter().any(|(k, _)| k == key);
        let mut opts = String::from(":s3,provider=AWS");
        if has("AWS_ACCESS_KEY_ID") {
            opts.push_str(",env_auth=true");
        }
        if let Some((_, url)) = env.iter().find(|(k, _)| k == "AWS_ENDPOINT_URL") {
            if url.contains('"') || url.contains('\'') {
                return Err(crate::Error::config(
                    "remote volume",
                    "AWS_ENDPOINT_URL must not contain quotes",
                ));
            }
            opts.push_str(&format!(",endpoint=\"{url}\""));
            // Streaming single-part uploads to a plain-http endpoint fail in
            // rclone's aws-sdk-v2 backend (a signed payload needs a seekable
            // body; https avoids it via UNSIGNED-PAYLOAD). Local MinIO-style
            // endpoints are exactly the http case, so force multipart there.
            if url.starts_with("http://") {
                opts.push_str(",upload_cutoff=0");
            }
        }
        Ok(format!("{opts}:{}", bucket.trim_matches('/')))
    }

    /// The shell command that mounts this volume, run through the `--init`
    /// exec machinery on every machine start.
    pub fn mount_command(&self, env: &[(String, String)]) -> crate::Result<String> {
        let remote = self.rclone_remote(env)?;
        let mode = if self.read_only {
            " --read-only"
        } else {
            " --vfs-cache-mode writes"
        };
        Ok(format!(
            "command -v rclone >/dev/null 2>&1 && command -v fusermount3 >/dev/null 2>&1 || \
             {{ echo \"remote volume {target} needs rclone and fuse3 in the image \
             (alpine: apk add rclone fuse3 | debian/ubuntu: apt-get install -y rclone fuse3)\" >&2; exit 1; }}; \
             [ -e /dev/fuse ] || mknod /dev/fuse c 10 229; \
             mkdir -p '{target}' && rclone mount '{remote}' '{target}' --daemon{mode}",
            target = self.target,
        ))
    }
}

/// Mount commands for every remote volume on a record, in declaration order.
pub fn mount_commands(
    volumes: &[RemoteVolume],
    env: &[(String, String)],
) -> crate::Result<Vec<String>> {
    volumes.iter().map(|v| v.mount_command(env)).collect()
}

/// A synchronous pre-launch check that the image can mount remote volumes at
/// all. The mount itself runs inside the detached workload container, whose
/// failures are not surfaced to the caller — so the most common failure
/// (image without the tools) is caught here first, where it can fail the
/// start with an actionable message.
pub fn preflight_command(volumes: &[RemoteVolume]) -> Option<String> {
    if volumes.is_empty() {
        return None;
    }
    Some(
        "command -v rclone >/dev/null 2>&1 && command -v fusermount3 >/dev/null 2>&1 || \
         { echo \"remote volumes need rclone and fuse3 in the image \
         (alpine: apk add rclone fuse3 | debian/ubuntu: apt-get install -y rclone fuse3; \
         or install them with --init, which runs before the mounts)\" >&2; exit 1; }"
            .to_string(),
    )
}

/// A post-launch check that every remote volume actually mounted. Joins the
/// workload container (like any exec) and polls /proc/mounts: if the mount
/// script failed, the workload container is dead and the fresh exec container
/// sees no mounts either way — so this fails the start instead of leaving a
/// silently broken machine.
pub fn verify_command(volumes: &[RemoteVolume]) -> Option<String> {
    if volumes.is_empty() {
        return None;
    }
    let checks: Vec<String> = volumes
        .iter()
        .map(|v| {
            format!(
                "awk -v m='{}' '$2==m' /proc/mounts | grep -q rclone",
                v.target
            )
        })
        .collect();
    let all = checks.join(" && ");
    Some(format!(
        "t=0; while [ $t -lt 30 ]; do if {all}; then exit 0; fi; t=$((t+1)); sleep 0.5; done; \
         echo \"remote volume(s) failed to mount — check the machine's agent-console.log for the rclone error\" >&2; exit 1"
    ))
}

/// Wrap a machine's workload command so its container mounts the remote
/// volumes before the workload runs.
///
/// A FUSE mount lives in the mount namespace of the container that created
/// it, and `exec`/`shell` sessions join the workload container's namespaces —
/// so the workload container is the one place a mount is visible everywhere
/// and lives exactly as long as the machine's workload. A machine with no
/// workload of its own gets a minimal holder (`sleep infinity`) so a live
/// container exists for exec sessions to join. Machines whose image CMD exits
/// immediately (an interactive shell, say) should be created with a
/// long-lived command such as `-- sleep infinity`.
/// Build the remote-volume mount script — the `&&`-joined mount commands — that
/// the agent runs inside the workload container, ahead of the image-resolved
/// workload command. The agent (not the host) does the wrapping, so a service
/// image's own ENTRYPOINT still runs rather than being replaced by a mount stub,
/// and an image with no command falls back to a keep-alive holder there.
pub fn mount_script(volumes: &[RemoteVolume], env: &[(String, String)]) -> crate::Result<String> {
    Ok(mount_commands(volumes, env)?.join(" && "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(specs: &[&str]) -> (Vec<String>, Vec<RemoteVolume>) {
        let specs: Vec<String> = specs.iter().map(|s| s.to_string()).collect();
        split_specs(&specs).unwrap()
    }

    #[test]
    fn host_specs_pass_through_untouched() {
        // Plain dirs, relative dirs, and Windows drive-letter paths all stay
        // on the HostMount path exactly as written.
        let (host, remote) = split(&["/data:/data:ro", "C:\\data:/data", "./x:/y"]);
        assert_eq!(host.len(), 3);
        assert!(remote.is_empty());
    }

    #[test]
    fn s3_sugar_parses_with_mode() {
        let (host, remote) = split(&["s3://my-bucket/prefix:/mnt/data:ro"]);
        assert!(host.is_empty());
        assert_eq!(
            remote,
            vec![RemoteVolume {
                source: "s3://my-bucket/prefix".into(),
                target: "/mnt/data".into(),
                read_only: true,
            }]
        );
    }

    #[test]
    fn raw_rclone_strings_keep_their_internal_colons() {
        // The connection string itself contains ':' and ','; the right-anchored
        // parse peels the guest path off the end.
        let (_, remote) = split(&[":sftp,host=example.com,user=me:/srv/files:/mnt/sftp"]);
        assert_eq!(
            remote[0].source,
            ":sftp,host=example.com,user=me:/srv/files"
        );
        assert_eq!(remote[0].target, "/mnt/sftp");
        assert!(!remote[0].read_only);
    }

    #[test]
    fn s3_translation_is_anonymous_without_credentials() {
        let v = &split(&["s3://bucket/p:/d"]).1[0];
        assert_eq!(v.rclone_remote(&[]).unwrap(), ":s3,provider=AWS:bucket/p");
    }

    #[test]
    fn s3_translation_uses_env_auth_and_endpoint_when_present() {
        let v = &split(&["s3://bucket:/d"]).1[0];
        let env = vec![
            ("AWS_ACCESS_KEY_ID".to_string(), "k".to_string()),
            (
                "AWS_ENDPOINT_URL".to_string(),
                "https://acc.r2.cloudflarestorage.com".to_string(),
            ),
        ];
        assert_eq!(
            v.rclone_remote(&env).unwrap(),
            ":s3,provider=AWS,env_auth=true,endpoint=\"https://acc.r2.cloudflarestorage.com\":bucket"
        );
        // http endpoints additionally force multipart uploads — streaming
        // single-part PUTs to plain http fail in rclone's aws-sdk-v2 backend.
        let http_env = vec![(
            "AWS_ENDPOINT_URL".to_string(),
            "http://100.96.0.1:9000".to_string(),
        )];
        assert_eq!(
            v.rclone_remote(&http_env).unwrap(),
            ":s3,provider=AWS,endpoint=\"http://100.96.0.1:9000\",upload_cutoff=0:bucket"
        );
    }

    #[test]
    fn rejects_relative_guest_path_and_quotes_and_empty_bucket() {
        assert!(split_specs(&["s3://b:data".to_string()]).is_err());
        assert!(split_specs(&["s3://b:/it's:ro".to_string()]).is_err());
        assert!(split_specs(&["s3://:/d".to_string()]).is_err());
        assert!(split_specs(&["s3://b:/d".to_string(), "s3://c:/d".to_string()]).is_err());
    }

    #[test]
    fn raw_remote_must_keep_its_path_colon() {
        // Missing ':path' (the trailing colon was eaten as the guest separator)
        // is rejected with the '::' hint...
        let err = split_specs(&[":http,url=\"https://host\":/mnt/d".to_string()]).unwrap_err();
        assert!(err.to_string().contains("::"));
        // ...and the double-colon empty-path form parses with the colon intact.
        let (_, remote) = split(&[":http,url=\"https://host\"::/mnt/d:ro"]);
        assert_eq!(remote[0].source, ":http,url=\"https://host\":");
        assert_eq!(remote[0].target, "/mnt/d");
    }

    #[test]
    fn mount_script_is_the_joined_mount_commands_not_a_wrapper() {
        // The host builds only the mount SCRIPT; the agent wraps it ahead of the
        // image-resolved command (so a service image's entrypoint is preserved).
        let vols = split(&["s3://bucket:/mnt/d:ro"]).1;
        let script = mount_script(&vols, &[]).unwrap();
        assert!(
            script.contains("rclone"),
            "script mounts via rclone: {script}"
        );
        assert!(
            script.contains("/mnt/d"),
            "script targets the guest path: {script}"
        );
        assert!(
            !script.contains("exec \"$@\"") && !script.contains("sleep infinity"),
            "wrapping is the agent's job now, not the host's: {script}"
        );
    }

    #[test]
    fn from_parts_matches_colon_parser() {
        let structured = from_parts("s3://bucket/pfx", "/mnt/d", true).unwrap();
        let parsed = &split(&["s3://bucket/pfx:/mnt/d:ro"]).1[0];
        assert_eq!(&structured, parsed);
        // Same validation as the colon parser: relative target, missing
        // rclone path colon, empty bucket.
        assert!(from_parts("s3://b", "relative", false).is_err());
        assert!(from_parts(":http,url=\"https://h\"", "/mnt/x", false).is_err());
        assert!(from_parts("s3://", "/mnt/x", false).is_err());
        // Colon in a structured target would mis-split the reassembled spec.
        assert!(from_parts("s3://b", "/mnt/a:ro", false).is_err());
        assert!(!is_remote_source("/host/dir"));
        assert!(is_remote_source("s3://b"));
        assert!(is_remote_source(":http,url=\"https://h\":"));
    }

    #[test]
    fn mount_command_quotes_and_picks_mode() {
        let v = &split(&["s3://bucket:/mnt/d:ro"]).1[0];
        let cmd = v.mount_command(&[]).unwrap();
        assert!(
            cmd.contains("rclone mount ':s3,provider=AWS:bucket' '/mnt/d' --daemon --read-only")
        );
        assert!(cmd.contains("mknod /dev/fuse"));
        let rw = &split(&["s3://bucket:/mnt/d"]).1[0];
        assert!(rw
            .mount_command(&[])
            .unwrap()
            .contains("--daemon --vfs-cache-mode writes"));
    }
}
