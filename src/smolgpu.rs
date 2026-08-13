//! Opt-in lifecycle client for GPU-resident CPU-style process populations.
//!
//! This module talks to the separately distributed `smolgpu-host` executable.
//! It does not change the libkrun machine backend, CUDA API remoting, or the
//! meaning of a SmolVM root filesystem. A SmolGPU workload is an explicit packed
//! RV64 process artifact with a bounded virtual Linux ABI.

use crate::{Error, Result};
use std::ffi::CString;
use std::io::{self, Read};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const READY_MAGIC: &[u8; 4] = b"SGHD";
const REQUEST_MAGIC: &[u8; 4] = b"SGHQ";
const RESPONSE_MAGIC: &[u8; 4] = b"SGHS";
const PROTOCOL_VERSION: u32 = 1;
const RESPONSE_HEADER_BYTES: usize = 72;
const MAX_REQUEST_BYTES: usize = 4096;
const FLAG_GROUP_SIMILAR: u32 = 1;

/// Configuration for one resident SmolGPU process population.
#[derive(Clone, Debug)]
pub struct SmolGpuPoolConfig {
    /// Path to the separately installed `smolgpu-host` executable.
    pub host_binary: PathBuf,
    /// Path to the CUDA SmolGPU runtime executable.
    pub runtime_binary: PathBuf,
    /// Packed `.sgpu` RV64 process artifact.
    pub workload: PathBuf,
    /// Number of simultaneously resident logical processes.
    pub contexts: u32,
    /// Preserve process state and identity between admissions.
    pub persistent: bool,
    /// Readiness and per-admission deadline.
    pub timeout: Duration,
    /// Optional target-specific linked JIT cache directory.
    pub jit_cache_dir: Option<PathBuf>,
    /// Maximum guest instructions allowed per admission.
    pub max_instructions: u64,
}

impl SmolGpuPoolConfig {
    /// Create an explicit process-pool configuration.
    pub fn new(
        host_binary: impl Into<PathBuf>,
        runtime_binary: impl Into<PathBuf>,
        workload: impl Into<PathBuf>,
        contexts: u32,
    ) -> Self {
        Self {
            host_binary: host_binary.into(),
            runtime_binary: runtime_binary.into(),
            workload: workload.into(),
            contexts,
            persistent: true,
            timeout: Duration::from_secs(300),
            jit_cache_dir: None,
            max_instructions: 10_000_000,
        }
    }

    /// Select persistent workers or COW-reforked workers for every admission.
    pub fn persistent(mut self, value: bool) -> Self {
        self.persistent = value;
        self
    }

    /// Set the readiness and per-admission deadline.
    pub fn timeout(mut self, value: Duration) -> Self {
        self.timeout = value;
        self
    }

    /// Select a target-specific linked JIT cache directory.
    pub fn jit_cache_dir(mut self, value: impl Into<PathBuf>) -> Self {
        self.jit_cache_dir = Some(value.into());
        self
    }

    /// Set the maximum guest instruction count per admission.
    pub fn max_instructions(mut self, value: u64) -> Self {
        self.max_instructions = value;
        self
    }

    fn shared_bytes(&self) -> io::Result<usize> {
        if self.contexts == 0 {
            return Err(invalid_input("SmolGPU context count must be positive"));
        }
        usize::try_from(self.contexts)
            .ok()
            .and_then(|contexts| contexts.checked_mul(MAX_REQUEST_BYTES + 4))
            .and_then(|bytes| bytes.checked_add(RESPONSE_HEADER_BYTES))
            .ok_or_else(|| invalid_input("SmolGPU host shared-memory size overflow"))
    }
}

/// Timings and COW-density telemetry for one completed admission.
#[derive(Clone, Debug)]
pub struct SmolGpuMetrics {
    /// Number of logical process contexts in the population.
    pub contexts: u32,
    /// GPU execution interval reported by the runtime.
    pub gpu_elapsed: Duration,
    /// Inner broker wall interval, including GPU transport and output gathering.
    pub broker_wall: Duration,
    /// Snapshot reset or persistent-resume preparation interval.
    pub preparation: Duration,
    /// Complete SmolVM adapter interval from frame construction through response validation.
    pub adapter_wall: Duration,
    /// Physical active bytes divided by resident context count.
    pub active_bytes_per_context: u64,
    /// Total guest instructions retired by the admission.
    pub retired_instructions: u64,
    /// COW pages allocated by the population.
    pub allocated_pages: u32,
    /// COW page capacity provisioned for the population.
    pub pool_capacity_pages: u32,
}

impl SmolGpuMetrics {
    /// GPU-interval aggregate tasks per second.
    pub fn gpu_tasks_per_second(&self) -> f64 {
        f64::from(self.contexts) / self.gpu_elapsed.as_secs_f64()
    }

    /// Complete adapter aggregate tasks per second.
    pub fn adapter_tasks_per_second(&self) -> f64 {
        f64::from(self.contexts) / self.adapter_wall.as_secs_f64()
    }

    /// Dirty COW pages per resident logical process.
    pub fn dirty_pages_per_context(&self) -> f64 {
        f64::from(self.allocated_pages) / f64::from(self.contexts)
    }
}

/// Borrowed outputs and telemetry for one completed admission.
///
/// The borrow prevents another admission from overwriting the shared response
/// until the caller has finished consuming it.
pub struct SmolGpuBatch<'a> {
    /// Independently length-delimited outputs in logical context order.
    pub outputs: SmolGpuOutputs<'a>,
    /// Execution and density telemetry.
    pub metrics: SmolGpuMetrics,
}

/// Independently length-delimited output records borrowed from shared memory.
pub struct SmolGpuOutputs<'a> {
    lengths: &'a [u8],
    payload: &'a [u8],
    contexts: usize,
}

impl<'a> SmolGpuOutputs<'a> {
    /// Number of logical process outputs.
    pub fn len(&self) -> usize {
        self.contexts
    }

    /// Whether the population returned no logical process outputs.
    pub fn is_empty(&self) -> bool {
        self.contexts == 0
    }

    /// Iterate through outputs in stable logical context order.
    pub fn iter(&self) -> SmolGpuOutputIter<'a> {
        SmolGpuOutputIter {
            lengths: self.lengths,
            payload: self.payload,
            remaining: self.contexts,
        }
    }

    /// Materialize an owned output vector for convenience-oriented callers.
    pub fn to_owned(&self) -> Vec<Vec<u8>> {
        self.iter().map(<[u8]>::to_vec).collect()
    }
}

/// Iterator over shared-memory SmolGPU outputs.
pub struct SmolGpuOutputIter<'a> {
    lengths: &'a [u8],
    payload: &'a [u8],
    remaining: usize,
}

impl<'a> Iterator for SmolGpuOutputIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let length = u32::from_le_bytes(self.lengths[..4].try_into().unwrap()) as usize;
        let (output, rest) = self.payload.split_at(length);
        self.lengths = &self.lengths[4..];
        self.payload = rest;
        self.remaining -= 1;
        Some(output)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for SmolGpuOutputIter<'_> {}

/// A running, fixed-capacity SmolGPU process population.
pub struct SmolGpuPool {
    config: SmolGpuPoolConfig,
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: ChildStdout,
    stderr_reader: Option<JoinHandle<Vec<u8>>>,
    shared: SharedMemory,
    max_request: u32,
    poisoned: bool,
}

struct ResponseMetadata {
    response_bytes: usize,
    payload_offset: usize,
    contexts: usize,
    metrics: SmolGpuMetrics,
}

impl SmolGpuPool {
    /// Prepare the JIT artifact, start the host process, and wait for readiness.
    pub fn start(config: SmolGpuPoolConfig) -> Result<Self> {
        Self::start_io(config)
            .map_err(|error| Error::agent("start SmolGPU pool", error.to_string()))
    }

    fn start_io(config: SmolGpuPoolConfig) -> io::Result<Self> {
        let shared = SharedMemory::new(config.shared_bytes()?)?;
        let shared_fd = shared.fd.as_raw_fd();
        let mut command = Command::new(&config.host_binary);
        command
            .arg("--runtime")
            .arg(&config.runtime_binary)
            .arg("--workload")
            .arg(&config.workload)
            .arg("--contexts")
            .arg(config.contexts.to_string())
            .arg(if config.persistent {
                "--persistent"
            } else {
                "--refork"
            })
            .arg("--timeout-seconds")
            .arg(config.timeout.as_secs().max(1).to_string())
            .arg("--max-instructions")
            .arg(config.max_instructions.to_string())
            .arg("--shm-fd")
            .arg(shared_fd.to_string())
            .arg("--shm-bytes")
            .arg(shared.len.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(cache) = &config.jit_cache_dir {
            command.arg("--jit-cache-dir").arg(cache);
        }
        // Keep the parent's descriptor close-on-exec and clear it only in the
        // post-fork child that immediately execs `smolgpu-host`.
        unsafe {
            command.pre_exec(move || {
                if libc::fcntl(shared_fd, libc::F_SETFD, 0) < 0 {
                    Err(io::Error::last_os_error())
                } else {
                    Ok(())
                }
            });
        }
        let mut child = command.spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| broken_pipe("missing SmolGPU host stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| broken_pipe("missing SmolGPU host stdout"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| broken_pipe("missing SmolGPU host stderr"))?;
        let stderr_reader = thread::spawn(move || {
            let mut bytes = Vec::new();
            let mut stream = stderr;
            let _ = stream.read_to_end(&mut bytes);
            bytes
        });
        let mut pool = Self {
            config,
            child,
            stdin: Some(stdin),
            stdout,
            stderr_reader: Some(stderr_reader),
            shared,
            max_request: 0,
            poisoned: false,
        };
        if let Err(error) = pool.read_readiness() {
            let stderr = pool.shutdown_and_capture();
            let detail = String::from_utf8_lossy(&stderr).trim().to_owned();
            return Err(if detail.is_empty() {
                error
            } else {
                io::Error::new(error.kind(), format!("{error}: {detail}"))
            });
        }
        Ok(pool)
    }

    /// Number of logical processes in this fixed-capacity population.
    pub fn contexts(&self) -> u32 {
        self.config.contexts
    }

    /// Maximum request bytes accepted for one logical process.
    pub fn max_request_bytes(&self) -> u32 {
        self.max_request
    }

    /// Whether the host process remains safe for another admission.
    pub fn is_healthy(&self) -> bool {
        !self.poisoned
    }

    /// Execute one request for every resident logical process.
    ///
    /// `group_similar` may reorder execution by request size, but outputs and
    /// mutable process state remain attached to their original context index.
    pub fn execute_batch<'a>(
        &'a mut self,
        requests: &[Vec<u8>],
        group_similar: bool,
    ) -> Result<SmolGpuBatch<'a>> {
        self.execute_batch_io(requests, group_similar)
            .map_err(|error| Error::agent("execute SmolGPU batch", error.to_string()))
    }

    fn execute_batch_io<'a>(
        &'a mut self,
        requests: &[Vec<u8>],
        group_similar: bool,
    ) -> io::Result<SmolGpuBatch<'a>> {
        if self.poisoned {
            return Err(broken_pipe(
                "SmolGPU pool cannot be reused after a host or protocol failure",
            ));
        }
        if requests.len() != self.config.contexts as usize {
            return Err(invalid_input(format!(
                "batch has {} requests for {} resident contexts",
                requests.len(),
                self.config.contexts
            )));
        }
        let mut payload_bytes = 0usize;
        for request in requests {
            if request.len() > self.max_request as usize {
                return Err(invalid_input(format!(
                    "request exceeds the {}-byte SmolGPU host limit",
                    self.max_request
                )));
            }
            payload_bytes = payload_bytes
                .checked_add(request.len())
                .ok_or_else(|| invalid_input("SmolGPU request payload overflow"))?;
        }
        let adapter_start = Instant::now();
        let mut cursor = 0usize;
        self.shared.write(&mut cursor, REQUEST_MAGIC)?;
        self.shared.write_u32(&mut cursor, PROTOCOL_VERSION)?;
        self.shared.write_u32(
            &mut cursor,
            if group_similar { FLAG_GROUP_SIMILAR } else { 0 },
        )?;
        self.shared.write_u32(&mut cursor, self.config.contexts)?;
        self.shared.write_u64(
            &mut cursor,
            u64::try_from(payload_bytes)
                .map_err(|_| invalid_input("SmolGPU payload does not fit protocol"))?,
        )?;
        for request in requests {
            self.shared.write_u32(
                &mut cursor,
                u32::try_from(request.len())
                    .map_err(|_| invalid_input("SmolGPU request is too large"))?,
            )?;
        }
        for request in requests {
            self.shared.write(&mut cursor, request)?;
        }

        let metadata = match self.finish_admission(cursor, adapter_start) {
            Ok(metadata) => metadata,
            Err(error) => {
                self.poisoned = true;
                let _ = self.stdin.take();
                let _ = self.child.kill();
                let _ = self.child.wait();
                self.join_stderr();
                return Err(error);
            }
        };
        let frame = &self.shared.as_slice()[..metadata.response_bytes];
        let lengths = &frame[RESPONSE_HEADER_BYTES..metadata.payload_offset];
        Ok(SmolGpuBatch {
            outputs: SmolGpuOutputs {
                lengths,
                payload: &frame[metadata.payload_offset..],
                contexts: metadata.contexts,
            },
            metrics: metadata.metrics,
        })
    }

    fn finish_admission(
        &mut self,
        request_bytes: usize,
        adapter_start: Instant,
    ) -> io::Result<ResponseMetadata> {
        let stdin = self
            .stdin
            .as_ref()
            .ok_or_else(|| broken_pipe("SmolGPU host is shut down"))?;
        let deadline = Instant::now() + self.config.timeout;
        write_all_deadline(
            stdin.as_raw_fd(),
            &(request_bytes as u64).to_le_bytes(),
            deadline,
        )?;
        let mut notification = [0u8; 8];
        read_exact_deadline(self.stdout.as_raw_fd(), &mut notification, deadline)?;
        let response_bytes = usize::try_from(u64::from_le_bytes(notification))
            .map_err(|_| invalid_data("SmolGPU host response size overflow"))?;
        if response_bytes < RESPONSE_HEADER_BYTES || response_bytes > self.shared.len {
            return Err(invalid_data("invalid SmolGPU host response size"));
        }
        let frame = &self.shared.as_slice()[..response_bytes];
        if &frame[..4] != RESPONSE_MAGIC || read_u32(frame, 4)? != PROTOCOL_VERSION {
            return Err(invalid_data("invalid SmolGPU host response header"));
        }
        let status = read_u32(frame, 8)?;
        let contexts = read_u32(frame, 12)?;
        let payload_bytes = usize::try_from(read_u64(frame, 16)?)
            .map_err(|_| invalid_data("SmolGPU output payload size overflow"))?;
        if status != 0 {
            if contexts != 0 || RESPONSE_HEADER_BYTES + payload_bytes != frame.len() {
                return Err(invalid_data("invalid SmolGPU host error response"));
            }
            return Err(io::Error::other(format!(
                "SmolGPU host rejected the admission: {}",
                String::from_utf8_lossy(&frame[RESPONSE_HEADER_BYTES..])
            )));
        }
        if contexts != self.config.contexts {
            return Err(invalid_data(
                "SmolGPU host returned the wrong context count",
            ));
        }
        let lengths_bytes = (contexts as usize)
            .checked_mul(4)
            .ok_or_else(|| invalid_data("SmolGPU output length table overflow"))?;
        let payload_offset = RESPONSE_HEADER_BYTES
            .checked_add(lengths_bytes)
            .ok_or_else(|| invalid_data("SmolGPU output frame overflow"))?;
        if payload_offset
            .checked_add(payload_bytes)
            .filter(|&expected| expected == frame.len())
            .is_none()
        {
            return Err(invalid_data("invalid SmolGPU output frame size"));
        }
        let lengths = &frame[RESPONSE_HEADER_BYTES..payload_offset];
        let mut described_payload = 0usize;
        for context in 0..contexts as usize {
            let length = read_u32(lengths, context * 4)? as usize;
            if length > self.max_request as usize {
                return Err(invalid_data(format!(
                    "SmolGPU context {context} returned an oversized output"
                )));
            }
            described_payload = described_payload
                .checked_add(length)
                .ok_or_else(|| invalid_data("SmolGPU output length overflow"))?;
        }
        if described_payload != payload_bytes {
            return Err(invalid_data("SmolGPU output lengths are inconsistent"));
        }
        let gpu_elapsed = Duration::from_nanos(read_u64(frame, 24)?);
        if gpu_elapsed.is_zero() {
            return Err(invalid_data("SmolGPU host reported zero execution time"));
        }
        Ok(ResponseMetadata {
            response_bytes,
            payload_offset,
            contexts: contexts as usize,
            metrics: SmolGpuMetrics {
                contexts,
                gpu_elapsed,
                broker_wall: Duration::from_nanos(read_u64(frame, 32)?),
                preparation: Duration::from_nanos(read_u64(frame, 40)?),
                adapter_wall: adapter_start.elapsed(),
                active_bytes_per_context: read_u64(frame, 48)?,
                retired_instructions: read_u64(frame, 56)?,
                allocated_pages: read_u32(frame, 64)?,
                pool_capacity_pages: read_u32(frame, 68)?,
            },
        })
    }

    /// Close admissions and reap the host process.
    pub fn shutdown(mut self) -> Result<()> {
        self.shutdown_io()
            .map_err(|error| Error::agent("stop SmolGPU pool", error.to_string()))
    }

    fn shutdown_io(&mut self) -> io::Result<()> {
        self.stdin.take();
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if self.child.try_wait()?.is_some() {
                self.join_stderr();
                return Ok(());
            }
            if Instant::now() >= deadline {
                self.child.kill()?;
                self.child.wait()?;
                self.join_stderr();
                return Ok(());
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn read_readiness(&mut self) -> io::Result<()> {
        let mut bytes = [0u8; 16];
        read_exact_deadline(
            self.stdout.as_raw_fd(),
            &mut bytes,
            Instant::now() + self.config.timeout,
        )?;
        if &bytes[..4] != READY_MAGIC
            || read_u32(&bytes, 4)? != PROTOCOL_VERSION
            || read_u32(&bytes, 8)? != self.config.contexts
        {
            return Err(invalid_data("invalid SmolGPU host readiness record"));
        }
        self.max_request = read_u32(&bytes, 12)?;
        if self.max_request == 0 || self.max_request as usize > MAX_REQUEST_BYTES {
            return Err(invalid_data("invalid SmolGPU host request limit"));
        }
        Ok(())
    }

    fn shutdown_and_capture(&mut self) -> Vec<u8> {
        self.stdin.take();
        let deadline = Instant::now() + Duration::from_secs(1);
        while self.child.try_wait().ok().flatten().is_none() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        self.stderr_reader
            .take()
            .and_then(|reader| reader.join().ok())
            .unwrap_or_default()
    }

    fn join_stderr(&mut self) {
        if let Some(reader) = self.stderr_reader.take() {
            let _ = reader.join();
        }
    }
}

impl Drop for SmolGpuPool {
    fn drop(&mut self) {
        let _ = self.shutdown_and_capture();
    }
}

struct SharedMemory {
    fd: OwnedFd,
    pointer: *mut u8,
    len: usize,
}

impl SharedMemory {
    fn new(len: usize) -> io::Result<Self> {
        let name = CString::new("smolvm-smolgpu").expect("static memfd name");
        let raw_fd = unsafe {
            libc::syscall(libc::SYS_memfd_create, name.as_ptr(), libc::MFD_CLOEXEC) as libc::c_int
        };
        if raw_fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
        if unsafe { libc::ftruncate(fd.as_raw_fd(), len as libc::off_t) } < 0 {
            return Err(io::Error::last_os_error());
        }
        let pointer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if pointer == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            fd,
            pointer: pointer.cast(),
            len,
        })
    }

    fn write(&mut self, cursor: &mut usize, bytes: &[u8]) -> io::Result<()> {
        let end = cursor
            .checked_add(bytes.len())
            .ok_or_else(|| invalid_input("SmolGPU shared request size overflow"))?;
        if end > self.len {
            return Err(invalid_input(
                "SmolGPU shared request exceeds mapping capacity",
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.pointer.add(*cursor), bytes.len());
        }
        *cursor = end;
        Ok(())
    }

    fn write_u32(&mut self, cursor: &mut usize, value: u32) -> io::Result<()> {
        self.write(cursor, &value.to_le_bytes())
    }

    fn write_u64(&mut self, cursor: &mut usize, value: u64) -> io::Result<()> {
        self.write(cursor, &value.to_le_bytes())
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.pointer, self.len) }
    }
}

impl Drop for SharedMemory {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.pointer.cast(), self.len);
        }
    }
}

// SAFETY: the mapping is only accessed through `&mut SmolGpuPool`; the stderr
// reader never receives it, so moving a pool between serialized owners is safe.
unsafe impl Send for SharedMemory {}

fn poll_fd(fd: RawFd, events: libc::c_short, deadline: Instant) -> io::Result<()> {
    loop {
        let now = Instant::now();
        if now >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "SmolGPU host timed out",
            ));
        }
        let remaining = deadline.duration_since(now);
        let timeout_ms = remaining
            .as_millis()
            .saturating_add(u128::from(
                !remaining.subsec_nanos().is_multiple_of(1_000_000),
            ))
            .clamp(1, i32::MAX as u128) as i32;
        let mut descriptor = libc::pollfd {
            fd,
            events,
            revents: 0,
        };
        let result = unsafe { libc::poll(&mut descriptor, 1, timeout_ms) };
        if result > 0 {
            if descriptor.revents & libc::POLLNVAL != 0 {
                return Err(broken_pipe("SmolGPU host descriptor is invalid"));
            }
            return Ok(());
        }
        if result == 0 {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "SmolGPU host timed out",
            ));
        }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

fn read_exact_deadline(fd: RawFd, output: &mut [u8], deadline: Instant) -> io::Result<()> {
    let mut offset = 0usize;
    while offset < output.len() {
        poll_fd(fd, libc::POLLIN, deadline)?;
        let bytes = unsafe {
            libc::read(
                fd,
                output[offset..].as_mut_ptr().cast(),
                output.len() - offset,
            )
        };
        if bytes > 0 {
            offset += bytes as usize;
        } else if bytes == 0 {
            return Err(broken_pipe("SmolGPU host exited unexpectedly"));
        } else {
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(error);
            }
        }
    }
    Ok(())
}

fn write_all_deadline(fd: RawFd, input: &[u8], deadline: Instant) -> io::Result<()> {
    let mut offset = 0usize;
    while offset < input.len() {
        poll_fd(fd, libc::POLLOUT, deadline)?;
        let bytes =
            unsafe { libc::write(fd, input[offset..].as_ptr().cast(), input.len() - offset) };
        if bytes > 0 {
            offset += bytes as usize;
        } else if bytes == 0 {
            return Err(broken_pipe("SmolGPU host accepted zero notification bytes"));
        } else {
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(error);
            }
        }
    }
    Ok(())
}

fn read_u32(bytes: &[u8], offset: usize) -> io::Result<u32> {
    bytes
        .get(offset..offset + 4)
        .and_then(|value| value.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| invalid_data("truncated SmolGPU host integer"))
}

fn read_u64(bytes: &[u8], offset: usize) -> io::Result<u64> {
    bytes
        .get(offset..offset + 8)
        .and_then(|value| value.try_into().ok())
        .map(u64::from_le_bytes)
        .ok_or_else(|| invalid_data("truncated SmolGPU host integer"))
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn broken_pipe(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::BrokenPipe, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_rejects_zero_contexts_and_checks_capacity() {
        let zero = SmolGpuPoolConfig::new("host", "runtime", "worker.sgpu", 0);
        assert!(zero.shared_bytes().is_err());
        let valid = SmolGpuPoolConfig::new("host", "runtime", "worker.sgpu", 100_000);
        assert_eq!(valid.shared_bytes().unwrap(), 410_000_072);
    }

    #[test]
    fn borrowed_outputs_iterate_without_materializing_per_context_state() {
        let lengths = [3u32.to_le_bytes(), 5u32.to_le_bytes()].concat();
        let outputs = SmolGpuOutputs {
            lengths: &lengths,
            payload: b"onethree",
            contexts: 2,
        };
        assert_eq!(outputs.len(), 2);
        assert!(!outputs.is_empty());
        assert_eq!(
            outputs.iter().collect::<Vec<_>>(),
            vec![b"one".as_slice(), b"three".as_slice()]
        );
        assert_eq!(outputs.to_owned(), vec![b"one".to_vec(), b"three".to_vec()]);
    }
}
