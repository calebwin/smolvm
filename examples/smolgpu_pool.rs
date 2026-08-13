#[cfg(target_os = "linux")]
mod linux {
    use smolvm::smolgpu::{SmolGpuPool, SmolGpuPoolConfig};
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::io;
    use std::path::PathBuf;

    const HISTORY_MULTIPLIER: u64 = 1_099_511_628_211;
    const SCORE_MULTIPLIER: u64 = 6_364_136_223_846_793_005;

    pub fn main() -> Result<(), Box<dyn Error>> {
        let mut arguments = env::args_os().skip(1);
        let host = PathBuf::from(arguments.next().ok_or_else(|| usage("host binary"))?);
        let runtime = PathBuf::from(arguments.next().ok_or_else(|| usage("GPU runtime"))?);
        let workload = PathBuf::from(arguments.next().ok_or_else(|| usage("workload"))?);
        let cache = PathBuf::from(arguments.next().ok_or_else(|| usage("JIT cache"))?);
        let contexts = arguments
            .next()
            .ok_or_else(|| usage("context count"))?
            .to_string_lossy()
            .parse::<u32>()
            .map_err(|_| usage("valid context count"))?;
        let request = fs::read(PathBuf::from(
            arguments.next().ok_or_else(|| usage("request file"))?,
        ))?;
        if arguments.next().is_some() {
            return Err(usage("exactly six arguments").into());
        }

        let config = SmolGpuPoolConfig::new(host, runtime, workload, contexts)
            .persistent(true)
            .jit_cache_dir(cache);
        let mut pool = SmolGpuPool::start(config)?;
        let mut histories = vec![0u64; contexts as usize];
        let broadcast = vec![request.clone(); contexts as usize];
        run_batch(&mut pool, &broadcast, false, &mut histories, 1)?;
        let variable = make_variable_requests(&request, contexts)?;
        run_batch(&mut pool, &variable, false, &mut histories, 2)?;
        run_batch(&mut pool, &variable, true, &mut histories, 3)?;
        pool.shutdown()?;
        Ok(())
    }

    fn run_batch(
        pool: &mut SmolGpuPool,
        requests: &[Vec<u8>],
        group_similar: bool,
        histories: &mut [u64],
        connection: u64,
    ) -> Result<(), Box<dyn Error>> {
        let batch = pool.execute_batch(requests, group_similar)?;
        for (context, (request, output)) in requests.iter().zip(batch.outputs.iter()).enumerate() {
            let value = parse_value(request)?;
            histories[context] = histories[context]
                .wrapping_mul(HISTORY_MULTIPLIER)
                .wrapping_add(value);
            let expected = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{{\"connection\":\"{connection:08x}\",\"history\":\"{:016x}\",\"score\":{}}}\n",
                histories[context],
                value.wrapping_mul(SCORE_MULTIPLIER),
            );
            if output != expected.as_bytes() {
                return Err(io::Error::other(format!(
                    "context {context} lost process identity or returned an incorrect response"
                ))
                .into());
            }
        }
        println!(
            "connection={} grouped={} contexts={} adapter_tasks_per_second={:.0} gpu_tasks_per_second={:.0} active_bytes_per_context={} dirty_pages_per_context={:.3}",
            connection,
            group_similar,
            batch.metrics.contexts,
            batch.metrics.adapter_tasks_per_second(),
            batch.metrics.gpu_tasks_per_second(),
            batch.metrics.active_bytes_per_context,
            batch.metrics.dirty_pages_per_context(),
        );
        Ok(())
    }

    fn make_variable_requests(base: &[u8], contexts: u32) -> io::Result<Vec<Vec<u8>>> {
        let marker = b"\"value\":42";
        let offset = base
            .windows(marker.len())
            .position(|window| window == marker)
            .ok_or_else(|| io::Error::other("request does not contain a two-digit value"))?
            + marker.len()
            - 2;
        Ok((0..contexts)
            .map(|context| {
                let mut request = base.to_vec();
                let value = 10 + context % 89;
                request[offset] = b'0' + (value / 10) as u8;
                request[offset + 1] = b'0' + (value % 10) as u8;
                request.extend(std::iter::repeat_n(b' ', (context % 3) as usize));
                request
            })
            .collect())
    }

    fn parse_value(request: &[u8]) -> io::Result<u64> {
        let key = b"\"value\":";
        let offset = request
            .windows(key.len())
            .position(|window| window == key)
            .ok_or_else(|| io::Error::other("request does not contain a value"))?
            + key.len();
        request[offset..]
            .iter()
            .take_while(|byte| byte.is_ascii_digit())
            .try_fold(0u64, |value, digit| {
                value
                    .checked_mul(10)
                    .and_then(|value| value.checked_add(u64::from(*digit - b'0')))
                    .ok_or_else(|| io::Error::other("request value overflowed"))
            })
    }

    fn usage(expected: &str) -> io::Error {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "expected {expected}: smolgpu_pool HOST_BINARY GPU_RUNTIME WORKLOAD JIT_CACHE CONTEXTS REQUEST"
            ),
        )
    }
}

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    linux::main()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("the SmolGPU pool adapter requires Linux");
    std::process::exit(1);
}
