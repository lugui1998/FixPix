use std::sync::OnceLock;

static THREAD_COUNT: OnceLock<usize> = OnceLock::new();

pub fn default_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .max(1)
}

pub fn configure_global_pool(requested: Option<usize>) -> usize {
    let threads = requested.unwrap_or_else(default_thread_count).max(1);
    *THREAD_COUNT.get_or_init(|| {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("fixpix-{index}"))
            .build_global();
        threads
    })
}
