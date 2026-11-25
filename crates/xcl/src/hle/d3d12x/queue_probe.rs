//! Bounded submission diagnostics, based on the earlier XCL queue timeline experiment.
use std::collections::{HashMap, VecDeque};
use std::io::{BufWriter, Write};
use std::sync::{Arc, LazyLock, Mutex, Once};
use std::time::Duration;
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::ID3D12Fence;

const EVENTS: usize = 16_384;
const LISTS: usize = 1024;
const FENCES: usize = 48;
const OPS: usize = 4;

#[derive(Clone, Copy)]
struct Event {
    seq: u64,
    us: u64,
    thread: u32,
    name: &'static str,
    queue: usize,
    object: usize,
    link: u64,
    value: u64,
    detail: u64,
}
#[derive(Clone, Copy)]
struct Op {
    name: &'static str,
    pipeline: usize,
    hashes: (u64, u64),
    size: [u32; 3],
}
struct List {
    generation: u64,
    count: u64,
    tail: [Option<Op>; OPS],
    last_mesh_indirect: Option<(u64, Op)>,
}
struct Fence {
    raw: Arc<ID3D12Fence>,
    queue: usize,
    completed: Option<u64>,
    removed: bool,
}
struct Trace {
    next: u64,
    events: VecDeque<Event>,
    lists: HashMap<usize, List>,
    fences: HashMap<usize, Fence>,
}
impl Trace {
    fn new() -> Self {
        Self {
            next: 0,
            events: VecDeque::with_capacity(EVENTS),
            lists: HashMap::new(),
            fences: HashMap::new(),
        }
    }
    fn push(
        &mut self,
        name: &'static str,
        queue: usize,
        object: usize,
        link: u64,
        value: u64,
        detail: u64,
    ) -> u64 {
        self.next += 1;
        if self.events.len() == EVENTS {
            self.events.pop_front();
        }
        self.events.push_back(Event {
            seq: self.next,
            us: (crate::log::uptime() * 1_000_000.0) as u64,
            thread: unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() },
            name,
            queue,
            object,
            link,
            value,
            detail,
        });
        self.next
    }
    fn reset(&mut self, list: usize) {
        if self.lists.contains_key(&list) || self.lists.len() < LISTS {
            let generation = self.push("list-reset", 0, list, 0, 0, 0);
            self.lists.insert(
                list,
                List {
                    generation,
                    count: 0,
                    tail: [None; OPS],
                    last_mesh_indirect: None,
                },
            );
        }
    }
}

static TRACE: LazyLock<Mutex<Trace>> = LazyLock::new(|| Mutex::new(Trace::new()));
pub(crate) fn enabled() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var_os("XCL_QUEUE_PROBE").is_some());
    *ON
}

pub(crate) fn event(
    name: &'static str,
    queue: usize,
    object: usize,
    link: u64,
    value: u64,
    detail: u64,
) -> u64 {
    if !enabled() {
        return 0;
    }
    TRACE
        .lock()
        .unwrap()
        .push(name, queue, object, link, value, detail)
}
pub(super) fn reset(list: usize) {
    if enabled() {
        TRACE.lock().unwrap().reset(list);
    }
}
pub(super) fn forget(list: usize) {
    if enabled() {
        TRACE.lock().unwrap().lists.remove(&list);
    }
}
pub(super) fn op(list: usize, name: &'static str, pipeline: usize, size: [u32; 3]) {
    if !enabled() {
        return;
    }
    let hashes = super::real::pso_shaders(pipeline).unwrap_or_default();
    let mut t = TRACE.lock().unwrap();
    if !t.lists.contains_key(&list) {
        t.reset(list);
    }
    if let Some(s) = t.lists.get_mut(&list) {
        let op = Op {
            name,
            pipeline,
            hashes,
            size,
        };
        s.tail[s.count as usize % OPS] = Some(op);
        if name == "mesh-indirect" {
            s.last_mesh_indirect = Some((s.count, op));
        }
        s.count += 1;
    }
}
pub(crate) fn submit(name: &'static str, queue: usize, lists: &[*mut std::ffi::c_void]) -> u64 {
    if !enabled() {
        return 0;
    }
    let mut t = TRACE.lock().unwrap();
    let id = t.push(name, queue, 0, 0, lists.len() as u64, 0);
    for &list in lists.iter().take(32) {
        let raw = list as usize;
        let data = t.lists.get(&raw).map(|s| (s.generation, s.count, s.tail, s.last_mesh_indirect));
        let (generation, count) = data.as_ref().map_or((0, 0), |(g, n, _, _)| (*g, *n));
        let entry = t.push("list", queue, raw, id, generation, count);
        if let Some((_, _, ops, last_mesh_indirect)) = data {
            let oldest = count.saturating_sub(OPS as u64);
            let older_mesh = last_mesh_indirect.filter(|(index, _)| *index < oldest).map(|(_, op)| op);
            let tail = (oldest..count).filter_map(|index| ops[index as usize % OPS]);
            for op in older_mesh.into_iter().chain(tail) {
                let [x, y, z] = op.size;
                let shader = t.push(op.name, queue, op.pipeline, entry, op.hashes.0, op.hashes.1);
                t.push(
                    "groups",
                    queue,
                    z as usize,
                    shader,
                    u64::from(x),
                    u64::from(y),
                );
            }
        }
    }
    id
}

/// Observe a signal already issued by XCL. This never enqueues additional GPU work.
pub(crate) fn signal(
    name: &'static str,
    queue: usize,
    fence: &ID3D12Fence,
    link: u64,
    value: u64,
    hr: i32,
) {
    if !enabled() {
        return;
    }
    let raw = fence.as_raw() as usize;
    let candidate = {
        let mut t = TRACE.lock().unwrap();
        t.push(name, queue, raw, link, value, u64::from(hr as u32));
        hr >= 0 && !t.fences.contains_key(&raw) && t.fences.len() < FENCES
    };
    if candidate {
        let fence = Arc::new(fence.clone());
        let mut t = TRACE.lock().unwrap();
        if t.fences.len() < FENCES {
            t.fences.entry(raw).or_insert_with(|| Fence {
                raw: fence.clone(),
                queue,
                completed: None,
                removed: false,
            });
        }
    }
}
fn update_completed(last: &mut Option<u64>, removed: &mut bool, value: u64) -> bool {
    if *removed {
        return false;
    }
    if value == u64::MAX {
        *removed = true;
    } else if last.is_none_or(|old| value > old) {
        *last = Some(value);
    } else {
        return false;
    }
    true
}

fn completed(fence: usize, value: u64) {
    if !enabled() {
        return;
    }
    let mut t = TRACE.lock().unwrap();
    if let Some(f) = t.fences.get_mut(&fence) {
        if !update_completed(&mut f.completed, &mut f.removed, value) {
            return;
        }
        let (queue, last, removed) = (f.queue, f.completed.unwrap_or(0), f.removed);
        t.push(
            if removed {
                "fence-removed"
            } else {
                "completed"
            },
            queue,
            fence,
            0,
            value,
            last,
        );
    }
}
pub(crate) fn start() {
    static STARTED: Once = Once::new();
    if !enabled() {
        return;
    }
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new()
            .name("xcl-queue-probe".into())
            .spawn(|| {
                loop {
                    std::thread::sleep(Duration::from_millis(25));
                    let mut watches = [const { None }; FENCES];
                    let used = {
                        let t = TRACE.lock().unwrap();
                        let mut used = 0;
                        for f in t.fences.values().filter(|f| !f.removed) {
                            watches[used] = Some(f.raw.clone());
                            used += 1;
                        }
                        used
                    };
                    for f in watches.into_iter().take(used).flatten() {
                        // Bypass the title hook and its sparse readback settlement.
                        let value = unsafe { super::real::fence_completed(&f) };
                        completed(f.as_raw() as usize, value);
                    }
                }
            });
    });
}

pub(crate) fn snapshot(filename: &str) {
    if !enabled() {
        return;
    }
    let (events, fences) = {
        let t = TRACE.lock().unwrap();
        let fences: Vec<_> = t
            .fences
            .iter()
            .map(|(&raw, f)| (raw, f.queue, f.completed, f.removed))
            .collect();
        (t.events.clone(), fences)
    };
    let result = (|| -> std::io::Result<()> {
        let path = crate::state::data_dir().join("logs").join(filename);
        let mut out = BufWriter::new(std::fs::File::create(path)?);
        writeln!(out, "# clock xcl.log uptime, microseconds")?;
        for (raw, queue, completed, removed) in fences {
            writeln!(
                out,
                "# fence {raw:#x} queue {queue:#x} last_completed {completed:?} removed {removed}"
            )?;
        }
        writeln!(
            out,
            "sequence\tmicroseconds\tthread\toperation\tqueue\tobject\tlink\tvalue\tdetail"
        )?;
        for e in events {
            writeln!(
                out,
                "{}\t{}\t{}\t{}\t{:#x}\t{:#x}\t{}\t{}\t{}",
                e.seq, e.us, e.thread, e.name, e.queue, e.object, e.link, e.value, e.detail
            )?;
        }
        out.flush()
    })();
    if let Err(e) = result {
        crate::xlog_warn!("D3D12 queue probe: failed to save {filename}: {e}");
    }
}

pub(super) fn device_removed() {
    static SAVED: Once = Once::new();
    if enabled() {
        SAVED.call_once(|| snapshot("queue-device-removed.tsv"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_removal_keeps_last_valid_completion() {
        let mut last = None;
        let mut removed = false;
        assert!(update_completed(&mut last, &mut removed, 19));
        assert!(!update_completed(&mut last, &mut removed, 18));
        assert!(update_completed(&mut last, &mut removed, u64::MAX));
        assert_eq!(last, Some(19));
        assert!(removed);
        assert!(!update_completed(&mut last, &mut removed, 20));
    }

    #[test]
    fn timeline_retains_the_latest_events() {
        let mut trace = Trace::new();
        for _ in 0..EVENTS + 2 {
            trace.push("test", 0, 0, 0, 0, 0);
        }
        assert_eq!(trace.events.len(), EVENTS);
        assert_eq!(trace.events.front().unwrap().seq, 3);
        assert_eq!(trace.events.back().unwrap().seq, EVENTS as u64 + 2);
    }
}
