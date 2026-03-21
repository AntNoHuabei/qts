use std::cell::{RefCell, RefMut};
use std::cmp::max;
use std::collections::BTreeMap;
use std::ffi::CStr;
use std::ptr::NonNull;
use std::rc::Rc;

use ggml::sys;

use crate::Qwen3TtsError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendKind {
    Cpu,
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    Metal,
    #[cfg(all(feature = "vulkan", not(target_vendor = "apple")))]
    Vulkan,
}

impl BackendKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            #[cfg(all(feature = "metal", target_vendor = "apple"))]
            Self::Metal => "Metal",
            #[cfg(all(feature = "vulkan", not(target_vendor = "apple")))]
            Self::Vulkan => "Vulkan",
        }
    }
}

#[derive(Clone)]
pub(crate) struct BackendSet(Rc<BackendSetInner>);

struct BackendSetInner {
    primary: OwnedBackend,
    cpu_fallback: Option<OwnedBackend>,
    primary_galloc: RefCell<OwnedGallocr>,
}

impl BackendSet {
    pub(crate) fn new() -> Result<Self, Qwen3TtsError> {
        unsafe {
            sys::ggml_backend_load_all();
            sys::ggml_cpu_init();
        }

        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        if let Some(backends) = Self::try_init_primary(BackendKind::Metal, b"Metal\0")? {
            return Ok(backends);
        }

        #[cfg(all(feature = "vulkan", not(target_vendor = "apple")))]
        if let Some(backends) = Self::try_init_primary(BackendKind::Vulkan, b"Vulkan\0")? {
            return Ok(backends);
        }

        let primary = OwnedBackend::cpu()?;
        if backend_debug_enabled() {
            eprintln!("[backend-debug] selected {}", BackendKind::Cpu.as_str());
        }
        Self::with_primary(primary, None)
    }

    #[cfg(any(
        all(feature = "metal", target_vendor = "apple"),
        all(feature = "vulkan", not(target_vendor = "apple"))
    ))]
    fn try_init_primary(kind: BackendKind, name: &[u8]) -> Result<Option<Self>, Qwen3TtsError> {
        if let Some(primary) = OwnedBackend::init_by_name(name)? {
            if backend_debug_enabled() {
                eprintln!("[backend-debug] selected {}", kind.as_str());
            }
            return Ok(Some(Self::with_primary(
                primary,
                Some(OwnedBackend::cpu()?),
            )?));
        }

        if backend_debug_enabled() {
            eprintln!(
                "[backend-debug] {} unavailable, falling back to CPU",
                kind.as_str()
            );
        }
        Ok(None)
    }

    fn with_primary(
        primary: OwnedBackend,
        cpu_fallback: Option<OwnedBackend>,
    ) -> Result<Self, Qwen3TtsError> {
        let primary_galloc = RefCell::new(OwnedGallocr::new(primary.as_ptr())?);
        Ok(Self(Rc::new(BackendSetInner {
            primary,
            cpu_fallback,
            primary_galloc,
        })))
    }

    pub(crate) fn primary_ptr(&self) -> sys::ggml_backend_t {
        self.0.primary.as_ptr()
    }

    pub(crate) fn configure_threads(&self, thread_count: usize) {
        self.0.primary.set_threads(thread_count);
        if let Some(cpu_fallback) = &self.0.cpu_fallback {
            cpu_fallback.set_threads(thread_count);
        }
    }

    fn primary_galloc(&self) -> RefMut<'_, OwnedGallocr> {
        self.0.primary_galloc.borrow_mut()
    }
}

struct OwnedBackend {
    raw: NonNull<sys::ggml_backend>,
    is_cpu: bool,
}

impl OwnedBackend {
    fn cpu() -> Result<Self, Qwen3TtsError> {
        let raw = unsafe { sys::ggml_backend_cpu_init() };
        let raw = NonNull::new(raw).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to initialize ggml CPU backend".into())
        })?;
        Ok(Self { raw, is_cpu: true })
    }

    #[cfg(any(
        all(feature = "metal", target_vendor = "apple"),
        all(feature = "vulkan", not(target_vendor = "apple"))
    ))]
    fn init_by_name(name: &[u8]) -> Result<Option<Self>, Qwen3TtsError> {
        let raw = unsafe { sys::ggml_backend_init_by_name(name.as_ptr().cast(), std::ptr::null()) };
        Ok(NonNull::new(raw).map(|raw| Self { raw, is_cpu: false }))
    }

    fn as_ptr(&self) -> sys::ggml_backend_t {
        self.raw.as_ptr()
    }

    fn set_threads(&self, thread_count: usize) {
        if self.is_cpu {
            unsafe {
                sys::ggml_backend_cpu_set_n_threads(
                    self.raw.as_ptr(),
                    normalize_threads(thread_count),
                );
            }
        }
    }
}

impl Drop for OwnedBackend {
    fn drop(&mut self) {
        unsafe {
            sys::ggml_backend_free(self.raw.as_ptr());
        }
    }
}

struct OwnedGallocr {
    raw: NonNull<sys::ggml_gallocr>,
}

impl OwnedGallocr {
    fn new(backend: sys::ggml_backend_t) -> Result<Self, Qwen3TtsError> {
        let raw =
            unsafe { sys::ggml_gallocr_new(sys::ggml_backend_get_default_buffer_type(backend)) };
        let raw = NonNull::new(raw).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to initialize ggml graph allocator".into())
        })?;
        Ok(Self { raw })
    }
}

impl Drop for OwnedGallocr {
    fn drop(&mut self) {
        unsafe {
            sys::ggml_gallocr_free(self.raw.as_ptr());
        }
    }
}

pub(crate) struct TensorUpload<'a> {
    pub(crate) tensor: *mut sys::ggml_tensor,
    pub(crate) bytes: &'a [u8],
}

pub(crate) struct TensorDownload<'a> {
    pub(crate) tensor: *mut sys::ggml_tensor,
    pub(crate) bytes: &'a mut [u8],
}

pub(crate) struct OwnedBuffer {
    raw: NonNull<sys::ggml_backend_buffer>,
}

impl OwnedBuffer {
    pub(crate) fn alloc(
        ctx: *mut sys::ggml_context,
        backend: sys::ggml_backend_t,
    ) -> Result<Self, Qwen3TtsError> {
        let raw = unsafe { sys::ggml_backend_alloc_ctx_tensors(ctx, backend) };
        let raw = NonNull::new(raw).ok_or_else(|| {
            Qwen3TtsError::InvalidInput("failed to allocate ggml backend tensor buffer".into())
        })?;
        Ok(Self { raw })
    }
}

impl Drop for OwnedBuffer {
    fn drop(&mut self) {
        unsafe {
            sys::ggml_backend_buffer_free(self.raw.as_ptr());
        }
    }
}

pub(crate) fn graph_metadata_mem_size(max_nodes: usize) -> usize {
    let tensor_overhead = unsafe { sys::ggml_tensor_overhead() };
    let graph_overhead = unsafe { sys::ggml_graph_overhead_custom(max_nodes, false) };
    max(
        1024 * 1024,
        graph_overhead + tensor_overhead * max_nodes * 16,
    )
}

pub(crate) fn slice_as_bytes<T>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), std::mem::size_of_val(slice)) }
}

pub(crate) fn slice_as_bytes_mut<T>(slice: &mut [T]) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(
            slice.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(slice),
        )
    }
}

pub(crate) fn execute_graph(
    backends: &BackendSet,
    graph: NonNull<sys::ggml_cgraph>,
    uploads: &[TensorUpload<'_>],
    downloads: &mut [TensorDownload<'_>],
    thread_count: usize,
    error_message: &str,
) -> Result<(), Qwen3TtsError> {
    maybe_log_backend_support(backends, graph, error_message);
    backends.configure_threads(thread_count);
    let galloc = backends.primary_galloc();
    let allocated = unsafe { sys::ggml_gallocr_alloc_graph(galloc.raw.as_ptr(), graph.as_ptr()) };
    if !allocated {
        return Err(Qwen3TtsError::InvalidInput(format!(
            "failed to allocate backend graph for {error_message}"
        )));
    }
    for upload in uploads {
        unsafe {
            sys::ggml_backend_tensor_set(
                upload.tensor,
                upload.bytes.as_ptr().cast(),
                0,
                upload.bytes.len(),
            );
        }
    }
    let status = unsafe { sys::ggml_backend_graph_compute(backends.primary_ptr(), graph.as_ptr()) };
    if status != sys::ggml_status_GGML_STATUS_SUCCESS {
        return Err(Qwen3TtsError::InvalidInput(error_message.into()));
    }
    for download in downloads {
        unsafe {
            sys::ggml_backend_tensor_get(
                download.tensor,
                download.bytes.as_mut_ptr().cast(),
                0,
                download.bytes.len(),
            );
        }
    }
    Ok(())
}

fn maybe_log_backend_support(backends: &BackendSet, graph: NonNull<sys::ggml_cgraph>, label: &str) {
    if !backend_debug_enabled() {
        return;
    }

    let n_nodes = unsafe { sys::ggml_graph_n_nodes(graph.as_ptr()) };
    let mut supported = 0usize;
    let mut offloaded = 0usize;
    let mut unsupported_ops = BTreeMap::<String, usize>::new();
    for idx in 0..n_nodes {
        let node = unsafe { sys::ggml_graph_node(graph.as_ptr(), idx) };
        if node.is_null() {
            continue;
        }
        let is_supported = unsafe { sys::ggml_backend_supports_op(backends.primary_ptr(), node) };
        let is_offloaded = unsafe { sys::ggml_backend_offload_op(backends.primary_ptr(), node) };
        if is_supported {
            supported += 1;
        } else {
            let op = unsafe {
                let desc = sys::ggml_op_desc(node);
                if desc.is_null() {
                    "<unknown>".to_string()
                } else {
                    CStr::from_ptr(desc).to_string_lossy().into_owned()
                }
            };
            *unsupported_ops.entry(op).or_default() += 1;
        }
        if is_offloaded {
            offloaded += 1;
        }
    }

    eprintln!(
        "[backend-debug] {label}: nodes={n_nodes} supported={supported} offloaded={offloaded}"
    );
    for (op, count) in unsupported_ops.into_iter().take(12) {
        eprintln!("[backend-debug] unsupported {op}: {count}");
    }
}

fn backend_debug_enabled() -> bool {
    std::env::var_os("QWEN3_TTS_DEBUG_BACKEND").is_some()
}

fn normalize_threads(thread_count: usize) -> i32 {
    max(1, thread_count) as i32
}
