//! System page-size discovery.
//!
//! Page-locked host buffers must be aligned to (and sized in multiples of) the
//! OS page so that the PCIe DMA controller can pin whole pages. We query the
//! real value per-platform rather than assuming 4 KiB, since huge-page and
//! non-x86 systems differ.

use std::sync::OnceLock;

static PAGE_SIZE: OnceLock<usize> = OnceLock::new();

/// The system memory page size in bytes (cached after first query).
pub fn page_size() -> usize {
    *PAGE_SIZE.get_or_init(query_page_size)
}

/// Round `n` up to the next multiple of the system page size.
pub fn round_up_to_page(n: usize) -> usize {
    let p = page_size();
    // `n + p - 1` cannot overflow for any realistic allocation request; saturating
    // guards the theoretical edge near usize::MAX.
    n.saturating_add(p - 1) / p * p
}

#[cfg(windows)]
fn query_page_size() -> usize {
    #[repr(C)]
    struct SystemInfo {
        w_processor_architecture: u16,
        w_reserved: u16,
        dw_page_size: u32,
        lp_minimum_application_address: *mut core::ffi::c_void,
        lp_maximum_application_address: *mut core::ffi::c_void,
        dw_active_processor_mask: usize,
        dw_number_of_processors: u32,
        dw_processor_type: u32,
        dw_allocation_granularity: u32,
        w_processor_level: u16,
        w_processor_revision: u16,
    }

    extern "system" {
        fn GetSystemInfo(info: *mut SystemInfo);
    }

    // SAFETY: GetSystemInfo fully initializes the struct it is handed.
    let mut info: SystemInfo = unsafe { core::mem::zeroed() };
    unsafe { GetSystemInfo(&mut info) };
    let size = info.dw_page_size as usize;
    if size == 0 {
        4096
    } else {
        size
    }
}

#[cfg(unix)]
fn query_page_size() -> usize {
    // SAFETY: sysconf with a valid name is always safe; it returns -1 on error,
    // which we fall back from.
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if size <= 0 {
        4096
    } else {
        size as usize
    }
}

#[cfg(not(any(windows, unix)))]
fn query_page_size() -> usize {
    4096
}
