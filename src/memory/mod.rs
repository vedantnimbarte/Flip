//! Host memory management: page discovery + page-locked DMA staging buffers.

pub mod page;
pub mod pinned;

pub use page::{page_size, round_up_to_page};
pub use pinned::{PinKind, PinnedBuffer};
