//! Cache Zone: PagedAttention KV cache management (`specs.md` §2.3).

pub mod paged;

pub use paged::{BlockId, KvCacheConfig, KvCacheStats, PagedKvCache};
