//! Distributed multi-server topology (`specs.md` §3.4): a master coordinator
//! streaming a pipelined forward pass across worker nodes, with heartbeats and a
//! local CPU-RAM fallback for fault tolerance.
//!
//! Transport is a length-prefixed binary protocol over TCP ([`protocol`]) rather
//! than gRPC/Protobuf, so hidden-state tensors round-trip bit-for-bit and the
//! whole layer is dependency-free and testable over localhost.

pub mod coordinator;
pub mod protocol;
pub mod shard;
pub mod worker;

pub use coordinator::{Coordinator, ShardRoute};
pub use protocol::{read_message, write_message, Message};
pub use shard::{partition_layers, LayerShard};
pub use worker::Worker;
