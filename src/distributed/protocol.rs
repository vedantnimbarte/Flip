//! Wire protocol for the distributed pipeline (`specs.md` §3.4).
//!
//! A length-prefixed **binary** framing over any `Read`/`Write` (TCP in
//! practice). Hidden-state tensors are sent as raw little-endian `f32` bytes —
//! exactly (`specs.md` §3.4: "raw byte streams") — so a value computed on a
//! worker round-trips bit-for-bit and a distributed forward pass matches a local
//! one exactly. Control messages (Ping/Pong) carry no payload.
//!
//! Frame: `[u32 payload_len][payload]`, where the payload's first byte is a type
//! tag followed by type-specific fields.

use std::io::{self, Read, Write};

const TAG_RUN_SHARD: u8 = 1;
const TAG_SHARD_RESULT: u8 = 2;
const TAG_PING: u8 = 3;
const TAG_PONG: u8 = 4;
const TAG_ERROR: u8 = 5;

/// A protocol message.
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    /// Master → worker: run your layer shard for one token.
    RunShard { position: u64, hidden: Vec<f32> },
    /// Worker → master: the shard's output hidden state.
    ShardResult { hidden: Vec<f32> },
    /// Heartbeat request.
    Ping,
    /// Heartbeat reply.
    Pong,
    /// An error string.
    Error(String),
}

fn floats_to_bytes(v: &[f32], out: &mut Vec<u8>) {
    out.extend_from_slice(&(v.len() as u64).to_le_bytes());
    for &f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
}

fn floats_from_bytes(buf: &[u8], cursor: &mut usize) -> io::Result<Vec<f32>> {
    let n = read_u64(buf, cursor)? as usize;
    let need = n * 4;
    if *cursor + need > buf.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short float payload"));
    }
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let b = &buf[*cursor..*cursor + 4];
        v.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        *cursor += 4;
    }
    Ok(v)
}

fn read_u64(buf: &[u8], cursor: &mut usize) -> io::Result<u64> {
    if *cursor + 8 > buf.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short u64"));
    }
    let mut b = [0u8; 8];
    b.copy_from_slice(&buf[*cursor..*cursor + 8]);
    *cursor += 8;
    Ok(u64::from_le_bytes(b))
}

/// Encode a message into its payload bytes (without the length prefix).
pub fn encode(msg: &Message) -> Vec<u8> {
    let mut out = Vec::new();
    match msg {
        Message::RunShard { position, hidden } => {
            out.push(TAG_RUN_SHARD);
            out.extend_from_slice(&position.to_le_bytes());
            floats_to_bytes(hidden, &mut out);
        }
        Message::ShardResult { hidden } => {
            out.push(TAG_SHARD_RESULT);
            floats_to_bytes(hidden, &mut out);
        }
        Message::Ping => out.push(TAG_PING),
        Message::Pong => out.push(TAG_PONG),
        Message::Error(s) => {
            out.push(TAG_ERROR);
            let b = s.as_bytes();
            out.extend_from_slice(&(b.len() as u64).to_le_bytes());
            out.extend_from_slice(b);
        }
    }
    out
}

/// Decode a message from its payload bytes.
pub fn decode(payload: &[u8]) -> io::Result<Message> {
    let tag = *payload
        .first()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "empty frame"))?;
    let mut cursor = 1usize;
    match tag {
        TAG_RUN_SHARD => {
            let position = read_u64(payload, &mut cursor)?;
            let hidden = floats_from_bytes(payload, &mut cursor)?;
            Ok(Message::RunShard { position, hidden })
        }
        TAG_SHARD_RESULT => {
            let hidden = floats_from_bytes(payload, &mut cursor)?;
            Ok(Message::ShardResult { hidden })
        }
        TAG_PING => Ok(Message::Ping),
        TAG_PONG => Ok(Message::Pong),
        TAG_ERROR => {
            let n = read_u64(payload, &mut cursor)? as usize;
            let s = String::from_utf8_lossy(&payload[cursor..(cursor + n).min(payload.len())])
                .into_owned();
            Ok(Message::Error(s))
        }
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown message tag {other}"),
        )),
    }
}

/// Write a length-prefixed message to `w`.
pub fn write_message(w: &mut impl Write, msg: &Message) -> io::Result<()> {
    let payload = encode(msg);
    w.write_all(&(payload.len() as u32).to_le_bytes())?;
    w.write_all(&payload)?;
    w.flush()
}

/// Read a length-prefixed message from `r`.
pub fn read_message(r: &mut impl Read) -> io::Result<Message> {
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut payload = vec![0u8; len];
    r.read_exact(&mut payload)?;
    decode(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(msg: Message) {
        let mut buf = Vec::new();
        write_message(&mut buf, &msg).unwrap();
        let decoded = read_message(&mut &buf[..]).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn messages_round_trip_exactly() {
        round_trip(Message::Ping);
        round_trip(Message::Pong);
        round_trip(Message::Error("boom".into()));
        round_trip(Message::RunShard {
            position: 7,
            hidden: vec![1.5, -2.25, 0.0, f32::MIN_POSITIVE, 12345.678],
        });
        round_trip(Message::ShardResult {
            hidden: vec![-0.001, 42.0],
        });
    }

    #[test]
    fn floats_are_bit_exact() {
        // Tricky values that a text encoding could perturb.
        let hidden = vec![0.1f32, 0.2, 0.3, 1.0 / 3.0, std::f32::consts::PI];
        let mut buf = Vec::new();
        write_message(&mut buf, &Message::ShardResult { hidden: hidden.clone() }).unwrap();
        let Message::ShardResult { hidden: got } = read_message(&mut &buf[..]).unwrap() else {
            panic!("wrong message");
        };
        assert_eq!(got, hidden); // exact bit equality
    }
}
