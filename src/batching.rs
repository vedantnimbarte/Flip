//! Continuous batching (`PRD.md` §3.3).
//!
//! Instead of running requests one after another, the scheduler keeps up to
//! `max_batch` generations **in flight at once** and advances every active one
//! by a single token per tick, admitting queued requests into slots as they free
//! up (rather than waiting for a whole batch to finish). This is the
//! "continuous" / in-flight batching that keeps the engine busy under a stream
//! of requests.
//!
//! Each request runs in its own [`GenerationSession`] with independent KV state,
//! so interleaving is transparent: a request's output is identical to running it
//! alone. On a batched forward kernel the per-tick step would fuse all active
//! sequences into one matmul; here it steps them in a loop — same scheduling,
//! same output, the fused speedup awaiting a batch kernel.

use crate::error::Result;
use crate::forward::ComputeKernel;
use crate::generate::{GenerationSession, Generator, Sampler};
use std::collections::VecDeque;

/// A queued request awaiting admission.
struct Pending {
    id: u64,
    prompt: Vec<u32>,
    max_new_tokens: usize,
    eos: Option<u32>,
}

/// An in-flight generation occupying a batch slot.
struct Active<'a, K: ComputeKernel> {
    id: u64,
    session: GenerationSession<'a, K>,
    remaining: usize,
    eos: Option<u32>,
}

/// A completed request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finished {
    pub id: u64,
    pub tokens: Vec<u32>,
}

/// What one scheduler tick produced: `(request id, token)` pairs emitted this
/// step, and the ids of requests that finished (their last token is in
/// `produced`). Streaming consumers forward `produced` and close on `finished`.
#[derive(Debug, Clone, Default)]
pub struct Tick {
    pub produced: Vec<(u64, u32)>,
    pub finished: Vec<u64>,
}

/// A continuous-batching scheduler over a borrowed generator.
pub struct BatchScheduler<'a, K: ComputeKernel> {
    generator: &'a Generator<K>,
    max_batch: usize,
    pending: VecDeque<Pending>,
    active: Vec<Active<'a, K>>,
}

impl<'a, K: ComputeKernel> BatchScheduler<'a, K> {
    /// Create a scheduler running at most `max_batch` concurrent generations.
    pub fn new(generator: &'a Generator<K>, max_batch: usize) -> Self {
        Self {
            generator,
            max_batch: max_batch.max(1),
            pending: VecDeque::new(),
            active: Vec::new(),
        }
    }

    /// Queue a request. Errors on an empty prompt.
    pub fn submit(
        &mut self,
        id: u64,
        prompt: Vec<u32>,
        max_new_tokens: usize,
        eos: Option<u32>,
    ) -> Result<()> {
        if prompt.is_empty() {
            return Err(crate::error::FlipError::InvalidConfig("prompt is empty".into()));
        }
        self.pending.push_back(Pending {
            id,
            prompt,
            max_new_tokens,
            eos,
        });
        Ok(())
    }

    /// Whether any request is pending or in flight.
    pub fn has_work(&self) -> bool {
        !self.pending.is_empty() || !self.active.is_empty()
    }

    /// In-flight request count.
    pub fn active_len(&self) -> usize {
        self.active.len()
    }

    /// Fill free slots from the pending queue (prefilling each new session).
    /// Returns ids of requests that finished immediately (zero max tokens).
    fn admit(&mut self) -> Result<Vec<u64>> {
        let mut zero_finished = Vec::new();
        while self.active.len() < self.max_batch {
            let Some(p) = self.pending.pop_front() else { break };
            if p.max_new_tokens == 0 {
                zero_finished.push(p.id);
                continue;
            }
            let session = self.generator.start_session(&p.prompt, Sampler::Greedy)?;
            self.active.push(Active {
                id: p.id,
                session,
                remaining: p.max_new_tokens,
                eos: p.eos,
            });
        }
        Ok(zero_finished)
    }

    /// One scheduler tick: admit queued requests, advance every active session by
    /// one token, and retire any that finished. Returns the tokens produced and
    /// the ids that completed this tick (for streaming).
    pub fn step(&mut self) -> Result<Tick> {
        let zero_finished = self.admit()?;
        let mut tick = Tick::default();
        tick.finished = zero_finished;
        let mut still_active = Vec::with_capacity(self.active.len());
        for mut a in self.active.drain(..) {
            let token = a.session.step()?;
            tick.produced.push((a.id, token));
            a.remaining -= 1;
            if a.remaining == 0 || Some(token) == a.eos {
                tick.finished.push(a.id);
            } else {
                still_active.push(a);
            }
        }
        self.active = still_active;
        Ok(tick)
    }

    /// Run ticks until every request has completed, returning the results (in
    /// completion order). Convenience wrapper over [`step`](Self::step).
    pub fn run(&mut self) -> Result<Vec<Finished>> {
        use std::collections::HashMap;
        let mut outputs: HashMap<u64, Vec<u32>> = HashMap::new();
        let mut results = Vec::new();
        while self.has_work() {
            let tick = self.step()?;
            for (id, token) in tick.produced {
                outputs.entry(id).or_default().push(token);
            }
            for id in tick.finished {
                results.push(Finished {
                    id,
                    tokens: outputs.remove(&id).unwrap_or_default(),
                });
            }
        }
        Ok(results)
    }
}
