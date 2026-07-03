//! Speculative decoding (`PRD.md` §3.3).
//!
//! A small, cheap **draft** model proposes `gamma` tokens; the large **target**
//! model verifies them. With greedy sampling the rule is exact: accept each draft
//! token while it equals the target's greedy choice; at the first mismatch take
//! the target's token instead; if all `gamma` are accepted, append the target's
//! next ("bonus") token. This yields between 1 and `gamma + 1` tokens per round
//! and — crucially — produces **exactly** the same sequence as plain target-greedy
//! decoding, just faster when the draft guesses well.
//!
//! The speed win comes from verifying all `gamma` positions in a single *batched*
//! target forward pass. `flip`'s CPU forward is single-token, so this
//! implementation verifies sequentially (one target step per position) — the
//! accept/reject logic and its exactness are identical; only the wall-clock
//! saving awaits a batched kernel. Acceptance statistics are reported so the
//! benefit is measurable.

use crate::error::Result;
use crate::forward::ComputeKernel;
use crate::generate::{GenerationConfig, Generator, Sampler};

/// Outcome of a speculative generation.
#[derive(Debug, Clone)]
pub struct SpeculativeResult {
    /// The generated continuation (exactly target-greedy).
    pub tokens: Vec<u32>,
    /// Draft tokens proposed across all rounds.
    pub proposed: usize,
    /// Draft tokens accepted (matched the target).
    pub accepted: usize,
}

impl SpeculativeResult {
    /// Fraction of proposed draft tokens the target accepted (0.0–1.0).
    pub fn acceptance_rate(&self) -> f64 {
        if self.proposed == 0 {
            0.0
        } else {
            self.accepted as f64 / self.proposed as f64
        }
    }
}

/// A speculative decoder pairing a target and a draft generator.
pub struct SpeculativeDecoder<T: ComputeKernel, D: ComputeKernel> {
    target: Generator<T>,
    draft: Generator<D>,
    gamma: usize,
}

impl<T: ComputeKernel, D: ComputeKernel> SpeculativeDecoder<T, D> {
    /// Pair a `target` with a `draft`, proposing `gamma` tokens per round.
    pub fn new(target: Generator<T>, draft: Generator<D>, gamma: usize) -> Self {
        Self {
            target,
            draft,
            gamma: gamma.max(1),
        }
    }

    /// The target's greedy next token given `seq`.
    fn target_next(&self, seq: &[u32]) -> Result<u32> {
        let cfg = GenerationConfig {
            max_new_tokens: 1,
            eos_token: None,
            sampler: Sampler::Greedy,
        };
        Ok(self.target.generate(seq, &cfg)?[0])
    }

    /// Generate up to `max_new_tokens` tokens for `prompt`.
    pub fn generate(&self, prompt: &[u32], max_new_tokens: usize) -> Result<SpeculativeResult> {
        let mut seq = prompt.to_vec();
        let mut out: Vec<u32> = Vec::new();
        let (mut proposed, mut accepted) = (0usize, 0usize);

        while out.len() < max_new_tokens {
            // Propose at most `gamma`, and never more than the tokens still
            // wanted (so the length cap can't masquerade as a rejection).
            let remaining = max_new_tokens - out.len();
            let draft_cfg = GenerationConfig {
                max_new_tokens: self.gamma.min(remaining),
                eos_token: None,
                sampler: Sampler::Greedy,
            };
            let draft_tokens = self.draft.generate(&seq, &draft_cfg)?;
            proposed += draft_tokens.len();

            // Verify each against the target's greedy choice.
            let mut all_accepted = true;
            for &dt in &draft_tokens {
                if out.len() >= max_new_tokens {
                    all_accepted = false;
                    break;
                }
                let target_tok = self.target_next(&seq)?;
                if target_tok == dt {
                    seq.push(dt);
                    out.push(dt);
                    accepted += 1;
                } else {
                    // Mismatch: take the target's correction and end the round.
                    seq.push(target_tok);
                    out.push(target_tok);
                    all_accepted = false;
                    break;
                }
            }

            // All draft tokens accepted → append the target's bonus token.
            if all_accepted && out.len() < max_new_tokens {
                let bonus = self.target_next(&seq)?;
                seq.push(bonus);
                out.push(bonus);
            }

            // Progress guard (draft never returns empty for gamma >= 1).
            if draft_tokens.is_empty() {
                let t = self.target_next(&seq)?;
                seq.push(t);
                out.push(t);
            }
        }

        out.truncate(max_new_tokens);
        Ok(SpeculativeResult {
            tokens: out,
            proposed,
            accepted,
        })
    }
}
