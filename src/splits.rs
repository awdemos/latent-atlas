//! Deterministic splits via FNV-1a — stable across runs, platforms, and
//! toolchain versions (unlike std's RandomState). No RNG state to preserve.

/// FNV-1a 64-bit hash; exposed for pinning/reuse.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// 60% train / 20% dev / 20% test, entity-disjoint by construction.
/// The split is a function of the exact id string — "Q42" and "wd:Q42"
/// partition differently; callers must pass the canonical entity_id.
pub fn entity_split(entity_id: &str) -> &'static str {
    match fnv1a64(entity_id.as_bytes()) % 10 {
        0..=5 => "train",
        6..=7 => "dev",
        _ => "test",
    }
}

/// Holdout templates always land in test regardless of the entity split.
pub fn example_split(entity_id: &str, holdout_template: bool) -> &'static str {
    if holdout_template {
        "test"
    } else {
        entity_split(entity_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn split_fixed_vectors_are_stable() {
        assert_eq!(fnv1a64(b""), 0xcbf29ce484222325);
        assert_eq!(entity_split("wd:Q1048"), "dev");
        assert_eq!(entity_split("wd:Q42"), "dev");
        assert_eq!(entity_split("wd:Q135000000"), "train");
    }

    #[test]
    fn split_distribution_is_roughly_60_20_20() {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for i in 0..5000 {
            *counts.entry(entity_split(&format!("wd:Q{i}"))).or_default() += 1;
        }
        let frac = |k: &str| *counts.get(k).unwrap_or(&0) as f64 / 5000.0;
        assert!((0.5..=0.7).contains(&frac("train")), "{}", frac("train"));
        assert!((0.1..=0.3).contains(&frac("dev")), "{}", frac("dev"));
        assert!((0.1..=0.3).contains(&frac("test")), "{}", frac("test"));
    }

    #[test]
    fn holdout_template_always_lands_in_test() {
        for i in 0..100 {
            assert_eq!(example_split(&format!("wd:Q{i}"), true), "test");
        }
    }

    #[test]
    fn example_split_delegates_when_not_holdout() {
        for id in ["wd:Q42", "wd:Q1048", "wd:Q135000000"] {
            assert_eq!(example_split(id, false), entity_split(id));
        }
    }
}
