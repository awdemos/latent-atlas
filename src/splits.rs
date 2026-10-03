//! Deterministic splits via FNV-1a — stable across runs, platforms, and
//! toolchain versions (unlike std's RandomState). No RNG state to preserve.

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// 60% train / 20% dev / 20% test, entity-disjoint by construction.
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
    fn split_is_deterministic() {
        for i in 0..100 {
            let id = format!("wd:Q{i}");
            assert_eq!(entity_split(&id), entity_split(&id));
        }
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
}
