//! Query generation: stratified eval dates and dense sweeps.
//! Deterministic: RNG is seeded from (cfg.seed ^ fnv(interval_id)).

use crate::splits::{example_split, fnv1a64};
use crate::types::*;
use crate::year::display_year;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Knobs for stratified eval and sweep generation.
#[derive(Clone, Debug)]
pub struct GenConfig {
    /// Half-width of the "near" bands around each bound.
    pub near_delta: i32,
    /// Reach of the "far_before" band behind the start bound.
    pub far_before_span: i32,
    /// Present-day year; substitutes for an open end bound.
    pub world_end: i32,
    /// First year of a sweep.
    pub sweep_from: i32,
    /// Last year of a sweep (inclusive if landed on).
    pub sweep_to: i32,
    /// Sweep stride in years.
    pub sweep_step: i32,
    /// Master seed; combined with fnv(interval_id) per interval.
    pub seed: u64,
}

impl Default for GenConfig {
    fn default() -> Self {
        Self {
            near_delta: 10,
            far_before_span: 2000,
            world_end: 2026,
            sweep_from: -3000,
            sweep_to: 2026,
            sweep_step: 10,
            seed: 0x1a7e47,
        }
    }
}

/// Which generation pass to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenMode {
    Eval,
    Sweep,
}

/// A prompt template with `{entity_name}` / `{year_display}` placeholders.
/// `holdout` templates are reserved for the test split.
pub struct Template {
    pub id: &'static str,
    pub relation: Relation,
    pub body: &'static str,
    pub holdout: bool,
}

/// Three templates per relation; exactly one holdout each.
pub const TEMPLATES: &[Template] = &[
    Template {
        id: "alive_v1",
        relation: Relation::Alive,
        holdout: false,
        body: "Answer only Yes or No.\n\nWas {entity_name} alive at any point during {year_display}?\n\nAnswer:",
    },
    Template {
        id: "alive_v2",
        relation: Relation::Alive,
        holdout: false,
        body: "Reply with exactly Yes or No.\n\nDid {entity_name} live during the year {year_display}?\n\nAnswer:",
    },
    Template {
        id: "alive_v3",
        relation: Relation::Alive,
        holdout: true,
        body: "Question: Was {entity_name} alive in {year_display}? Answer Yes or No.\n\nAnswer:",
    },
    Template {
        id: "ongoing_v1",
        relation: Relation::Ongoing,
        holdout: false,
        body: "Answer only Yes or No.\n\nWas {entity_name} ongoing at any point during {year_display}?\n\nAnswer:",
    },
    Template {
        id: "ongoing_v2",
        relation: Relation::Ongoing,
        holdout: false,
        body: "Reply with exactly Yes or No.\n\nDid any part of {entity_name} take place in {year_display}?\n\nAnswer:",
    },
    Template {
        id: "ongoing_v3",
        relation: Relation::Ongoing,
        holdout: true,
        body: "Question: Was {entity_name} in progress during {year_display}? Answer Yes or No.\n\nAnswer:",
    },
    Template {
        id: "exists_v1",
        relation: Relation::Exists,
        holdout: false,
        body: "Answer only Yes or No.\n\nDid {entity_name} exist as a political or legal entity during {year_display}?\n\nAnswer:",
    },
    Template {
        id: "exists_v2",
        relation: Relation::Exists,
        holdout: false,
        body: "Reply with exactly Yes or No.\n\nWas {entity_name} in existence at any point during {year_display}?\n\nAnswer:",
    },
    Template {
        id: "exists_v3",
        relation: Relation::Exists,
        holdout: true,
        body: "Question: Did {entity_name} exist in {year_display}? Answer Yes or No.\n\nAnswer:",
    },
    Template {
        id: "active_v1",
        relation: Relation::Active,
        holdout: false,
        body: "Answer only Yes or No.\n\nWas {entity_name} active as an organization during {year_display}?\n\nAnswer:",
    },
    Template {
        id: "active_v2",
        relation: Relation::Active,
        holdout: false,
        body: "Reply with exactly Yes or No.\n\nDid {entity_name} operate at any point during {year_display}?\n\nAnswer:",
    },
    Template {
        id: "active_v3",
        relation: Relation::Active,
        holdout: true,
        body: "Question: Was {entity_name} active in {year_display}? Answer Yes or No.\n\nAnswer:",
    },
    Template {
        id: "available_v1",
        relation: Relation::Available,
        holdout: false,
        body: "Answer only Yes or No.\n\nWas {entity_name} already in existence and available by {year_display}?\n\nAnswer:",
    },
    Template {
        id: "available_v2",
        relation: Relation::Available,
        holdout: false,
        body: "Reply with exactly Yes or No.\n\nCould someone in {year_display} have accessed or used {entity_name}?\n\nAnswer:",
    },
    Template {
        id: "available_v3",
        relation: Relation::Available,
        holdout: true,
        body: "Question: Was {entity_name} available in {year_display}? Answer Yes or No.\n\nAnswer:",
    },
];

/// All templates for a relation (three, one holdout).
pub fn templates_for(relation: Relation) -> Vec<&'static Template> {
    TEMPLATES
        .iter()
        .filter(|t| t.relation == relation)
        .collect()
}

fn seeded_rng(seed: u64, key: &str) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed ^ fnv1a64(key.as_bytes()))
}

fn jitter(rng: &mut ChaCha8Rng, y: i32, lo: i32, hi: i32) -> i32 {
    (y + rng.random_range(-2..=2)).clamp(lo, hi)
}

/// Midpoint of the same-type peer interval nearest to our own midpoint,
/// kept only when it lands outside our interval (gold No) — a date a model
/// could confuse with our entity's era. `None` when our interval has no
/// midpoint or no peer qualifies.
fn era_confusable(iv: &Interval, peers: &[&Interval], world_end: i32) -> Option<i32> {
    let mid = iv.midpoint(world_end)?;
    let (peer_mid, _) = peers
        .iter()
        .filter(|p| p.interval_id != iv.interval_id)
        .filter_map(|p| p.midpoint(world_end).map(|m| (m, p)))
        .min_by_key(|(m, _)| (m - mid).abs())?;
    (iv.label_at(peer_mid) == GoldLabel::No).then_some(peer_mid)
}

fn make_example(iv: &Interval, name: &str, year: i32, band: &str, tpl: &Template) -> Example {
    let label = iv.label_at(year);
    Example {
        example_id: format!(
            "{}:year_{year}:{}:{band}",
            iv.interval_id.replace(':', "_"),
            tpl.id
        ),
        task: "temporal_interval".into(),
        relation: iv.relation,
        subject_id: iv.entity_id.clone(),
        subject_name: name.to_string(),
        year_astronomical: year,
        year_display: display_year(year),
        gold_label: label,
        label_int: label.label_int(),
        interval_start: iv.start_year,
        interval_end: iv.end_year,
        sample_band: band.to_string(),
        template_id: tpl.id.to_string(),
        prompt: tpl
            .body
            .replace("{entity_name}", name)
            .replace("{year_display}", &display_year(year)),
        source_ids: vec![iv.source_id.clone()],
        label_confidence: iv.confidence,
        split: example_split(&iv.entity_id, tpl.holdout).to_string(),
    }
}

/// Eight stratified eval examples for one interval: interior, near/far
/// bands around the bounds, plus one era-confusable (or far fallback) date.
/// Open-ended intervals reallocate the after-bands into interior and
/// near_before. Empty for intervals with no start bound; inverted
/// intervals (start > end, which `normalize` does not reject) also yield
/// no examples rather than panicking in `jitter`'s clamp.
pub fn gen_eval(iv: &Interval, name: &str, peers: &[&Interval], cfg: &GenConfig) -> Vec<Example> {
    if let (Some(a), Some(b)) = (iv.start_year, iv.end_year)
        && a > b
    {
        return vec![];
    }
    let templates = templates_for(iv.relation);
    let mut rng = seeded_rng(cfg.seed, &iv.interval_id);
    let mut dated: Vec<(i32, &str)> = Vec::new();
    match (iv.start_year, iv.end_year) {
        (Some(a), Some(b)) => {
            for frac in [0.2f64, 0.5, 0.8] {
                let y = a + (((b - a) as f64 * frac) as i32);
                dated.push((jitter(&mut rng, y, a, b), "interior"));
            }
            dated.push((rng.random_range((a - cfg.near_delta)..a), "near_before"));
            dated.push((
                rng.random_range((b + 1)..=(b + cfg.near_delta)),
                "near_after",
            ));
            dated.push((
                rng.random_range((a - cfg.far_before_span)..(a - cfg.near_delta)),
                "far_before",
            ));
            if b + cfg.near_delta < cfg.world_end {
                dated.push((
                    rng.random_range((b + cfg.near_delta + 1)..=cfg.world_end),
                    "far_after",
                ));
            } else {
                dated.push((
                    rng.random_range((a - cfg.far_before_span)..(a - cfg.near_delta)),
                    "far_fallback",
                ));
            }
            match era_confusable(iv, peers, cfg.world_end) {
                Some(y) => dated.push((y, "era_confusable")),
                None => dated.push((
                    rng.random_range((a - cfg.far_before_span)..(a - cfg.near_delta)),
                    "far_fallback",
                )),
            }
        }
        (Some(a), None) => {
            for frac in [0.15f64, 0.4, 0.65, 0.9] {
                let y = a + (((cfg.world_end - a) as f64 * frac) as i32);
                dated.push((jitter(&mut rng, y, a, cfg.world_end), "interior"));
            }
            // Two near_before draws: distinct years/templates, so ids differ.
            dated.push((rng.random_range((a - cfg.near_delta)..a), "near_before"));
            dated.push((rng.random_range((a - cfg.near_delta)..a), "near_before"));
            dated.push((
                rng.random_range((a - cfg.far_before_span)..(a - cfg.near_delta)),
                "far_before",
            ));
            match era_confusable(iv, peers, cfg.world_end) {
                Some(y) => dated.push((y, "era_confusable")),
                None => dated.push((
                    rng.random_range((a - cfg.far_before_span)..(a - cfg.near_delta)),
                    "far_fallback",
                )),
            }
        }
        (None, _) => return vec![],
    }
    dated
        .iter()
        .enumerate()
        .map(|(i, (y, band))| {
            // Slot suffix keeps ids unique when (template, band, year) collide.
            let mut ex = make_example(iv, name, *y, band, templates[i % templates.len()]);
            ex.example_id = format!("{}:{i}", ex.example_id);
            ex
        })
        .collect()
}

/// Dense sweep across the whole axis at `cfg.sweep_step`, all in the
/// "sweep" split/band, using the relation's first (non-holdout) template.
pub fn gen_sweep(iv: &Interval, name: &str, cfg: &GenConfig) -> Vec<Example> {
    let tpl = templates_for(iv.relation)[0];
    let mut out = Vec::new();
    let mut y = cfg.sweep_from;
    while y <= cfg.sweep_to {
        let mut ex = make_example(iv, name, y, "sweep", tpl);
        ex.split = "sweep".into();
        out.push(ex);
        y += cfg.sweep_step;
    }
    out
}

/// Generate for every interval. Peers for era-confusable sampling are chosen
/// among entities of the same entity_type; peer lists are built once per
/// type and skipped entirely in Sweep mode (gen_sweep ignores peers).
pub fn generate(
    entities: &[Entity],
    intervals: &[Interval],
    cfg: &GenConfig,
    mode: GenMode,
) -> Vec<Example> {
    use std::collections::HashMap;
    let by_id: HashMap<&str, &Entity> =
        entities.iter().map(|e| (e.entity_id.as_str(), e)).collect();
    let peers_by_type: HashMap<EntityType, Vec<&Interval>> = match mode {
        GenMode::Eval => {
            let type_of: HashMap<&str, EntityType> = entities
                .iter()
                .map(|e| (e.entity_id.as_str(), e.entity_type))
                .collect();
            let mut m: HashMap<EntityType, Vec<&Interval>> = HashMap::new();
            for iv in intervals {
                if let Some(t) = type_of.get(iv.entity_id.as_str()) {
                    m.entry(*t).or_default().push(iv);
                }
            }
            m
        }
        GenMode::Sweep => HashMap::new(),
    };
    let mut out = Vec::new();
    for iv in intervals {
        let Some(entity) = by_id.get(iv.entity_id.as_str()) else {
            continue;
        };
        match mode {
            GenMode::Eval => {
                let peers = peers_by_type
                    .get(&entity.entity_type)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                out.extend(gen_eval(iv, &entity.canonical_name, peers, cfg));
            }
            GenMode::Sweep => out.extend(gen_sweep(iv, &entity.canonical_name, cfg)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::tests_helpers::*;
    use std::collections::HashMap;

    fn cfg() -> GenConfig {
        GenConfig::default()
    }

    fn bands(examples: &[Example]) -> HashMap<&str, usize> {
        let mut m = HashMap::new();
        for e in examples {
            *m.entry(e.sample_band.as_str()).or_default() += 1;
        }
        m
    }

    #[test]
    fn three_templates_per_relation_one_holdout() {
        for r in [
            Relation::Alive,
            Relation::Ongoing,
            Relation::Exists,
            Relation::Active,
            Relation::Available,
        ] {
            let ts = templates_for(r);
            assert_eq!(ts.len(), 3, "{r:?}");
            assert_eq!(ts.iter().filter(|t| t.holdout).count(), 1, "{r:?}");
        }
    }

    #[test]
    fn eval_generates_8_stratified_dates() {
        let iv = caesar_interval();
        let ex = gen_eval(&iv, "Julius Caesar", &[], &cfg());
        assert_eq!(ex.len(), 8);
        let b = bands(&ex);
        assert_eq!(b["interior"], 3);
        assert_eq!(b["near_before"], 1);
        assert_eq!(b["near_after"], 1);
        assert_eq!(b["far_before"], 1);
        assert_eq!(b["far_after"], 1);
        assert_eq!(
            b.get("era_confusable").unwrap_or(&0) + b.get("far_fallback").unwrap_or(&0),
            1
        );
        for e in &ex {
            match e.sample_band.as_str() {
                "interior" => {
                    assert!((-99..=-43).contains(&e.year_astronomical));
                    assert_eq!(e.gold_label, GoldLabel::Yes);
                }
                "near_before" => assert!((-109..=-100).contains(&e.year_astronomical)),
                "near_after" => assert!((-42..=-33).contains(&e.year_astronomical)),
                _ => {}
            }
        }
    }

    #[test]
    fn eval_is_deterministic() {
        let iv = caesar_interval();
        assert_eq!(
            gen_eval(&iv, "Julius Caesar", &[], &cfg()),
            gen_eval(&iv, "Julius Caesar", &[], &cfg())
        );
    }

    #[test]
    fn open_ended_reallocates_after_bands() {
        let iv = aeneid_interval(); // start -18, end None
        let ex = gen_eval(&iv, "Aeneid", &[], &cfg());
        assert_eq!(ex.len(), 8);
        let b = bands(&ex);
        assert_eq!(b["interior"], 4);
        assert_eq!(b["near_before"], 2);
        assert!(!b.contains_key("near_after"));
        assert!(!b.contains_key("far_after"));
    }

    #[test]
    fn era_confusable_picks_peer_midpoint_outside_own_interval() {
        let iv = caesar_interval();
        let mut peer = caesar_interval();
        peer.interval_id = "alive:wd:Q945".into();
        peer.entity_id = "wd:Q945".into();
        peer.start_year = Some(-62); // Augustus
        peer.end_year = Some(14);
        let ex = gen_eval(&iv, "Julius Caesar", &[&peer], &cfg());
        let era = ex
            .iter()
            .find(|e| e.sample_band == "era_confusable")
            .unwrap();
        assert_eq!(era.year_astronomical, (-62 + 14) / 2);
        assert_eq!(era.gold_label, GoldLabel::No);
    }

    #[test]
    fn holdout_template_forces_test_split() {
        let iv = caesar_interval();
        let ex = gen_eval(&iv, "Julius Caesar", &[], &cfg());
        for e in &ex {
            let t = templates_for(iv.relation)
                .into_iter()
                .find(|t| t.id == e.template_id)
                .unwrap();
            if t.holdout {
                assert_eq!(e.split, "test");
            }
        }
    }

    #[test]
    fn sweep_covers_axis_with_sweep_split() {
        let iv = caesar_interval();
        let ex = gen_sweep(&iv, "Julius Caesar", &cfg());
        assert_eq!(ex.len(), (2026 + 3000) / 10 + 1);
        assert!(
            ex.iter()
                .all(|e| e.split == "sweep" && e.sample_band == "sweep")
        );
        assert_eq!(ex[0].year_astronomical, -3000);
        assert_eq!(ex[1].year_astronomical, -2990);
        assert!(ex.iter().any(|e| e.gold_label == GoldLabel::Yes));
        assert!(ex.iter().any(|e| e.gold_label == GoldLabel::No));
    }

    #[test]
    fn prompt_renders_name_and_year_without_placeholders() {
        let iv = caesar_interval();
        let ex = gen_eval(&iv, "Julius Caesar", &[], &cfg());
        for e in &ex {
            assert!(e.prompt.contains("Julius Caesar"), "{}", e.prompt);
            assert!(!e.prompt.contains('{'), "{}", e.prompt);
        }
    }

    #[test]
    fn inverted_interval_yields_no_examples() {
        let mut iv = caesar_interval();
        iv.start_year = Some(-43);
        iv.end_year = Some(-99);
        assert!(gen_eval(&iv, "Julius Caesar", &[], &cfg()).is_empty());
    }

    #[test]
    fn point_interval_event_generates_8() {
        let mut iv = caesar_interval();
        iv.interval_id = "ongoing:wd:Q37839".into();
        iv.relation = Relation::Ongoing;
        iv.start_year = Some(1066);
        iv.end_year = Some(1066);
        let ex = gen_eval(&iv, "Battle of Hastings", &[], &cfg());
        assert_eq!(ex.len(), 8);
        for e in ex.iter().filter(|e| e.sample_band == "interior") {
            assert_eq!(e.year_astronomical, 1066);
            assert_eq!(e.gold_label, GoldLabel::Yes);
        }
    }

    #[test]
    fn missing_start_yields_no_examples() {
        let iv = Interval {
            start_year: None,
            ..caesar_interval()
        };
        assert!(gen_eval(&iv, "Julius Caesar", &[], &cfg()).is_empty());
    }

    #[test]
    fn far_fallback_when_near_world_end() {
        let mut iv = caesar_interval();
        iv.end_year = Some(2020);
        let ex = gen_eval(&iv, "Julius Caesar", &[], &cfg());
        let b = bands(&ex);
        assert!(b.contains_key("far_fallback"));
        assert!(!b.contains_key("far_after"));
    }

    #[test]
    fn eval_example_ids_are_unique_per_interval() {
        use std::collections::HashSet;
        for (iv, name) in [
            (aeneid_interval(), "Aeneid"),
            (caesar_interval(), "Julius Caesar"),
        ] {
            let ex = gen_eval(&iv, name, &[], &cfg());
            assert_eq!(ex.len(), 8);
            let ids: HashSet<&str> = ex.iter().map(|e| e.example_id.as_str()).collect();
            assert_eq!(ids.len(), 8, "{name}");
        }
    }

    #[test]
    fn generate_groups_peers_by_type_and_skips_missing() {
        let caesar = caesar_entity(); // Person, wd:Q1048
        let mut augustus = caesar_entity();
        augustus.entity_id = "wd:Q945".into();
        augustus.canonical_name = "Augustus".into();
        let aeneid = aeneid_entity(); // Work

        let caesar_iv = caesar_interval(); // alive, wd:Q1048, -99..=-43
        let mut augustus_iv = caesar_interval();
        augustus_iv.interval_id = "alive:wd:Q945".into();
        augustus_iv.entity_id = "wd:Q945".into();
        augustus_iv.start_year = Some(-62);
        augustus_iv.end_year = Some(14);
        let aeneid_iv = aeneid_interval();
        let mut orphan_iv = caesar_interval();
        orphan_iv.interval_id = "alive:wd:Q999".into();
        orphan_iv.entity_id = "wd:Q999".into(); // no matching entity

        let entities = [caesar, augustus, aeneid];
        let intervals = [caesar_iv, augustus_iv, aeneid_iv, orphan_iv];
        let out = generate(&entities, &intervals, &cfg(), GenMode::Eval);
        assert!(out.iter().all(|e| e.subject_id != "wd:Q999"));
        assert_eq!(out.len(), 24); // 3 resolvable intervals * 8
        let era = |subject: &str, year: i32| {
            out.iter().any(|e| {
                e.subject_id == subject
                    && e.sample_band == "era_confusable"
                    && e.year_astronomical == year
            })
        };
        // The two Person entities era-confuse each other at each other's midpoint.
        assert!(era("wd:Q1048", (-62 + 14) / 2));
        assert!(era("wd:Q945", -71)); // caesar midpoint
    }
}
