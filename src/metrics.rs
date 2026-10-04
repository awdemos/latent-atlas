//! Metrics: never accuracy alone. Sweep-split rows feed boundary/IoU/smoothness
//! but are excluded from top-line classification metrics.

use crate::types::{GoldLabel, Response};
use serde::Serialize;
use std::collections::BTreeMap;

/// Fraction of rows where the thresholded prediction (P>=0.5) matches the
/// gold label; 0.0 on empty input.
pub fn accuracy(points: &[(f64, u8)]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    points
        .iter()
        .filter(|(p, y)| (*p >= 0.5) == (*y == 1))
        .count() as f64
        / points.len() as f64
}

/// Mean squared error of p_yes against the 0/1 label; 0.0 on empty input.
pub fn brier(points: &[(f64, u8)]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    points
        .iter()
        .map(|(p, y)| (p - *y as f64).powi(2))
        .sum::<f64>()
        / points.len() as f64
}

/// Mean binary cross-entropy, probabilities clamped to [1e-15, 1-1e-15];
/// 0.0 on empty input.
pub fn log_loss(points: &[(f64, u8)]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    let eps = 1e-15;
    points
        .iter()
        .map(|(p, y)| {
            let p = p.clamp(eps, 1.0 - eps);
            if *y == 1 { -p.ln() } else { -(1.0 - p).ln() }
        })
        .sum::<f64>()
        / points.len() as f64
}

/// Rank-based AUROC with average ranks for ties; None when single-class.
pub fn auroc(points: &[(f64, u8)]) -> Option<f64> {
    let n_pos = points.iter().filter(|(_, y)| *y == 1).count();
    let n_neg = points.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return None;
    }
    let mut idx: Vec<usize> = (0..points.len()).collect();
    idx.sort_by(|&i, &j| points[i].0.partial_cmp(&points[j].0).unwrap());
    let mut rank_sum = 0.0;
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && points[idx[j + 1]].0 == points[idx[i]].0 {
            j += 1;
        }
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            if points[idx[k]].1 == 1 {
                rank_sum += avg_rank;
            }
        }
        i = j + 1;
    }
    Some((rank_sum - n_pos as f64 * (n_pos + 1) as f64 / 2.0) / (n_pos * n_neg) as f64)
}

/// |predicted - true| at the P>=0.5 boundaries of an entity's year curve.
/// Returns (start_error, end_error); None if no year is predicted active.
pub fn boundary_error(
    curve: &[(i32, f64)],
    true_start: i32,
    true_end: Option<i32>,
    world_end: i32,
) -> Option<(i32, i32)> {
    let active: Vec<i32> = curve
        .iter()
        .filter(|(_, p)| *p >= 0.5)
        .map(|(y, _)| *y)
        .collect();
    let a_hat = *active.iter().min()?;
    let b_hat = *active.iter().max()?;
    Some((
        (a_hat - true_start).abs(),
        (b_hat - true_end.unwrap_or(world_end)).abs(),
    ))
}

/// Mean absolute year-to-year change in P(active); 0 for a flat curve.
pub fn smoothness(curve: &[(i32, f64)]) -> f64 {
    if curve.len() < 2 {
        return 0.0;
    }
    curve
        .windows(2)
        .map(|w| (w[1].1 - w[0].1).abs())
        .sum::<f64>()
        / (curve.len() - 1) as f64
}

/// IoU between the predicted-active year span (P>=0.5) and the true interval;
/// span-based (min/max of predicted-active years); pair with smoothness for
/// hole detection.
pub fn interval_iou(curve: &[(i32, f64)], a: i32, b: Option<i32>, world_end: i32) -> f64 {
    let b = b.unwrap_or(world_end);
    let pred: Vec<i32> = curve
        .iter()
        .filter(|(_, p)| *p >= 0.5)
        .map(|(y, _)| *y)
        .collect();
    let (Some(&pa), Some(&pb)) = (pred.iter().min(), pred.iter().max()) else {
        return 0.0;
    };
    let inter = (pb.min(b) - pa.max(a) + 1).max(0) as f64;
    let union = (pb.max(b) - pa.min(a) + 1) as f64;
    inter / union
}

/// Classification metrics for one slice of eval rows; unknown-labeled rows
/// are counted in `n_excluded_unknown` and excluded from `n` and the metrics.
#[derive(Clone, Debug, Default, Serialize)]
pub struct GroupMetrics {
    pub n: usize,
    pub n_excluded_unknown: usize,
    pub accuracy: f64,
    pub auroc: Option<f64>,
    pub log_loss: f64,
    pub brier: f64,
}

/// Full report for one model: overall and grouped classification metrics
/// plus per-interval curve means (boundary error, IoU, smoothness).
#[derive(Clone, Debug, Serialize)]
pub struct MetricsReport {
    pub model: String,
    pub overall: GroupMetrics,
    pub by_relation: BTreeMap<String, GroupMetrics>,
    pub by_band: BTreeMap<String, GroupMetrics>,
    pub by_split: BTreeMap<String, GroupMetrics>,
    pub mean_boundary_error_start: Option<f64>,
    pub mean_boundary_error_end: Option<f64>,
    pub mean_interval_iou: f64,
    pub mean_smoothness: f64,
}

fn group_metrics(rs: &[&Response]) -> GroupMetrics {
    let n_excluded_unknown = rs
        .iter()
        .filter(|r| r.example.gold_label == GoldLabel::UnknownOrAmbiguous)
        .count();
    let points: Vec<(f64, u8)> = rs
        .iter()
        .filter(|r| r.example.gold_label != GoldLabel::UnknownOrAmbiguous)
        .map(|r| (r.p_yes, r.example.label_int))
        .collect();
    GroupMetrics {
        n: points.len(),
        n_excluded_unknown,
        accuracy: accuracy(&points),
        auroc: auroc(&points),
        log_loss: log_loss(&points),
        brier: brier(&points),
    }
}

/// Partition rows by `key` and compute `GroupMetrics` per partition.
fn group_by<'a>(
    rs: &[&'a Response],
    key: impl Fn(&'a Response) -> String,
) -> BTreeMap<String, GroupMetrics> {
    let mut groups: BTreeMap<String, Vec<&Response>> = BTreeMap::new();
    for r in rs {
        groups.entry(key(r)).or_default().push(r);
    }
    groups
        .into_iter()
        .map(|(k, v)| (k, group_metrics(&v)))
        .collect()
}

/// Aggregate probe responses into a single-model metrics report. `world_end`
/// should match the `GenConfig` used at generation time. The model name is
/// taken from the first row: responses for one report must come from a
/// single-model probe file.
pub fn compute_report(responses: &[Response], world_end: i32) -> MetricsReport {
    let eval: Vec<&Response> = responses
        .iter()
        .filter(|r| r.example.split != "sweep")
        .collect();
    let model = responses
        .first()
        .map(|r| r.model.clone())
        .unwrap_or_default();

    // Per-interval curves (sweep included) for boundary/IoU/smoothness.
    // relation + entity identifies one interval (normalize's interval_id
    // scheme), so an entity with several relations gets separate curves.
    let mut by_interval: BTreeMap<String, Vec<&Response>> = BTreeMap::new();
    for r in responses {
        by_interval
            .entry(format!(
                "{}:{}",
                r.example.relation.as_str(),
                r.example.subject_id
            ))
            .or_default()
            .push(r);
    }
    let (mut se, mut ee, mut ious, mut smooth) = (vec![], vec![], vec![], vec![]);
    for rs in by_interval.values() {
        let mut curve: Vec<(i32, bool, &str, f64)> = rs
            .iter()
            .map(|r| {
                (
                    r.example.year_astronomical,
                    r.example.split != "sweep",
                    r.example.example_id.as_str(),
                    r.p_yes,
                )
            })
            .collect();
        // Sweep rows win year-ties by construction: (year, is_eval) sorts
        // sweep rows before eval rows at equal years (false < true). Remaining
        // eval-eval ties break on example_id, so the surviving row is
        // data-dependent, not probe completion order — buffer_unordered makes
        // that order nondeterministic, and an input-order-dependent survivor
        // would flip IoU for identical data. The stable sort + dedup keeps
        // the first row at each year regardless of probe output order.
        curve.sort_by_key(|(y, is_eval, id, _)| (*y, *is_eval, *id));
        curve.dedup_by_key(|(y, _, _, _)| *y);
        let curve: Vec<(i32, f64)> = curve.into_iter().map(|(y, _, _, p)| (y, p)).collect();
        let first = rs[0];
        let (Some(a), b) = (first.example.interval_start, first.example.interval_end) else {
            continue;
        };
        if let Some(b_end) = b
            && a > b_end
        {
            continue;
        }
        if let Some((s, e)) = boundary_error(&curve, a, b, world_end) {
            se.push(s as f64);
            ee.push(e as f64);
        }
        ious.push(interval_iou(&curve, a, b, world_end));
        smooth.push(smoothness(&curve));
    }
    let mean = |v: &[f64]| {
        if v.is_empty() {
            None
        } else {
            Some(v.iter().sum::<f64>() / v.len() as f64)
        }
    };

    MetricsReport {
        model,
        overall: group_metrics(&eval),
        by_relation: group_by(&eval, |r| r.example.relation.as_str().to_string()),
        by_band: group_by(&eval, |r| r.example.sample_band.clone()),
        by_split: group_by(&eval, |r| r.example.split.clone()),
        mean_boundary_error_start: mean(&se),
        mean_boundary_error_end: mean(&ee),
        mean_interval_iou: mean(&ious).unwrap_or(0.0),
        mean_smoothness: mean(&smooth).unwrap_or(0.0),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)] // exact values are deterministic on these inputs
    use super::*;

    #[test]
    fn accuracy_counts_thresholded_correctness() {
        let pts = vec![(0.9, 1u8), (0.8, 1), (0.3, 0), (0.7, 0)];
        assert_eq!(accuracy(&pts), 0.75);
        assert_eq!(accuracy(&[]), 0.0);
    }

    #[test]
    fn auroc_known_values() {
        let perfect = vec![(0.9, 1u8), (0.8, 1), (0.2, 0), (0.1, 0)];
        assert_eq!(auroc(&perfect), Some(1.0));
        let reversed = vec![(0.1, 1u8), (0.2, 1), (0.8, 0), (0.9, 0)];
        assert_eq!(auroc(&reversed), Some(0.0));
        let ties = vec![(0.5, 1u8), (0.5, 0)];
        assert_eq!(auroc(&ties), Some(0.5));
        let single_class = vec![(0.9, 1u8), (0.1, 1)];
        assert_eq!(auroc(&single_class), None);
    }

    #[test]
    fn brier_and_logloss_sanity() {
        let good = vec![(0.9, 1u8), (0.1, 0)];
        let bad = vec![(0.1, 1u8), (0.9, 0)];
        assert!(brier(&good) < brier(&bad));
        assert!(log_loss(&good) < log_loss(&bad));
        assert!((brier(&good) - (0.01 + 0.01) / 2.0).abs() < 1e-9);
    }

    #[test]
    fn boundary_error_finds_crossings() {
        // p crosses 0.5 between years -60 and -40 (step 10), true start -50
        let curve: Vec<(i32, f64)> = (-100..=100)
            .step_by(10)
            .map(|y| (y, if y < -50 { 0.1 } else { 0.9 }))
            .collect();
        let (es, ee) = boundary_error(&curve, -50, Some(50), 2026).unwrap();
        assert!(es <= 10, "start err {es}");
        assert!(ee <= 60, "end err {ee}"); // active runs to 100, true end 50
        assert!(boundary_error(&[], -50, Some(50), 2026).is_none());
    }

    #[test]
    fn iou_perfect_and_zero() {
        let curve: Vec<(i32, f64)> = (-100..=100)
            .step_by(10)
            .map(|y| (y, if (-50..=50).contains(&y) { 0.9 } else { 0.1 }))
            .collect();
        let iou = interval_iou(&curve, -50, Some(50), 2026);
        assert_eq!(iou, 1.0);
        let dead: Vec<(i32, f64)> = (-100..=100).step_by(10).map(|y| (y, 0.1)).collect();
        assert_eq!(interval_iou(&dead, -50, Some(50), 2026), 0.0);
    }

    #[test]
    fn smoothness_measures_jitter() {
        let flat = vec![(0, 0.5), (1, 0.5), (2, 0.5)];
        assert_eq!(smoothness(&flat), 0.0);
        let jag = vec![(0, 0.0), (1, 1.0), (2, 0.0)];
        assert_eq!(smoothness(&jag), 1.0);
    }

    #[test]
    fn report_excludes_unknown_and_groups() {
        use crate::querygen::{GenConfig, gen_eval};
        use crate::types::tests_helpers::caesar_interval;
        use crate::types::*;
        let cfg = GenConfig::default();
        let mut responses: Vec<Response> = gen_eval(&caesar_interval(), "Julius Caesar", &[], &cfg)
            .into_iter()
            .map(|example| Response {
                p_yes: if example.label_int == 1 { 0.9 } else { 0.1 },
                logit_diff: 0.0,
                top_logprobs: vec![],
                model: "m".into(),
                latency_ms: 1,
                example,
            })
            .collect();
        // force one row unknown to check exclusion
        responses[0].example.gold_label = GoldLabel::UnknownOrAmbiguous;
        let report = compute_report(&responses, 2026);
        assert_eq!(report.model, "m");
        assert_eq!(report.overall.n, 7);
        assert_eq!(report.overall.n_excluded_unknown, 1);
        assert!(report.by_relation.contains_key("alive"));
        assert!(report.by_band.contains_key("interior"));
        assert!(report.mean_interval_iou > 0.0);
    }

    #[test]
    fn auroc_mid_value() {
        // 3 of 4 pos/neg pairs concordant (0.4 loses only to 0.6).
        let pts = vec![(0.9, 1u8), (0.4, 1), (0.6, 0), (0.1, 0)];
        assert_eq!(auroc(&pts), Some(0.75));
    }

    #[test]
    fn report_is_deterministic_under_input_order() {
        use crate::querygen::{GenConfig, gen_eval, gen_sweep};
        use crate::types::tests_helpers::caesar_interval;
        use crate::types::*;
        // sweep_step 1 makes the sweep cover every year, so every eval year
        // collides with a sweep row regardless of the RNG-seeded eval years.
        let cfg = GenConfig {
            sweep_step: 1,
            ..Default::default()
        };
        let iv = caesar_interval();
        // Sweep and eval rows deliberately use different p scales, so at a
        // year-tie the surviving row changes the curve unless ties break
        // deterministically (sweep wins by construction).
        let in_interval = |y: i32| (-99..=-43).contains(&y);
        let mk = |ex: Example, p: f64| Response {
            p_yes: p,
            logit_diff: 0.0,
            top_logprobs: vec![],
            model: "m".into(),
            latency_ms: 1,
            example: ex,
        };
        let sweep: Vec<Response> = gen_sweep(&iv, "Julius Caesar", &cfg)
            .unwrap()
            .into_iter()
            .map(|ex| {
                let p = if in_interval(ex.year_astronomical) {
                    0.9
                } else {
                    0.1
                };
                mk(ex, p)
            })
            .collect();
        let eval: Vec<Response> = gen_eval(&iv, "Julius Caesar", &[], &cfg)
            .into_iter()
            .map(|ex| {
                let p = if in_interval(ex.year_astronomical) {
                    0.8
                } else {
                    0.2
                };
                mk(ex, p)
            })
            .collect();
        let eval_then_sweep = [eval.clone(), sweep.clone()].concat();
        let sweep_then_eval = [sweep, eval].concat();
        let r1 = compute_report(&eval_then_sweep, 2026);
        let r2 = compute_report(&sweep_then_eval, 2026);
        assert_eq!(r1.mean_interval_iou, r2.mean_interval_iou);
        assert_eq!(r1.mean_smoothness, r2.mean_smoothness);
        assert_eq!(r1.mean_boundary_error_start, r2.mean_boundary_error_start);
        assert_eq!(r1.mean_boundary_error_end, r2.mean_boundary_error_end);
    }

    #[test]
    fn report_is_deterministic_under_eval_eval_year_ties() {
        use crate::querygen::{GenConfig, gen_eval};
        use crate::types::tests_helpers::caesar_interval;
        use crate::types::*;
        // Point interval: three interior eval rows share the year 1066 under
        // different templates; which row survives the per-year dedup decides
        // IoU (1.0 vs 0.0), so input order must not matter.
        let iv = Interval {
            start_year: Some(1066),
            end_year: Some(1066),
            ..caesar_interval()
        };
        let cfg = GenConfig::default();
        let mut eval: Vec<Response> = gen_eval(&iv, "Battle of Hastings", &[], &cfg)
            .into_iter()
            .map(|example| Response {
                p_yes: 0.1,
                logit_diff: 0.0,
                top_logprobs: vec![],
                model: "m".into(),
                latency_ms: 1,
                example,
            })
            .collect();
        // The documented tie-break: at a tied year the lexicographically
        // smallest example_id survives dedup. Give exactly that row a
        // crossing p so IoU is 1.0 iff the tie-break holds.
        let mut tied: Vec<&str> = eval
            .iter()
            .filter(|r| r.example.year_astronomical == 1066)
            .map(|r| r.example.example_id.as_str())
            .collect();
        tied.sort();
        let winner = tied[0].to_string();
        for r in &mut eval {
            if r.example.example_id == winner {
                r.p_yes = 0.9;
            }
        }
        let mut reversed = eval.clone();
        reversed.reverse();
        let r1 = compute_report(&eval, 2026);
        let r2 = compute_report(&reversed, 2026);
        assert_eq!(r1.mean_interval_iou, r2.mean_interval_iou);
        assert_eq!(r1.mean_smoothness, r2.mean_smoothness);
        assert_eq!(r1.mean_boundary_error_start, r2.mean_boundary_error_start);
        assert_eq!(r1.mean_boundary_error_end, r2.mean_boundary_error_end);
        // Deterministic survivor: the smallest example_id at the tied year.
        assert_eq!(r1.mean_interval_iou, 1.0);
    }

    #[test]
    fn multi_interval_same_entity_separate_curves() {
        use crate::querygen::{GenConfig, gen_sweep};
        use crate::types::tests_helpers::caesar_interval;
        use crate::types::*;
        let cfg = GenConfig::default();
        // Same entity, second relation with different bounds: one interval
        // per (relation, entity), so the curves must not merge.
        let alive = caesar_interval();
        let exists = Interval {
            interval_id: "exists:wd:Q1048".into(),
            relation: Relation::Exists,
            start_year: Some(-508),
            end_year: Some(-26),
            ..caesar_interval()
        };
        let sweep_responses = |iv: &Interval| -> Vec<Response> {
            let (a, b) = (iv.start_year.unwrap(), iv.end_year.unwrap());
            gen_sweep(iv, "Julius Caesar", &cfg)
                .unwrap()
                .into_iter()
                .map(|ex| {
                    let y = ex.year_astronomical;
                    Response {
                        p_yes: if (a..=b).contains(&y) { 0.9 } else { 0.1 },
                        logit_diff: 0.0,
                        top_logprobs: vec![],
                        model: "m".into(),
                        latency_ms: 1,
                        example: ex,
                    }
                })
                .collect()
        };
        let responses = [sweep_responses(&alive), sweep_responses(&exists)].concat();
        let report = compute_report(&responses, 2026);
        assert!(report.mean_boundary_error_start.is_some());
        // Expected mean = average of the two per-interval IoUs, each computed
        // on its own curve and bounds; equality proves bounds weren't crossed.
        let curve_of = |iv: &Interval| -> Vec<(i32, f64)> {
            let (a, b) = (iv.start_year.unwrap(), iv.end_year.unwrap());
            (cfg.sweep_from..=cfg.sweep_to)
                .step_by(cfg.sweep_step as usize)
                .map(|y| (y, if (a..=b).contains(&y) { 0.9 } else { 0.1 }))
                .collect()
        };
        let expected = (interval_iou(&curve_of(&alive), -99, Some(-43), 2026)
            + interval_iou(&curve_of(&exists), -508, Some(-26), 2026))
            / 2.0;
        assert_eq!(report.mean_interval_iou, expected);
    }
}
