//! Score-map rendering: entity×year PNG heatmap, per-entity SVG curves,
//! boundary-error histogram, calibration plot.

use crate::metrics::boundary_error;
use crate::store::ensure_parent;
use crate::types::{GoldLabel, Relation, Response, Split};
use anyhow::Context;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// One (relation, entity) interval's year curve plus the bounds it was
/// generated from, ready to plot.
#[derive(Clone, Debug)]
pub struct EntityCurve {
    pub name: String,
    pub relation: Relation,
    pub subject_id: String,
    pub interval_start: Option<i32>,
    pub interval_end: Option<i32>,
    pub curve: Vec<(i32, f64)>,
}

/// Group responses into per-interval curves keyed by `relation:subject_id`
/// (an entity can carry several relations, each its own interval), sorted by
/// year, deduped, and ordered by true start year: the "civilization bands"
/// ordering.
pub fn curves_from_responses(responses: &[Response]) -> Vec<EntityCurve> {
    let mut by_entity: BTreeMap<String, Vec<&Response>> = BTreeMap::new();
    for r in responses {
        by_entity
            .entry(r.example.relation.group_key(&r.example.subject_id))
            .or_default()
            .push(r);
    }
    let mut out: Vec<EntityCurve> = by_entity
        .into_values()
        .map(|rs| {
            let mut curve: Vec<(i32, f64)> = rs
                .iter()
                .map(|r| (r.example.year_astronomical, r.p_yes))
                .collect();
            curve.sort_by_key(|(y, _)| *y);
            curve.dedup_by_key(|(y, _)| *y);
            let first = rs[0];
            EntityCurve {
                name: first.example.subject_name.clone(),
                relation: first.example.relation,
                subject_id: first.example.subject_id.clone(),
                interval_start: first.example.interval_start,
                interval_end: first.example.interval_end,
                curve,
            }
        })
        .collect();
    // sort by true start year: the "civilization bands" ordering
    out.sort_by_key(|c| c.interval_start.unwrap_or(i32::MAX));
    out
}

/// p_yes at `year`, linearly interpolated, clamped at curve edges.
pub fn sample(curve: &[(i32, f64)], year: i32) -> f64 {
    match curve.binary_search_by_key(&year, |(y, _)| *y) {
        Ok(i) => curve[i].1,
        Err(0) => curve.first().map(|(_, p)| *p).unwrap_or(0.5),
        Err(i) if i >= curve.len() => curve.last().map(|(_, p)| *p).unwrap_or(0.5),
        Err(i) => {
            let (y0, p0) = curve[i - 1];
            let (y1, p1) = curve[i];
            let t = (year - y0) as f64 / (y1 - y0) as f64;
            p0 + t * (p1 - p0)
        }
    }
}

fn color(p: f64) -> [u8; 3] {
    // dark navy (no) -> gold (yes)
    let lerp = |a: u8, b: u8| (a as f64 + (b as f64 - a as f64) * p.clamp(0.0, 1.0)) as u8;
    [lerp(20, 240), lerp(20, 200), lerp(60, 40)]
}

/// Render the entity×year heatmap: one row per curve, one column per sweep
/// year, pixels interpolated along each curve.
pub fn heatmap_png(curves: &[EntityCurve], path: &Path) -> anyhow::Result<()> {
    let mut years: Vec<i32> = curves
        .iter()
        .flat_map(|c| c.curve.iter().map(|(y, _)| *y))
        .collect();
    years.sort_unstable();
    years.dedup();
    anyhow::ensure!(!years.is_empty() && !curves.is_empty(), "nothing to render");
    let img = image::ImageBuffer::from_fn(years.len() as u32, curves.len() as u32, |x, y| {
        let p = sample(&curves[y as usize].curve, years[x as usize]);
        image::Rgb(color(p))
    });
    ensure_parent(path)?;
    img.save(path)
        .with_context(|| format!("saving {}", path.display()))?;
    Ok(())
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Writes an SVG artifact: creates the parent directory when missing and
/// writes `svg` to `path`.
fn write_svg(path: &Path, svg: &str) -> anyhow::Result<()> {
    ensure_parent(path)?;
    std::fs::write(path, svg).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Render one curve as an SVG: title, true-interval band, year axis with
/// BCE/CE ticks every 500 years, and the P(active) polyline.
pub fn curve_svg(ec: &EntityCurve, world: (i32, i32), path: &Path) -> anyhow::Result<()> {
    let (w, h) = (800.0f64, 240.0f64);
    let (pl, pr, pt, pb) = (50.0, 20.0, 30.0, 34.0);
    // A single-year world would divide by zero below (every x = NaN); widen
    // by a year so the degenerate range still renders sane axes.
    let (y0, y1) = match world {
        (y0, y1) if y0 == y1 => (y0 - 1, y1 + 1),
        world => world,
    };
    let x = |year: i32| pl + (year - y0) as f64 / (y1 - y0) as f64 * (w - pl - pr);
    let yv = |p: f64| pt + (1.0 - p) * (h - pt - pb);
    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">"#
    );
    svg.push_str(r#"<rect width="100%" height="100%" fill="white"/>"#);
    svg.push_str(&format!(
        r#"<text x="{pl}" y="18" font-size="14" font-family="sans-serif">{}</text>"#,
        xml_escape(&ec.name)
    ));
    if let Some(a) = ec.interval_start {
        let b = ec.interval_end.unwrap_or(y1);
        svg.push_str(&format!(
            r##"<rect x="{:.1}" y="{pt}" width="{:.1}" height="{:.1}" fill="#d9f2d9"/>"##,
            x(a),
            x(b) - x(a),
            h - pt - pb
        ));
    }
    svg.push_str(&format!(
        r#"<line x1="{pl}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="black"/>"#,
        h - pb,
        w - pr,
        h - pb
    ));
    let mut t = (y0 / 500) * 500;
    while t <= y1 {
        svg.push_str(&format!(
            r#"<text x="{:.1}" y="{:.1}" font-size="10" text-anchor="middle" font-family="sans-serif">{}</text>"#,
            x(t),
            h - pb + 16.0,
            crate::year::display_year(t)
        ));
        t += 500;
    }
    let pts: Vec<String> = ec
        .curve
        .iter()
        .map(|(yy, p)| format!("{:.1},{:.1}", x(*yy), yv(*p)))
        .collect();
    svg.push_str(&format!(
        r##"<polyline fill="none" stroke="#1a4a8a" stroke-width="1.5" points="{}"/>"##,
        pts.join(" ")
    ));
    svg.push_str("</svg>");
    write_svg(path, &svg)
}

/// Render labeled count buckets as a bar-chart SVG.
pub fn histogram_svg(title: &str, buckets: &[(String, usize)], path: &Path) -> anyhow::Result<()> {
    let (w, h) = (640.0f64, 300.0f64);
    let (pl, pt, pb) = (60.0, 34.0, 40.0);
    let max = buckets.iter().map(|(_, n)| *n).max().unwrap_or(1).max(1) as f64;
    let bw = (w - pl - 20.0) / buckets.len().max(1) as f64;
    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">"#
    );
    svg.push_str(r#"<rect width="100%" height="100%" fill="white"/>"#);
    svg.push_str(&format!(
        r#"<text x="{pl}" y="20" font-size="14" font-family="sans-serif">{}</text>"#,
        xml_escape(title)
    ));
    for (i, (label, n)) in buckets.iter().enumerate() {
        let bh = (*n as f64 / max) * (h - pt - pb);
        svg.push_str(&format!(
            r##"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="#1a4a8a"/>"##,
            pl + i as f64 * bw + 2.0,
            h - pb - bh,
            bw - 4.0,
            bh
        ));
        svg.push_str(&format!(
            r#"<text x="{:.1}" y="{:.1}" font-size="10" text-anchor="middle" font-family="sans-serif">{}</text>"#,
            pl + i as f64 * bw + bw / 2.0,
            h - pb + 16.0,
            xml_escape(label)
        ));
    }
    svg.push_str("</svg>");
    write_svg(path, &svg)
}

/// Render a calibration plot: 10 predicted-probability bins as (mean
/// predicted, empirical rate) circles against the diagonal.
pub fn calibration_svg(points: &[(f64, u8)], path: &Path) -> anyhow::Result<()> {
    let (w, h) = (420.0f64, 420.0f64);
    let (pl, pr, pt) = (50.0, 20.0, 34.0);
    let s = w - pl - pr;
    let mut bins = vec![(0.0f64, 0.0f64, 0usize); 10];
    for (p, y) in points {
        let b = ((p * 10.0) as usize).min(9);
        bins[b].0 += p;
        bins[b].1 += *y as f64;
        bins[b].2 += 1;
    }
    let xy = |v: f64| pl + v * s;
    let yv = |v: f64| pt + (1.0 - v) * s;
    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">"#
    );
    svg.push_str(r#"<rect width="100%" height="100%" fill="white"/>"#);
    svg.push_str(&format!(
        r#"<text x="{pl}" y="20" font-size="14" font-family="sans-serif">Calibration (predicted vs empirical)</text>"#
    ));
    svg.push_str(&format!(
        r##"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="#999" stroke-dasharray="4"/>"##,
        xy(0.0),
        yv(0.0),
        xy(1.0),
        yv(1.0)
    ));
    svg.push_str(&format!(
        r#"<rect x="{pl}" y="{pt}" width="{s}" height="{s}" fill="none" stroke="black"/>"#
    ));
    for (sp, sy, n) in bins {
        if n == 0 {
            continue;
        }
        let (mp, my) = (sp / n as f64, sy / n as f64);
        svg.push_str(&format!(
            r##"<circle cx="{:.1}" cy="{:.1}" r="4" fill="#1a4a8a"><title>n={n}</title></circle>"##,
            xy(mp),
            yv(my)
        ));
    }
    svg.push_str("</svg>");
    write_svg(path, &svg)
}

fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect()
}

/// Render all score maps for a run. Sweep responses feed the heatmap/curves;
/// eval responses feed calibration. `world_end` substitutes for a missing end
/// bound in boundary-error histograms and should match the `GenConfig` used
/// at generation time. Returns written paths.
pub fn render_run(
    responses: &[Response],
    out_dir: &Path,
    max_curves: usize,
    world_end: i32,
) -> anyhow::Result<Vec<PathBuf>> {
    std::fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let mut written = vec![];
    let sweep: Vec<Response> = responses
        .iter()
        .filter(|r| r.example.split == Split::Sweep)
        .cloned()
        .collect();
    let curve_source: &[Response] = if sweep.is_empty() { responses } else { &sweep };
    let curves = curves_from_responses(curve_source);
    if !curves.is_empty() {
        let p = out_dir.join("heatmap.png");
        heatmap_png(&curves, &p)?;
        written.push(p);
        let y0 = curves
            .iter()
            .flat_map(|c| c.curve.iter().map(|(y, _)| *y))
            .min()
            .unwrap();
        let y1 = curves
            .iter()
            .flat_map(|c| c.curve.iter().map(|(y, _)| *y))
            .max()
            .unwrap();
        for ec in curves.iter().take(max_curves) {
            // Keyed by relation:subject_id, so name alone can collide (same
            // sanitized name, or one entity with several relations).
            let p = out_dir.join(format!(
                "curve_{}_{}_{}.svg",
                sanitize(&ec.name),
                ec.relation.as_str(),
                sanitize(&ec.subject_id)
            ));
            curve_svg(ec, (y0, y1), &p)?;
            written.push(p);
        }
    }
    let mut buckets: BTreeMap<&str, usize> = BTreeMap::new();
    for ec in &curves {
        let Some(a) = ec.interval_start else { continue };
        if let Some((se, _)) = boundary_error(&ec.curve, a, ec.interval_end, world_end) {
            let label = match se {
                0..=10 => "0-10",
                11..=25 => "11-25",
                26..=50 => "26-50",
                51..=100 => "51-100",
                _ => ">100",
            };
            *buckets.entry(label).or_default() += 1;
        }
    }
    if !buckets.is_empty() {
        let ordered: Vec<(String, usize)> = ["0-10", "11-25", "26-50", "51-100", ">100"]
            .iter()
            .map(|k| (k.to_string(), buckets.get(k).copied().unwrap_or(0)))
            .collect();
        let p = out_dir.join("boundary_error.svg");
        histogram_svg("Boundary start error (years)", &ordered, &p)?;
        written.push(p);
    }
    let pts: Vec<(f64, u8)> = responses
        .iter()
        .filter(|r| {
            r.example.split != Split::Sweep && r.example.gold_label != GoldLabel::UnknownOrAmbiguous
        })
        .map(|r| (r.p_yes, r.example.label_int))
        .collect();
    if !pts.is_empty() {
        let p = out_dir.join("calibration.svg");
        calibration_svg(&pts, &p)?;
        written.push(p);
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)] // exact values are deterministic on these inputs
    use super::*;
    use crate::querygen::{GenConfig, gen_sweep};
    use crate::types::tests_helpers::{caesar_interval, fake_response};
    use crate::types::*;

    fn fake_responses() -> Vec<Response> {
        let iv = caesar_interval();
        gen_sweep(&iv, "Julius Caesar", &GenConfig::default())
            .unwrap()
            .into_iter()
            .map(|example| {
                let p_yes = if example.label_int == 1 { 0.9 } else { 0.1 };
                fake_response(example, p_yes)
            })
            .collect()
    }

    #[test]
    fn curves_group_and_sort() {
        let curves = curves_from_responses(&fake_responses());
        assert_eq!(curves.len(), 1);
        assert_eq!(curves[0].name, "Julius Caesar");
        assert!(curves[0].curve.windows(2).all(|w| w[0].0 < w[1].0));
    }

    #[test]
    fn sample_interpolates_between_years() {
        let curve = vec![(-100, 0.0), (0, 1.0)];
        assert_eq!(sample(&curve, -100), 0.0);
        assert_eq!(sample(&curve, -50), 0.5);
        assert_eq!(sample(&curve, 500), 1.0); // clamps at edge
    }

    #[test]
    fn heatmap_writes_png_with_entity_rows() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("heatmap.png");
        let curves = curves_from_responses(&fake_responses());
        heatmap_png(&curves, &path).unwrap();
        let (w, h) = image::image_dimensions(&path).unwrap();
        assert_eq!(h, 1);
        assert_eq!(w, 503); // sweep years -3000..=2026 step 10
    }

    #[test]
    fn curve_svg_contains_polyline_and_name() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("curve.svg");
        let curves = curves_from_responses(&fake_responses());
        curve_svg(&curves[0], (-3000, 2026), &path).unwrap();
        let svg = std::fs::read_to_string(&path).unwrap();
        assert!(svg.contains("<polyline"));
        assert!(svg.contains("Julius Caesar"));
    }

    #[test]
    fn curve_svg_single_year_range_has_no_nan() {
        // All responses on one year: y0 == y1 divided by zero and every x
        // coordinate became NaN. The range is widened by a year instead.
        let tmp = tempfile::tempdir().unwrap();
        let iv = Interval {
            start_year: Some(1066),
            end_year: Some(1066),
            ..caesar_interval()
        };
        let cfg = GenConfig {
            sweep_from: 1066,
            sweep_to: 1066,
            sweep_step: 1,
            ..GenConfig::default()
        };
        let responses: Vec<Response> = gen_sweep(&iv, "Battle of Hastings", &cfg)
            .unwrap()
            .into_iter()
            .map(|example| fake_response(example, 0.9))
            .collect();
        render_run(&responses, tmp.path(), 5, GenConfig::default().world_end).unwrap();
        let svg = std::fs::read_to_string(
            tmp.path()
                .join("curve_Battle_of_Hastings_alive_wd_Q1048.svg"),
        )
        .unwrap();
        assert!(!svg.contains("NaN"), "svg has NaN coordinates");
    }

    #[test]
    fn histogram_and_calibration_render() {
        let tmp = tempfile::tempdir().unwrap();
        let hist = tmp.path().join("hist.svg");
        histogram_svg("t", &[("0-10".into(), 3), ("11-25".into(), 1)], &hist).unwrap();
        assert!(std::fs::read_to_string(&hist).unwrap().contains("<rect"));
        let cal = tmp.path().join("cal.svg");
        calibration_svg(&[(0.9, 1), (0.1, 0), (0.8, 1)], &cal).unwrap();
        assert!(std::fs::read_to_string(&cal).unwrap().contains("line"));
    }

    #[test]
    fn render_run_emits_artifacts() {
        let tmp = tempfile::tempdir().unwrap();
        let written = render_run(
            &fake_responses(),
            tmp.path(),
            5,
            GenConfig::default().world_end,
        )
        .unwrap();
        let names: Vec<String> = written
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into())
            .collect();
        assert!(names.contains(&"heatmap.png".to_string()));
        assert!(
            names.contains(&"curve_Julius_Caesar_alive_wd_Q1048.svg".to_string()),
            "curve filename carries relation + subject id: {names:?}"
        );
        assert!(names.contains(&"boundary_error.svg".to_string()));
    }
}
