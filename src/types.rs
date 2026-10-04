//! Canonical dataset schemas. Field names match the spec JSON verbatim.

use serde::{Deserialize, Serialize};

macro_rules! string_enum {
    ($name:ident { $($variant:ident => $s:literal),+ $(,)? }) => {
        // Plan-mandated API name (used by parquet_io and the CLI); would
        // otherwise trip clippy::should_implement_trait vs std::str::FromStr.
        #[allow(clippy::should_implement_trait)]
        impl $name {
            pub fn as_str(self) -> &'static str {
                match self { $($name::$variant => $s),+ }
            }
            pub fn from_str(s: &str) -> Option<Self> {
                match s { $($s => Some($name::$variant),)+ _ => None }
            }
        }
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EntityType {
    Person,
    Event,
    Polity,
    Organization,
    Work,
    Technology,
}
string_enum!(EntityType { Person => "person", Event => "event", Polity => "polity",
    Organization => "organization", Work => "work", Technology => "technology" });

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Relation {
    Alive,
    Ongoing,
    Exists,
    Active,
    Available,
}
string_enum!(Relation { Alive => "alive", Ongoing => "ongoing", Exists => "exists",
    Active => "active", Available => "available" });

impl Relation {
    /// Canonical grouping key for one interval: `relation:subject_id`. An
    /// entity can carry several relations, each its own interval, so name
    /// alone is not a key. Shared by metrics (per-interval curves) and
    /// render (entity curves) so both partition responses identically.
    pub fn group_key(self, subject_id: &str) -> String {
        format!("{}:{}", self.as_str(), subject_id)
    }
}

/// Which dataset split an example belongs to. `Sweep` rows are the dense
/// per-year probes feeding curves/heatmaps; the rest are eval rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Train,
    Dev,
    Test,
    Sweep,
}
string_enum!(Split { Train => "train", Dev => "dev", Test => "test", Sweep => "sweep" });

/// Stratification band of an eval example's year: where the year sits
/// relative to the interval bounds (or the whole world axis for fallbacks).
/// Dense sweep rows use `Sweep` for both band and split.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Band {
    Interior,
    NearBefore,
    NearAfter,
    FarBefore,
    FarAfter,
    EraConfusable,
    FarFallback,
    Sweep,
}
string_enum!(Band { Interior => "interior", NearBefore => "near_before", NearAfter => "near_after",
    FarBefore => "far_before", FarAfter => "far_after", EraConfusable => "era_confusable",
    FarFallback => "far_fallback", Sweep => "sweep" });

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum YearPrecision {
    Day,
    Month,
    Year,
    Decade,
    Century,
    Millennium,
    Approximate,
    Range,
    Unknown,
}
string_enum!(YearPrecision { Day => "day", Month => "month", Year => "year", Decade => "decade",
    Century => "century", Millennium => "millennium", Approximate => "approximate",
    Range => "range", Unknown => "unknown" });

impl YearPrecision {
    /// Half-window (years) around a bound too coarse for honest yearly labels.
    pub fn ambiguity_window(self) -> i32 {
        match self {
            YearPrecision::Day | YearPrecision::Month | YearPrecision::Year => 0,
            YearPrecision::Decade => 5,
            YearPrecision::Approximate | YearPrecision::Range => 25,
            YearPrecision::Century => 50,
            YearPrecision::Millennium => 500,
            YearPrecision::Unknown => i32::MAX,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Confidence {
    High,
    Medium,
    Low,
}
string_enum!(Confidence { High => "high", Medium => "medium", Low => "low" });

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoldLabel {
    #[serde(rename = "yes")]
    Yes,
    #[serde(rename = "no")]
    No,
    #[serde(rename = "unknown_or_ambiguous")]
    UnknownOrAmbiguous,
}

impl GoldLabel {
    pub fn as_str(self) -> &'static str {
        match self {
            GoldLabel::Yes => "yes",
            GoldLabel::No => "no",
            GoldLabel::UnknownOrAmbiguous => "unknown_or_ambiguous",
        }
    }
    /// 1 iff yes. Never a sync authority: `label_int` is a denormalized copy
    /// of this value, and unknown-labeled rows must be excluded via
    /// `gold_label` by every consumer (as `compute_report` does), not trusted
    /// via `label_int`.
    pub fn label_int(self) -> u8 {
        match self {
            GoldLabel::Yes => 1,
            _ => 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    pub entity_id: String,
    pub entity_type: EntityType,
    pub canonical_name: String,
    pub aliases: Vec<String>,
    pub description: String,
    pub language: String,
    pub source_id: String,
    pub source_url: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Interval {
    pub interval_id: String,
    pub entity_id: String,
    pub relation: Relation,
    pub start_year: Option<i32>,
    pub end_year: Option<i32>,
    pub start_precision: YearPrecision,
    pub end_precision: YearPrecision,
    /// Preserved for source fidelity / future use; v0 labeling always
    /// treats bounds as inclusive (see `label_at`).
    pub start_inclusive: bool,
    /// Preserved for source fidelity / future use; v0 labeling always
    /// treats bounds as inclusive (see `label_at`).
    pub end_inclusive: bool,
    pub confidence: Confidence,
    pub date_basis: String,
    pub source_id: String,
    pub notes: String,
}

impl Interval {
    /// Three-valued label honoring bounds, open-endedness, and precision.
    /// Years within `ambiguity_window()` of a bound are `UnknownOrAmbiguous`
    /// on EITHER side of that bound, including years strictly inside the
    /// interval. Bounds are always treated as inclusive in v0.
    pub fn label_at(&self, year: i32) -> GoldLabel {
        // Distances are computed in i64: (year - bound) overflows i32 for
        // extreme years (e.g. a = i32::MIN, year positive).
        if let Some(a) = self.start_year
            && (i64::from(year) - i64::from(a)).abs()
                < i64::from(self.start_precision.ambiguity_window())
        {
            return GoldLabel::UnknownOrAmbiguous;
        }
        if let Some(b) = self.end_year
            && (i64::from(year) - i64::from(b)).abs()
                < i64::from(self.end_precision.ambiguity_window())
        {
            return GoldLabel::UnknownOrAmbiguous;
        }
        match (self.start_year, self.end_year) {
            (Some(a), Some(b)) if (a..=b).contains(&year) => GoldLabel::Yes,
            (Some(_), Some(_)) => GoldLabel::No,
            (Some(a), None) => {
                if year >= a {
                    GoldLabel::Yes
                } else {
                    GoldLabel::No
                }
            }
            (None, _) => GoldLabel::UnknownOrAmbiguous,
        }
    }

    /// Midpoint year (`world_end` substitutes for a missing end bound);
    /// `None` when the start bound is missing. Computed in i64 so extreme
    /// years cannot overflow i32 in debug builds.
    pub fn midpoint(&self, world_end: i32) -> Option<i32> {
        let mid = |a: i32, b: i32| (i64::from(a) + (i64::from(b) - i64::from(a)) / 2) as i32;
        match (self.start_year, self.end_year) {
            (Some(a), Some(b)) => Some(mid(a, b)),
            (Some(a), None) => Some(mid(a, world_end)),
            (None, _) => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub entity_id: String,
    pub field: String,
    pub value: serde_json::Value,
    pub source_name: String,
    pub source_id: String,
    pub retrieved_at: String,
    pub source_statement_id: Option<String>,
    pub human_review_status: String,
}

/// A prompt-ready evaluation query (generated layer).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Example {
    pub example_id: String,
    pub task: String,
    pub relation: Relation,
    pub subject_id: String,
    pub subject_name: String,
    pub year_astronomical: i32,
    pub year_display: String,
    pub gold_label: GoldLabel,
    /// Deliberate denormalization of `gold_label.label_int()` for flat
    /// parquet/ML consumers. There are no constructors enforcing the sync:
    /// the invariant is by convention at generation sites (querygen sets both
    /// from one `label_at` call), and every consumer must filter on
    /// `gold_label` itself, as `compute_report` does.
    pub label_int: u8,
    pub interval_start: Option<i32>,
    pub interval_end: Option<i32>,
    pub sample_band: Band,
    pub template_id: String,
    pub prompt: String,
    pub source_ids: Vec<String>,
    pub label_confidence: Confidence,
    pub split: Split,
}

/// A probe response: the full example denormalized plus model output,
/// so score/render need no join back to generated/.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Response {
    #[serde(flatten)]
    pub example: Example,
    pub model: String,
    pub logit_diff: f64,
    pub p_yes: f64,
    pub top_logprobs: Vec<(String, f64)>,
    pub latency_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::tests_helpers::*;
    use super::*;

    #[test]
    fn label_at_interior_and_outside() {
        let iv = caesar_interval();
        assert_eq!(iv.label_at(-59), GoldLabel::Yes);
        assert_eq!(iv.label_at(-99), GoldLabel::Yes);
        assert_eq!(iv.label_at(-43), GoldLabel::Yes);
        assert_eq!(iv.label_at(-100), GoldLabel::No);
        assert_eq!(iv.label_at(0), GoldLabel::No);
    }

    #[test]
    fn label_at_open_ended() {
        let iv = Interval {
            end_year: None,
            ..caesar_interval()
        };
        assert_eq!(iv.label_at(2026), GoldLabel::Yes);
        assert_eq!(iv.label_at(-100), GoldLabel::No);
    }

    #[test]
    fn label_at_coarse_precision_is_ambiguous_near_bound() {
        let iv = Interval {
            start_precision: YearPrecision::Century,
            ..caesar_interval()
        };
        assert_eq!(iv.label_at(-104), GoldLabel::UnknownOrAmbiguous); // 5y outside bound, window 50
        assert_eq!(iv.label_at(-59), GoldLabel::UnknownOrAmbiguous); // 40y inside bound, still in fuzzy window
        assert_eq!(iv.label_at(-45), GoldLabel::Yes); // interior, beyond the window
    }

    #[test]
    fn label_at_null_start_is_unknown() {
        let iv = Interval {
            start_year: None,
            ..caesar_interval()
        };
        assert_eq!(iv.label_at(-59), GoldLabel::UnknownOrAmbiguous);
    }

    #[test]
    fn label_at_extreme_years_does_not_overflow() {
        // (year - a) in i32 overflows for extremes; the distance must be
        // computed in i64. Window 0 precision, so no ambiguity is triggered
        // and plain interval membership decides.
        let iv = Interval {
            start_year: Some(i32::MIN),
            end_year: None,
            ..caesar_interval()
        };
        assert_eq!(iv.label_at(i32::MAX), GoldLabel::Yes);
        let bounded = Interval {
            end_year: Some(i32::MAX),
            ..iv.clone()
        };
        assert_eq!(bounded.label_at(i32::MAX), GoldLabel::Yes);
        let ended = Interval {
            end_year: Some(0),
            ..iv
        };
        assert_eq!(ended.label_at(i32::MAX), GoldLabel::No);
    }

    #[test]
    fn gold_label_serde_strings() {
        assert_eq!(serde_json::to_string(&GoldLabel::Yes).unwrap(), "\"yes\"");
        assert_eq!(serde_json::to_string(&GoldLabel::No).unwrap(), "\"no\"");
        assert_eq!(
            serde_json::to_string(&GoldLabel::UnknownOrAmbiguous).unwrap(),
            "\"unknown_or_ambiguous\""
        );
    }

    #[test]
    fn interval_matches_spec_json() {
        let json = r#"{
            "interval_id": "alive:wd:Q1048", "entity_id": "wd:Q1048",
            "relation": "alive", "start_year": -99, "end_year": -43,
            "start_precision": "year", "end_precision": "year",
            "start_inclusive": true, "end_inclusive": true,
            "confidence": "high", "date_basis": "birth_death",
            "source_id": "wikidata:Q1048",
            "notes": "Astronomical integer year convention used internally."
        }"#;
        let iv: Interval = serde_json::from_str(json).unwrap();
        assert_eq!(iv, caesar_interval());
    }

    #[test]
    fn enum_string_roundtrips() {
        for r in [
            Relation::Alive,
            Relation::Ongoing,
            Relation::Exists,
            Relation::Active,
            Relation::Available,
        ] {
            assert_eq!(Relation::from_str(r.as_str()), Some(r));
        }
        for p in [
            YearPrecision::Day,
            YearPrecision::Month,
            YearPrecision::Year,
            YearPrecision::Decade,
            YearPrecision::Century,
            YearPrecision::Millennium,
            YearPrecision::Approximate,
            YearPrecision::Range,
            YearPrecision::Unknown,
        ] {
            assert_eq!(YearPrecision::from_str(p.as_str()), Some(p));
        }
        for t in [
            EntityType::Person,
            EntityType::Event,
            EntityType::Polity,
            EntityType::Organization,
            EntityType::Work,
            EntityType::Technology,
        ] {
            assert_eq!(EntityType::from_str(t.as_str()), Some(t));
        }
        for c in [Confidence::High, Confidence::Medium, Confidence::Low] {
            assert_eq!(Confidence::from_str(c.as_str()), Some(c));
        }
        for s in [Split::Train, Split::Dev, Split::Test, Split::Sweep] {
            assert_eq!(Split::from_str(s.as_str()), Some(s));
        }
        for b in [
            Band::Interior,
            Band::NearBefore,
            Band::NearAfter,
            Band::FarBefore,
            Band::FarAfter,
            Band::EraConfusable,
            Band::FarFallback,
            Band::Sweep,
        ] {
            assert_eq!(Band::from_str(b.as_str()), Some(b));
        }
    }

    #[test]
    fn enum_serde_strings_match_as_str() {
        // Variant lists are complete as of v0; update when adding variants.
        for v in [
            EntityType::Person,
            EntityType::Event,
            EntityType::Polity,
            EntityType::Organization,
            EntityType::Work,
            EntityType::Technology,
        ] {
            assert_eq!(
                serde_json::to_string(&v).unwrap(),
                format!("\"{}\"", v.as_str())
            );
        }
        for v in [
            Relation::Alive,
            Relation::Ongoing,
            Relation::Exists,
            Relation::Active,
            Relation::Available,
        ] {
            assert_eq!(
                serde_json::to_string(&v).unwrap(),
                format!("\"{}\"", v.as_str())
            );
        }
        for v in [
            YearPrecision::Day,
            YearPrecision::Month,
            YearPrecision::Year,
            YearPrecision::Decade,
            YearPrecision::Century,
            YearPrecision::Millennium,
            YearPrecision::Approximate,
            YearPrecision::Range,
            YearPrecision::Unknown,
        ] {
            assert_eq!(
                serde_json::to_string(&v).unwrap(),
                format!("\"{}\"", v.as_str())
            );
        }
        for v in [Confidence::High, Confidence::Medium, Confidence::Low] {
            assert_eq!(
                serde_json::to_string(&v).unwrap(),
                format!("\"{}\"", v.as_str())
            );
        }
        for v in [Split::Train, Split::Dev, Split::Test, Split::Sweep] {
            assert_eq!(
                serde_json::to_string(&v).unwrap(),
                format!("\"{}\"", v.as_str())
            );
        }
        for v in [
            Band::Interior,
            Band::NearBefore,
            Band::NearAfter,
            Band::FarBefore,
            Band::FarAfter,
            Band::EraConfusable,
            Band::FarFallback,
            Band::Sweep,
        ] {
            assert_eq!(
                serde_json::to_string(&v).unwrap(),
                format!("\"{}\"", v.as_str())
            );
        }
    }
}

/// Shared fixtures for tests across modules (compiled only under cfg(test)).
#[cfg(test)]
pub mod tests_helpers {
    use super::*;

    pub fn caesar_entity() -> Entity {
        Entity {
            entity_id: "wd:Q1048".into(),
            entity_type: EntityType::Person,
            canonical_name: "Julius Caesar".into(),
            aliases: vec!["Gaius Julius Caesar".into(), "Caesar".into()],
            description: "Roman general, statesman, and dictator".into(),
            language: "en".into(),
            source_id: "wikidata:Q1048".into(),
            source_url: "https://www.wikidata.org/wiki/Q1048".into(),
        }
    }

    pub fn caesar_interval() -> Interval {
        Interval {
            interval_id: "alive:wd:Q1048".into(),
            entity_id: "wd:Q1048".into(),
            relation: Relation::Alive,
            start_year: Some(-99),
            end_year: Some(-43),
            start_precision: YearPrecision::Year,
            end_precision: YearPrecision::Year,
            start_inclusive: true,
            end_inclusive: true,
            confidence: Confidence::High,
            date_basis: "birth_death".into(),
            source_id: "wikidata:Q1048".into(),
            notes: "Astronomical integer year convention used internally.".into(),
        }
    }

    pub fn aeneid_entity() -> Entity {
        Entity {
            entity_id: "wd:Q60272".into(),
            entity_type: EntityType::Work,
            canonical_name: "Aeneid".into(),
            aliases: vec![],
            description: "Latin epic poem by Virgil".into(),
            language: "en".into(),
            source_id: "wikidata:Q60272".into(),
            source_url: "https://www.wikidata.org/wiki/Q60272".into(),
        }
    }

    pub fn aeneid_interval() -> Interval {
        Interval {
            interval_id: "available:wd:Q60272".into(),
            entity_id: "wd:Q60272".into(),
            relation: Relation::Available,
            start_year: Some(-18),
            end_year: None,
            start_precision: YearPrecision::Approximate,
            end_precision: YearPrecision::Unknown,
            start_inclusive: true,
            end_inclusive: true,
            confidence: Confidence::Medium,
            date_basis: "publication".into(),
            source_id: "wikidata:Q60272".into(),
            notes: "Open-ended: still available.".into(),
        }
    }

    pub fn caesar_provenance() -> Provenance {
        Provenance {
            entity_id: "wd:Q1048".into(),
            field: "start_year".into(),
            value: serde_json::json!(-99),
            source_name: "Wikidata".into(),
            source_id: "Q1048".into(),
            retrieved_at: "2026-10-02".into(),
            source_statement_id: None,
            human_review_status: "unreviewed".into(),
        }
    }

    /// Wraps a generated example in a minimal probe `Response` (model "m",
    /// zero logit diff, no top-logprobs, 1 ms) for tests that only care about
    /// `p_yes` and the denormalized example fields.
    pub fn fake_response(example: Example, p_yes: f64) -> Response {
        Response {
            p_yes,
            logit_diff: 0.0,
            top_logprobs: vec![],
            model: "m".into(),
            latency_ms: 1,
            example,
        }
    }
}
