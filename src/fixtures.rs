//! Bundled demo fixture: a small hand-checked entity set spanning antiquity
//! to the industrial era. Powers `atlas demo --mock` and the e2e test.

use crate::types::*;

fn mk(
    id: &str,
    entity_type: EntityType,
    name: &str,
    desc: &str,
    relation: Relation,
    start: i32,
    end: Option<i32>,
) -> (Entity, Interval) {
    let entity = Entity {
        entity_id: id.into(),
        entity_type,
        canonical_name: name.into(),
        aliases: vec![],
        description: desc.into(),
        language: "en".into(),
        source_id: format!("fixture:{id}"),
        source_url: String::new(),
    };
    let interval = Interval {
        interval_id: format!("{}:{id}", relation.as_str()),
        entity_id: id.into(),
        relation,
        start_year: Some(start),
        end_year: end,
        start_precision: YearPrecision::Year,
        end_precision: if end.is_some() {
            YearPrecision::Year
        } else {
            YearPrecision::Unknown
        },
        start_inclusive: true,
        end_inclusive: true,
        confidence: Confidence::High,
        date_basis: "fixture".into(),
        source_id: format!("fixture:{id}"),
        notes: String::new(),
    };
    (entity, interval)
}

/// The 15 hand-checked demo pairs: each entity with its single interval,
/// ordered roughly chronologically. All years are astronomical integers.
pub fn demo_entities() -> Vec<(Entity, Interval)> {
    use EntityType::*;
    use Relation::*;
    vec![
        mk(
            "fixture:alexander",
            Person,
            "Alexander the Great",
            "Macedonian king and conqueror",
            Alive,
            -355,
            Some(-322),
        ),
        mk(
            "fixture:pericles",
            Person,
            "Pericles",
            "Athenian statesman",
            Alive,
            -494,
            Some(-428),
        ),
        mk(
            "fixture:caesar",
            Person,
            "Julius Caesar",
            "Roman general and dictator",
            Alive,
            -99,
            Some(-43),
        ),
        mk(
            "fixture:cleopatra",
            Person,
            "Cleopatra",
            "Last Ptolemaic ruler of Egypt",
            Alive,
            -68,
            Some(-30),
        ),
        mk(
            "fixture:augustus",
            Person,
            "Augustus",
            "First Roman emperor",
            Alive,
            -62,
            Some(14),
        ),
        mk(
            "fixture:marcus_aurelius",
            Person,
            "Marcus Aurelius",
            "Roman emperor and Stoic",
            Alive,
            121,
            Some(180),
        ),
        mk(
            "fixture:peloponnesian_war",
            Event,
            "Peloponnesian War",
            "Athens vs Sparta",
            Ongoing,
            -430,
            Some(-403),
        ),
        mk(
            "fixture:third_punic_war",
            Event,
            "Third Punic War",
            "Rome destroys Carthage",
            Ongoing,
            -148,
            Some(-145),
        ),
        mk(
            "fixture:roman_republic",
            Polity,
            "Roman Republic",
            "Rome from 509 to 27 BCE",
            Exists,
            -508,
            Some(-26),
        ),
        mk(
            "fixture:roman_empire",
            Polity,
            "Roman Empire",
            "Rome from 27 BCE",
            Exists,
            -26,
            Some(1453),
        ),
        mk(
            "fixture:western_roman_empire",
            Polity,
            "Western Roman Empire",
            "Western half after 395",
            Exists,
            395,
            Some(476),
        ),
        mk(
            "fixture:han_dynasty",
            Polity,
            "Han dynasty",
            "Chinese dynasty",
            Exists,
            -205,
            Some(220),
        ),
        mk(
            "fixture:aeneid",
            Work,
            "Aeneid",
            "Epic by Virgil",
            Available,
            -18,
            None,
        ),
        mk(
            "fixture:printing_press",
            Technology,
            "Printing press",
            "Movable-type printing",
            Available,
            1450,
            None,
        ),
        mk(
            "fixture:steam_locomotive",
            Technology,
            "Steam locomotive",
            "Rail steam power",
            Available,
            1804,
            None,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GoldLabel;

    #[test]
    fn fixture_intervals_are_sane() {
        let pairs = demo_entities();
        assert!(pairs.len() >= 12);
        for (e, iv) in &pairs {
            assert_eq!(e.entity_id, iv.entity_id);
            if let (Some(a), Some(b)) = (iv.start_year, iv.end_year) {
                assert!(a <= b, "{}: {a} > {b}", e.canonical_name);
            }
            assert_eq!(
                iv.label_at(-9999),
                GoldLabel::No,
                "{} has no business in 10000 BCE",
                e.canonical_name
            );
        }
        // every relation used in the demo must have templates
        for (_, iv) in &pairs {
            assert_eq!(crate::querygen::templates_for(iv.relation).len(), 3);
        }
    }
}
