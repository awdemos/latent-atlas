//! Fixture -> generate -> mock probe -> score -> render, all in a tempdir.

use latent_atlas::DatasetRoot;

#[tokio::test]
async fn fixture_to_render_end_to_end() {
    let tmp = tempfile::tempdir().unwrap();
    let root = DatasetRoot::new(tmp.path());
    let (report, written) = latent_atlas::demo::run_demo(&root, true).await.unwrap();

    // mock model should recover intervals well above chance
    assert!(
        report.overall.accuracy > 0.75,
        "accuracy {}",
        report.overall.accuracy
    );
    assert!(
        report.mean_interval_iou > 0.5,
        "iou {}",
        report.mean_interval_iou
    );
    assert!(report.by_relation.contains_key("alive"));
    assert!(report.by_band.contains_key("near_before"));
    // Pinned from the fixture set: 4 relations (alive, ongoing, exists,
    // available) and 7 bands (interior, near_before, near_after, far_before,
    // far_after, era_confusable, far_fallback); generation is seeded, so
    // these counts are deterministic.
    assert_eq!(
        report.by_relation.len(),
        4,
        "by_relation {:?}",
        report.by_relation.keys().collect::<Vec<_>>()
    );
    assert_eq!(
        report.by_band.len(),
        7,
        "by_band {:?}",
        report.by_band.keys().collect::<Vec<_>>()
    );

    // artifacts exist and are non-empty
    let run_dir = tmp.path().join("runs/mock-atlas-v1");
    let metrics_path = run_dir.join("metrics.json");
    assert!(metrics_path.is_file(), "missing {}", metrics_path.display());

    // metrics.json round-trips the in-memory report
    let metrics_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&metrics_path).unwrap()).unwrap();
    let file_accuracy = metrics_json["overall"]["accuracy"].as_f64().unwrap();
    assert!(
        (file_accuracy - report.overall.accuracy).abs() < 1e-12,
        "metrics.json accuracy {file_accuracy} vs report {}",
        report.overall.accuracy
    );

    // one sweep response file per fixture relation, next to score_maps/
    for relation in ["alive", "ongoing", "exists", "available"] {
        let sweep = run_dir.join(format!("{relation}_yesno_sweep.responses.ndjson"));
        assert!(
            sweep.is_file(),
            "missing sweep responses {}",
            sweep.display()
        );
    }

    let heatmap = run_dir.join("score_maps/heatmap.png");
    assert!(heatmap.is_file(), "missing {}", heatmap.display());
    let bytes = std::fs::read(&heatmap).unwrap();
    assert!(
        bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]),
        "heatmap should be a PNG"
    );
    assert!(
        written
            .iter()
            .any(|p| p.file_name().unwrap() == "calibration.svg")
    );
    assert!(
        written
            .iter()
            .any(|p| p.file_name().unwrap() == "boundary_error.svg")
    );

    // canonical parquet tables exist and round-trip
    let entities =
        latent_atlas::parquet_io::read_entities(&tmp.path().join("canonical/entities.parquet"))
            .unwrap();
    assert!(
        entities.len() >= 12,
        "expected >= 12 entities, got {}",
        entities.len()
    );
    let intervals =
        latent_atlas::parquet_io::read_intervals(&tmp.path().join("canonical/intervals.parquet"))
            .unwrap();
    assert!(
        intervals.len() >= 12,
        "expected >= 12 intervals, got {}",
        intervals.len()
    );
}
