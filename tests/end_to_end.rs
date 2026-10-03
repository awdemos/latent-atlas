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

    // artifacts exist and are non-empty
    let run_dir = tmp.path().join("runs/mock-atlas-v1");
    assert!(run_dir.join("metrics.json").is_file());
    let heatmap = run_dir.join("score_maps/heatmap.png");
    assert!(heatmap.is_file());
    let bytes = std::fs::read(&heatmap).unwrap();
    assert_eq!(
        &bytes[..4],
        &[0x89, b'P', b'N', b'G'],
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
    assert!(entities.len() >= 12);
}
