from pathlib import Path

import numpy as np
import openpyxl
import pytest
import rasterio
from affine import Affine

from getpak import custom_equations as ce
from getpak.automation import Pipelines
from getpak.commons import Utils
from getpak.output import Raster


def test_examples_and_overrides_are_vectorized():
    bands = {
        "Red": np.array([0.01, 0.02]),
        "RedEdge1": np.array([0.02, 0.06]),
    }
    resolved = ce.resolve_configuration({
        "custom_equations": {
            "enabled": "['example_equation_01', 'example_equation_02']",
            "example_equation_02.a": "1",
            "example_equation_02.b": "2",
            "example_equation_02.c": "3",
        }
    })
    results = ce.evaluate_enabled(resolved, bands)
    np.testing.assert_allclose(results["example_equation_01"][1], [2, 3])
    np.testing.assert_allclose(results["example_equation_02"][1], [17, 34])


def test_custom_domains_missing_bands_signed_values_and_overflow():
    spec = ce.CUSTOM_EQUATION_REGISTRY["example_equation_01"]
    values, diagnostics = ce.evaluate(
        spec,
        {"Red": np.array([1.0, 0.0, -1.0]), "RedEdge1": np.array([2.0, 1.0, -2.0])},
        {},
    )
    np.testing.assert_allclose(values[[0, 2]], [2, 2])
    assert np.isnan(values[1])
    assert diagnostics["zero_denominator"] == 1

    missing, diagnostics = ce.evaluate(spec, {"Red": np.ones(2)}, {})
    assert diagnostics["status"] == "missing_required_band"
    assert np.isnan(missing).all()


def test_configuration_rejects_bad_selection_and_coefficients():
    with pytest.raises(ValueError, match="Unknown custom equation"):
        ce.resolve_configuration({"custom_equations": {"enabled": "['not_registered']"}})
    with pytest.raises(ValueError, match="Unknown coefficient"):
        ce.resolve_configuration({
            "custom_equations": {
                "enabled": "['example_equation_02']",
                "example_equation_02.z": "1",
            }
        })
    with pytest.raises(ValueError, match="duplicate"):
        ce.resolve_configuration({
            "custom_equations": {"enabled": "['example_equation_01', 'example_equation_01']"}
        })


def test_default_off_and_true_false_parsing(tmp_path, monkeypatch):
    config = {
        "client_folder": {
            "inputs": str(tmp_path), "output": str(tmp_path / "out"),
            "wmask_folder": str(tmp_path),
        },
        "processing": {
            "s2_tile": "50RKU", "ac_processor": "GRS", "grs_version": "v20",
            "run_custom_equations": "False",
        },
        "timeseries": {"l2b_algorithms": "[]"}, "roi_vectors": [],
    }
    monkeypatch.setattr("getpak.automation.u.read_config", lambda config_path=None: config)
    pipeline = Pipelines()
    assert pipeline.run_custom_equations is False
    config["processing"]["run_custom_equations"] = "True"
    assert pipeline._as_bool(config["processing"]["run_custom_equations"]) is True


def test_float32_round_trip_and_custom_report_group(tmp_path, monkeypatch):
    values, metadata = Utils.to_float32_custom(
        np.array([[-2.5, 0.0, 3.25, np.nan]]),
        product="example_equation_01", return_metadata=True,
    )
    target = tmp_path / "example_equation_01.tif"
    metadata.update({
        "scene_uid": "scene", "record_id": "record", "processor": "GRS",
        "processor_version": "v20", "tile": "50RKU",
        "equation_id": "example_equation_01", "equation_version": "1.0",
        "coefficients": "{}", "implementation_fingerprint": "fingerprint",
    })
    Raster.array2tiff(values, target, Affine.identity(), "EPSG:32720",
                      no_data=np.nan, metadata=metadata)
    with rasterio.open(target) as source:
        assert source.dtypes == ("float32",)
        assert np.isnan(source.nodata)
        assert source.read(1)[0, :3].tolist() == pytest.approx([-2.5, 0.0, 3.25])
        assert source.tags()["ENCODING_PROFILE"] == "custom-float32"

    report = tmp_path / "report.xlsx"
    Pipelines.build_excel({
        "record": {
            "record_id": "record", "scene_uid": "scene",
            "acquisition_datetime_utc": "2021-09-11 02:00:00",
            "example_equation_01_min": -2.5,
            "example_equation_01_mean": 0.25,
            "example_equation_01_count": 3,
            "example_equation_01_status": "success",
            "example_equation_01_equation_id": "example_equation_01",
        }
    }, report)
    workbook = openpyxl.load_workbook(report)
    assert workbook.sheetnames == ["Water quality", "Processing details"]
    headers = [cell.value for cell in workbook["Water quality"][1]]
    assert "example_equation_01_mean" in headers
    assert "example_equation_01_equation_id" not in headers


def test_underscore_product_discovery(tmp_path, monkeypatch):
    config = {
        "client_folder": {
            "inputs": str(tmp_path), "output": str(tmp_path / "out"),
            "wmask_folder": str(tmp_path),
        },
        "processing": {"s2_tile": "50RKU", "ac_processor": "GRS", "grs_version": "v20"},
        "timeseries": {"l2b_algorithms": "[]"}, "roi_vectors": [],
    }
    monkeypatch.setattr("getpak.automation.u.read_config", lambda config_path=None: config)
    pipeline = Pipelines()
    root = Path(config["client_folder"]["output"]) / "50RKU"
    folder = root / "example_equation_01"
    folder.mkdir(parents=True)
    output = folder / "example_equation_01_20210911T025551_T50RKU_record.tif"
    output.write_bytes(b"fixture")
    found = pipeline.match_file_uid(root, "20210911T025551_T50RKU_record")
    assert found["example_equation_01"] == str(output)
