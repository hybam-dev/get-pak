from datetime import timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
from affine import Affine

from getpak.automation import Pipelines
from getpak.input import ACOLITE_S2, GRS, Input
from getpak.methods import Methods


S2B_BANDS = {
    "Aerosol": "442", "Blue": "492", "Green": "559", "Red": "665",
    "RedEdge1": "704", "RedEdge2": "739", "RedEdge3": "780",
    "Nir1": "833", "Nir2": "864", "Swir1": "1610", "Swir2": "2186",
}


def write_acolite(path, *, tile="T20LLQ", product="L2R", generated_by="ACOLITE",
                  include_bands=True):
    x = np.array([300010.0, 300030.0, 300050.0])
    y = np.array([9000030.0, 9000010.0])
    data = {}
    if include_bands:
        for index, wavelength in enumerate(S2B_BANDS.values(), start=1):
            values = np.full((2, 3), index / 100.0, dtype="float32")
            values[0, 0] = np.nan
            data[f"rhos_{wavelength}"] = (("y", "x"), values)
    crs = rasterio.crs.CRS.from_epsg(32720)
    data["transverse_mercator"] = ((), np.int32(0), {"crs_wkt": crs.to_wkt()})
    ds = xr.Dataset(data, coords={"x": x, "y": y}, attrs={
        "generated_by": generated_by,
        "acolite_file_type": product,
        "sensor": "S2B_MSI",
        "isodate": "2024-06-06T14:45:27.467892+00:00",
        "mgrs_tile": tile,
        "projection_key": "transverse_mercator",
        "proj4_string": "+proj=utm +zone=20 +south +datum=WGS84 +units=m +no_defs",
    })
    ds.to_netcdf(path, engine="h5netcdf")
    return path


def settings(tmp_path, processor=None):
    processing = {"s2_tile": "20LLQ", "grs_version": "v20"}
    if processor is not None:
        processing["ac_processor"] = processor
    return {
        "client_folder": {
            "inputs": str(tmp_path / "inputs"), "output": str(tmp_path / "output"),
            "wmask_folder": str(tmp_path / "masks"),
        },
        "processing": processing,
        "timeseries": {"l2b_algorithms": "[]"},
        "roi_vectors": [],
    }


def pipeline(monkeypatch, tmp_path, processor=None):
    config = settings(tmp_path, processor)
    monkeypatch.setattr("getpak.automation.u.read_config", lambda config_path=None: config)
    return Pipelines()


def test_attribute_metadata_and_filename_fallback(tmp_path):
    path = write_acolite(tmp_path / "arbitrary.nc")
    meta = ACOLITE_S2.metadata(path, require_l2r=True)
    assert (meta["mission"], meta["tile"], meta["prod_type"], meta["str_date"]) == (
        "S2B", "20LLQ", "L2R", "20240606"
    )
    assert meta["pydate"].tzinfo == timezone.utc

    fallback = ACOLITE_S2.metadata(
        tmp_path / "S2B_MSI_2024_06_06_14_45_27_T20LLQ_L2R.nc"
    )
    assert fallback["mission"] == "S2B"
    assert fallback["tile"] == "20LLQ"
    assert fallback["pydate"].hour == 14


def test_bands_values_nan_crs_transform_and_lazy_compute(tmp_path):
    path = write_acolite(tmp_path / "scene.nc")
    reader = ACOLITE_S2()
    rrs, meta, crs, transform = reader.get_aco_dict(path)
    assert list(rrs.data_vars) == list(S2B_BANDS)
    assert rrs["Blue"].chunks is not None
    assert np.isnan(rrs["Blue"].isel(y=0, x=0).compute().item())
    assert rrs["Blue"].isel(y=1, x=1).compute().item() == pytest.approx(0.02 / np.pi)
    assert crs.to_epsg() == 32720
    assert crs.to_epsg() != 16120
    assert transform == Affine(20, 0, 300000, 0, -20, 9000040)
    assert rrs.rio.bounds() == pytest.approx((300000, 9000000, 300060, 9000040))
    rrs.close()


def test_missing_bands_are_reported_together(tmp_path):
    path = write_acolite(tmp_path / "missing.nc", include_bands=False)
    with pytest.raises(ValueError) as error:
        ACOLITE_S2().get_aco_dict(path)
    assert "rhos_442" in str(error.value)
    assert "rhos_2186" in str(error.value)


def test_discovery_filters_tile_and_rejects_contradictions(monkeypatch, tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    write_acolite(inputs / "S2B_MSI_2024_06_06_14_45_27_T20LLQ_L2R.nc")
    write_acolite(inputs / "S2B_MSI_2024_06_07_14_45_27_T21LWG_L2R.nc", tile="T21LWG")
    write_acolite(inputs / "S2B_MSI_2024_06_08_14_45_27_T20LLQ_L2W.nc", product="L2W")
    (inputs / "unrelated.nc").write_bytes(b"not netcdf")
    p = pipeline(monkeypatch, tmp_path, " acolite ")
    records = p.discover_input_files()
    assert [record[1]["str_date"] for record in records] == ["20240606"]

    write_acolite(inputs / "S2B_MSI_2024_06_09_14_45_27_T20LLQ_L2R.nc", generated_by="OTHER")
    with pytest.raises(ValueError, match="Contradictory processor metadata"):
        p.discover_input_files()


def test_configuration_default_validation_and_zero_inputs(monkeypatch, tmp_path):
    (tmp_path / "inputs" / "20LLQ").mkdir(parents=True)
    p = pipeline(monkeypatch, tmp_path)
    assert p.ac_processor == "GRS"
    with pytest.raises(ValueError, match="No valid GRS inputs"):
        p.discover_input_files()

    p.settings["processing"]["ac_processor"] = "SeaDAS"
    with pytest.raises(ValueError, match="GRS, ACOLITE"):
        _ = p.ac_processor


def test_pipeline_zero_matchups_is_explicit(monkeypatch, tmp_path):
    inputs = tmp_path / "inputs"
    masks = tmp_path / "masks"
    inputs.mkdir()
    masks.mkdir()
    write_acolite(inputs / "S2B_MSI_2024_06_06_14_45_27_T20LLQ_L2R.nc")
    p = pipeline(monkeypatch, tmp_path, "ACOLITE")
    monkeypatch.setattr("getpak.automation.u.set_gdal_driver_path", lambda: None)
    with pytest.raises(ValueError, match="No ACOLITE/WaterDetect matchups"):
        p.get_matchups()


def test_input_routes_processors(monkeypatch):
    monkeypatch.setattr(GRS, "get_grs_dict", lambda self, grs_nc_file, grs_version: ("grs", {}, None, None))
    monkeypatch.setattr(ACOLITE_S2, "get_aco_dict", lambda self, aco_nc_file: ("acolite", {}, None, None))
    monkeypatch.setattr(Input, "test_valid_file", lambda self, file: None)
    reader = Input()
    assert reader.get_input_nc("a.nc", AC_processor="grs", grs_version="v20") == "grs"
    assert reader.get_input_nc("a.nc", AC_processor=" ACOLITE ") == "acolite"


def test_zero_matchups_and_ambiguous_masks():
    with pytest.raises(ValueError, match="Ambiguous WaterDetect masks"):
        Methods.sch_date_matchups(
            ["20240606"], ["20240606", "20240606"], [Path("scene.nc")],
            [Path("A_20240606_T20LLQ_water_mask.tif"), Path("B_20240606_T20LLQ_water_mask.tif")],
            tile_id="20LLQ",
        )
    matches, _, dates = Methods.sch_date_matchups(
        ["20240606"], ["20240606"], [Path("scene.nc")],
        [Path("A_20240606_T21LWG_water_mask.tif")], tile_id="20LLQ",
    )
    assert matches == {}
    assert dates == []


def test_water_mask_overlap_and_no_overlap(tmp_path):
    scene = write_acolite(tmp_path / "scene.nc")
    rrs, _, _, _ = ACOLITE_S2().get_aco_dict(scene)
    profile = {
        "driver": "GTiff", "height": 2, "width": 3, "count": 1, "dtype": "uint8",
        "crs": "EPSG:32720", "transform": Affine(20, 0, 300000, 0, -20, 9000040),
        "nodata": 0,
    }
    matching = tmp_path / "matching.tif"
    with rasterio.open(matching, "w", **profile) as target:
        target.write(np.ones((2, 3), dtype="uint8"), 1)
    masked = Methods.intersect_watermask(rrs, matching)
    assert masked is not None
    assert masked["Red"].notnull().any().compute().item()

    distant = tmp_path / "distant.tif"
    profile["transform"] = Affine(20, 0, 600000, 0, -20, 8000000)
    with rasterio.open(distant, "w", **profile) as target:
        target.write(np.ones((2, 3), dtype="uint8"), 1)
    assert Methods.intersect_watermask(rrs, distant) is None
    masked.close()
    rrs.close()
