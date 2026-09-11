from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
from affine import Affine
from shapely.geometry import box

from getpak import input as input_module
from getpak.input import GRS
from getpak.methods import Methods
from getpak.output import Raster


CRS = "EPSG:32650"


def regular_grs_dataset():
    x = np.array([100.0, 120.0, 140.0, 160.0])
    y = np.array([220.0, 200.0, 180.0])
    data = {
        name: (("y", "x"), np.arange(12, dtype="float32").reshape(3, 4) + index)
        for index, name in enumerate(("Red", "Green"))
    }
    ds = xr.Dataset(data, coords={"x": x, "y": y})
    return ds, x, y


def test_stale_transform_is_replaced_and_export_matches_coordinates(tmp_path):
    ds, x, y = regular_grs_dataset()
    repaired, transform, diagnostic = GRS._validated_grid(
        ds,
        rasterio.crs.CRS.from_string(CRS),
        Affine(10, 0, 90, 0, -10, 230),
        "synthetic stale transform",
    )

    assert transform == Affine(20, 0, 90, 0, -20, 230)
    assert repaired.rio.transform() == transform
    assert repaired.rio.bounds() == pytest.approx((90, 170, 170, 230))
    assert diagnostic["original_transform"] == (10.0, 0.0, 90.0, 0.0, -10.0, 230.0)
    assert diagnostic["selected_transform"] == (20.0, 0.0, 90.0, 0.0, -20.0, 230.0)
    assert "replaced stale" in diagnostic["reason"]

    output = tmp_path / "red.tif"
    Raster.array2tiff(repaired["Red"].values, output, transform, CRS, no_data=-9999)
    with rasterio.open(output) as source:
        assert (source.width, source.height) == (4, 3)
        assert source.transform == transform
        assert source.bounds == pytest.approx((90, 170, 170, 230))
        assert source.xy(0, 0) == pytest.approx((100, 220))
        np.testing.assert_array_equal(source.read(1), repaired["Red"].values)
    repaired.close()


def test_mask_alignment_and_roi_use_the_selected_grid(tmp_path):
    ds, _, _ = regular_grs_dataset()
    repaired, transform, _ = GRS._validated_grid(
        ds,
        rasterio.crs.CRS.from_string(CRS),
        Affine(10, 0, 90, 0, -10, 230),
        "synthetic stale transform",
    )
    mask_values = np.array([[0, 1, 0, 1], [1, 1, 0, 0], [0, 1, 1, 0]], dtype="uint8")
    mask_path = tmp_path / "water.tif"
    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        height=3,
        width=4,
        count=1,
        dtype="uint8",
        crs=CRS,
        transform=transform,
        nodata=255,
    ) as target:
        target.write(mask_values, 1)

    masked, status = Methods.intersect_watermask(
        repaired, mask_path, return_status=True
    )
    assert status == "matched"
    np.testing.assert_array_equal(
        np.isfinite(masked["Red"].values), mask_values.astype(bool)
    )

    output = tmp_path / "masked.tif"
    Raster.array2tiff(masked["Red"].values, output, transform, CRS, no_data=-9999)
    stats = Methods.shp_stats(output, box(89, 179, 149, 231))
    assert stats["count"] == int(mask_values[:, :3].sum())
    masked.close()
    repaired.close()


def test_coherent_grid_is_retained_and_coordinate_failures_are_explicit():
    ds, _, _ = regular_grs_dataset()
    coherent, transform, diagnostic = GRS._validated_grid(
        ds,
        rasterio.crs.CRS.from_string(CRS),
        Affine(20, 0, 90, 0, -20, 230),
        "synthetic coherent transform",
    )
    assert transform == Affine(20, 0, 90, 0, -20, 230)
    assert diagnostic["reason"].startswith("validated coordinate-derived transform agrees")
    coherent.close()

    irregular = ds.assign_coords(x=[100.0, 120.0, 141.0, 160.0])
    with pytest.raises(ValueError, match="spacing is irregular"):
        GRS._validated_grid(
            irregular,
            rasterio.crs.CRS.from_string(CRS),
            Affine.identity(),
            "synthetic",
        )
    with pytest.raises(ValueError, match="projected CRS"):
        GRS._validated_grid(
            ds,
            rasterio.crs.CRS.from_epsg(4326),
            Affine.identity(),
            "synthetic",
        )
    bad_dims = ds.rename({"x": "x2"}).assign_coords(x=ds.x.values)
    with pytest.raises(ValueError, match="dimensions"):
        GRS._validated_grid(
            bad_dims,
            rasterio.crs.CRS.from_string(CRS),
            Affine.identity(),
            "synthetic",
        )
    ds.close()


def test_actual_v20_reader_reconciles_stale_subdataset_transform(monkeypatch, tmp_path):
    _, x, y = regular_grs_dataset()
    wavelengths = np.array([443, 490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190])
    values = np.arange(11 * 3 * 4, dtype="float32").reshape(11, 3, 4)
    crs = rasterio.crs.CRS.from_string(CRS)
    spatial_ref = xr.DataArray(
        0,
        attrs={"crs_wkt": crs.to_wkt(), "GeoTransform": "90 10 0 230 0 -10"},
    )
    source = xr.Dataset(
        {
            "Rrs": (
                ("wl", "y", "x"),
                values,
                {"grid_mapping": "spatial_ref"},
            ),
            "spatial_ref": spatial_ref,
        },
        coords={
            "wl": wavelengths,
            "x": x,
            "y": y,
            "time": np.datetime64("2021-05-09T02:55:39"),
        },
    )

    class FakeGdal:
        def GetGeoTransform(self):
            return (90.0, 10.0, 0.0, 230.0, 0.0, -10.0)

    monkeypatch.setattr(input_module.gdal, "Open", lambda _: FakeGdal())
    monkeypatch.setattr(input_module.xr, "open_dataset", lambda *args, **kwargs: source)

    path = tmp_path / "S2B_MSIL2Agrs_20210509T025539_N0500_R032_T50RKU_20230309T130856.nc"
    path.write_bytes(b"synthetic")
    grs, _, _, transform = GRS.get_grs_dict(path, "v20")
    assert transform == Affine(20, 0, 90, 0, -20, 230)
    assert grs.rio.transform() == transform
    assert grs.attrs["grid_validation"]["original_transform"][0] == 10.0
    assert grs.attrs["grid_validation"]["selected_transform"][0] == 20.0
    assert grs["Red"].dims == ("y", "x")
    grs.close()
