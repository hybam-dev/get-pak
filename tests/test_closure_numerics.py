import subprocess
import sys
import os

import numpy as np
import pytest
import xarray as xr
import rasterio

from getpak import inversion_functions as ifunc
from getpak.methods import Methods


@pytest.mark.parametrize(
    "function, kwargs",
    [
        (ifunc.chl_gilerson2, {"Red": [0.0, -0.1, np.nan], "RedEdge1": [0.1, 0.01, 0.1]}),
        (ifunc.chl_gons, {"Red": [0.0, 0.1, 0.1], "RedEdge1": [0.1, 0.1, 0.1], "RedEdge3": [0.01, 0.2, 0.2]}),
        (ifunc.chl_OC2, {"Blue": [0.0, -0.1, np.nan], "Green": [0.1, 0.1, 0.1]}),
        (ifunc.spm_jiang2021_green, {"Aerosol": [0.01, 0.01, 0.01], "Blue": [0.01, 0.0, 0.01], "Green": [0.01, -0.4, np.nan], "Red": [0.01, 0.01, 0.01]}),
        (ifunc.spm_jiang2021_red, {"Aerosol": [0.01, -0.01, 0.01], "Blue": [0.01, 0.01, 0.01], "Green": [0.01, 0.01, 0.01], "Red": [0.01, 0.01, -0.4]}),
        (ifunc.spm_s3, {"Red": [-0.01, 0.0, np.nan], "Nir2": [0.01, 0.01, 0.01]}),
    ],
)
def test_formula_edges_emit_no_runtime_warning(function, kwargs):
    with np.errstate(all="raise"):
        values, diagnostics = function(return_diagnostics=True, **kwargs)
    assert values.shape == (3,)
    assert diagnostics["algorithm_evaluations"] == 3
    assert diagnostics["finite_output_count"] <= 3
    assert any(diagnostics[key] for key in diagnostics if key != "finite_output_count")


def test_formula_valid_values_match_reference():
    red = np.array([0.01, 0.02])
    red_edge = np.array([0.02, 0.03])
    expected_gilerson = (0.7864 * (red_edge / red) / 0.022 - 0.4245 / 0.022) ** 1.124
    np.testing.assert_allclose(ifunc.chl_gilerson2(red, red_edge), expected_gilerson)

    blue = np.array([0.01, 0.02])
    green = np.array([0.02, 0.04])
    x = np.log10(blue / green)
    expected_oc2 = 10 ** (0.2389 + -1.9369 * x + 1.7627 * x**2 + -3.0777 * x**3 + -0.1054 * x**4)
    np.testing.assert_allclose(ifunc.chl_OC2(blue, green), expected_oc2)


def test_spm_s3_does_not_evaluate_unselected_branch():
    with np.errstate(all="raise"):
        values, diagnostics = ifunc.spm_s3(
            np.array([0.01, 0.04]), np.array([-1.0, 0.05]), return_diagnostics=True
        )
    assert np.isfinite(values[0])
    assert np.isfinite(values[1])
    assert diagnostics["missing_or_masked_input"] == 0
    expected = 759.12 * ((0.05 / 0.04) ** 1.92)
    assert np.isclose(values[0], 2.79101975e5 * 0.01**2.34858344 + 4.20023206)
    assert np.isclose(values[1], expected)


def test_blend_zero_support_is_missing_not_zero():
    shape = (2, 2)
    rrs = xr.Dataset({
        "Red": (("y", "x"), np.full(shape, 0.01)),
        "Blue": (("y", "x"), np.full(shape, 0.01)),
        "Green": (("y", "x"), np.full(shape, 0.01)),
        "RedEdge1": (("y", "x"), np.full(shape, 0.01)),
        "RedEdge3": (("y", "x"), np.full(shape, 0.01)),
    })
    classes = np.zeros((2, *shape), dtype=int)
    weights = np.zeros((2, *shape), dtype=float)
    methods = Methods()
    result = methods.blended_chla(rrs, classes, weights)
    assert np.isnan(result).all()
    assert methods._numerical_diagnostics["zero_blend_support"] == 4


def test_numpy_error_state_is_not_modified():
    before = np.geterr().copy()
    with pytest.raises(FloatingPointError):
        with np.errstate(all="raise"):
            np.array([1.0]) / 0.0
    assert np.geterr() == before


def test_subprocess_warning_stderr_is_clean(tmp_path):
    script = tmp_path / "warning_probe.py"
    script.write_text(
        "import numpy as np\n"
        "from getpak.inversion_functions import chl_OC2\n"
        "with np.errstate(all='raise'):\n"
        "    value, diagnostics = chl_OC2([0.0, -1.0], [1.0, 1.0], return_diagnostics=True)\n"
        "assert np.isnan(value).any()\n"
    )
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert "RuntimeWarning" not in result.stderr
    assert "Exception ignored" not in result.stderr


def test_jiang_red_does_not_require_unused_green():
    value = ifunc.spm_jiang2021_red(
        Aerosol=0.01, Blue=0.02, Green=np.nan, Red=0.03
    )
    assert value == pytest.approx(56.45774045)


def test_jiang_red_required_band_nan_remains_invalid():
    values = ifunc.spm_jiang2021_red(
        Aerosol=np.array([0.01, np.nan]), Blue=0.02,
        Green=np.nan, Red=0.03,
    )
    assert np.isfinite(values[0])
    assert np.isnan(values[1])


def test_power_domain_uses_exact_exponent_and_zero_rules():
    integer = ifunc.chl_gilerson2(Red=1.0, RedEdge1=0.0, b=2.0)
    assert integer == pytest.approx(372.31456612)
    fractional = ifunc.chl_gilerson2(Red=1.0, RedEdge1=0.0, b=1.124)
    assert np.isnan(fractional)

    zero_base_a = 1.0
    zero_base_red_edge = 0.4245 / 0.7864
    assert ifunc.chl_gilerson2(Red=1.0, RedEdge1=zero_base_red_edge, a=zero_base_a, b=2.0) == pytest.approx(0.0)
    assert np.isnan(ifunc.chl_gilerson2(Red=1.0, RedEdge1=zero_base_red_edge, a=zero_base_a, b=-1.0))
    assert np.isnan(ifunc.chl_gilerson2(Red=1.0, RedEdge1=zero_base_red_edge, a=zero_base_a, b=0.0))


def test_grs_reader_closes_source_when_grid_validation_fails(monkeypatch):
    source = xr.Dataset(
        {
            "Rrs": (("wl", "y", "x"), np.ones((1, 2, 2))),
            "spatial_ref": ((), 0),
        },
        coords={"wl": [443], "time": ("wl", [0]), "x": [10.0, 30.0], "y": [30.0, 10.0]},
    )
    source["spatial_ref"].attrs["crs_wkt"] = rasterio.crs.CRS.from_epsg(32720).to_wkt()
    closed = []

    class TrackedSource:
        variables = source.variables

        def __getitem__(self, name):
            return source[name]

        def close(self):
            closed.append(True)
            source.close()

    tracked_source = TrackedSource()
    gdal_source = type(
        "GDALSource", (), {"GetGeoTransform": lambda self: (0, 20, 0, 40, 0, -20)}
    )()
    monkeypatch.setattr("getpak.input.GRS.metadata", lambda path: {"pydate": None})
    monkeypatch.setattr("getpak.input.dd", type("DD", (), {"grs_v20nc_s2bands": {"Red": 443}}))
    monkeypatch.setattr("getpak.input.gdal", type("GDAL", (), {"Open": staticmethod(lambda _: gdal_source)}))
    monkeypatch.setattr("getpak.input.xr.open_dataset", lambda *args, **kwargs: tracked_source)
    def fail_validation(**kwargs):
        raise ValueError("grid validation failure")
    monkeypatch.setattr("getpak.input.GRS._validated_grid", fail_validation)

    with pytest.raises(ValueError, match="grid validation failure"):
        from getpak.input import GRS
        GRS.get_grs_dict("scene.nc", "v20")
    assert closed == [True]
