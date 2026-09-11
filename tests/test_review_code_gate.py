from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
from affine import Affine

from getpak import inversion_functions
from getpak.automation import Pipelines
from getpak.commons import Utils
from getpak.methods import Methods
from getpak.output import Raster


def config_for(tmp_path, *, processor='ACOLITE', report_rrs=False):
    return {
        'client_folder': {
            'inputs': str(tmp_path / 'inputs'),
            'output': str(tmp_path / 'output'),
            'wmask_folder': str(tmp_path / 'masks'),
        },
        'processing': {
            's2_tile': '20LLQ',
            'grs_version': 'v20',
            'ac_processor': processor,
            'report_rrs': report_rrs,
            'parallel': False,
        },
        'timeseries': {'l2b_algorithms': '[]'},
        'roi_vectors': [],
    }


def make_pipeline(monkeypatch, tmp_path, **kwargs):
    config = config_for(tmp_path, **kwargs)
    monkeypatch.setattr(
        'getpak.automation.u.read_config', lambda config_path=None: config
    )
    monkeypatch.setattr('getpak.automation.u.set_gdal_driver_path', lambda: None)
    return Pipelines(), config


def scene(path, acquired, mission='S2A'):
    return (
        Path(path),
        {
            'input_file': Path(path),
            'basename': Path(path).name,
            'mission': mission,
            'str_date': acquired.strftime('%Y%m%d'),
            'pydate': acquired,
            'tile': '20LLQ',
        },
    )


def rrs_dataset():
    transform = Affine(20, 0, 0, 0, -20, 40)
    ds = xr.Dataset(
        {'Red': (('y', 'x'), np.ones((2, 3), dtype='float32'))},
        coords={'x': [10.0, 30.0, 50.0], 'y': [30.0, 10.0]},
    )
    ds = ds.rio.set_spatial_dims(x_dim='x', y_dim='y')
    ds = ds.rio.write_crs('EPSG:32720')
    ds.attrs.update({'trans': transform, 'proj': rasterio.crs.CRS.from_epsg(32720)})
    return ds


def write_mask(path, values, transform):
    profile = {
        'driver': 'GTiff',
        'height': values.shape[0],
        'width': values.shape[1],
        'count': 1,
        'dtype': 'uint8',
        'crs': 'EPSG:32720',
        'transform': transform,
        'nodata': 0,
    }
    with rasterio.open(path, 'w', **profile) as target:
        target.write(values.astype('uint8'), 1)
    return path


def test_scaling_policy_round_trip_and_raster_metadata(tmp_path):
    values = np.array([np.nan, -1.0, 0.0, 0.1234, 1000.0, np.inf])
    encoded, metadata = Utils.to_uint16_scaled(
        values, scale=100, unit='NTU', product='Turb',
        return_metadata=True,
    )
    assert encoded.tolist() == [65535, 65535, 0, 12, 65535, 65535]
    assert metadata['invalid_count'] == 2
    assert metadata['negative_count'] == 1
    assert metadata['overflow_count'] == 1
    assert metadata['valid_zero_count'] == 1
    assert encoded[3] * metadata['decode_multiplier'] == pytest.approx(0.12)

    rrs, rrs_meta = Utils.to_uint16_scaled(
        np.array([0.0001]), scale=10000, unit='sr-1', product='Red',
        return_metadata=True,
    )
    assert rrs[0] == 1
    assert rrs[0] * rrs_meta['decode_multiplier'] == pytest.approx(0.0001)

    output = tmp_path / 'scaled.tif'
    Raster.array2tiff(
        encoded.reshape(2, 3), output, Affine.identity(), 'EPSG:4326',
        no_data=65535, metadata=metadata,
    )
    with rasterio.open(output) as source:
        assert source.nodata == 65535
        assert source.scales == pytest.approx((0.01,))
        assert source.units == ('NTU',)
        assert source.tags()['OVERFLOW_COUNT'] == '1'


def test_gilerson3_dispatch_passes_all_three_bands(monkeypatch):
    captured = {}

    def fake(Red, RedEdge1, RedEdge2):
        captured.update(Red=Red, RedEdge1=RedEdge1, RedEdge2=RedEdge2)
        return np.full(Red.shape, 20.0)

    monkeypatch.setattr(inversion_functions, 'chl_gilerson3', fake)
    data = xr.Dataset({
        'Red': (('y', 'x'), [[0.01]]),
        'RedEdge1': (('y', 'x'), [[0.02]]),
        'RedEdge2': (('y', 'x'), [[0.03]]),
    })
    result = Methods().chlorophylla(data, None, limits=False, alg='gilerson3')
    assert set(captured) == {'Red', 'RedEdge1', 'RedEdge2'}
    assert result.item() == pytest.approx(20.0)


def test_unknown_algorithm_is_explicit(monkeypatch, tmp_path):
    pipeline, cfg = make_pipeline(monkeypatch, tmp_path)
    cfg['timeseries']['l2b_algorithms'] = "['DOES_NOT_EXIST']"
    with pytest.raises(ValueError, match='Unknown L2B algorithm'):
        _ = pipeline.l2b_fx_required_bands


def test_grs_discovery_smoke(monkeypatch, tmp_path):
    pipeline, _ = make_pipeline(monkeypatch, tmp_path, processor='GRS')
    path = Path(
        'S2A_MSIL1C_20240101T101112_N0300_R001_T20LLQ_'
        '20240101T120000_cc010_v20.nc'
    )
    monkeypatch.setattr('getpak.automation.u.walktalk', lambda *args, **kwargs: [path])
    records = pipeline.discover_input_files()
    assert records[0][1]['mission'] == 'S2A'
    assert records[0][1]['pydate'] == datetime(2024, 1, 1, 10, 11, 12)


def test_unmatched_scene_is_retained_in_ledger(monkeypatch, tmp_path):
    pipeline, _ = make_pipeline(monkeypatch, tmp_path)
    records = [
        scene('one.nc', datetime(2024, 1, 1, 10)),
        scene('two.nc', datetime(2024, 1, 2, 10)),
    ]
    monkeypatch.setattr(pipeline, 'discover_input_files', lambda: records)
    monkeypatch.setattr(
        'getpak.automation.m.get_waterdetect_masks',
        lambda input_folder: (
            ['20240101'], [Path('WD_20240101T100000_T20LLQ_water_mask.tif')]
        ),
    )
    matches, _, _, _ = pipeline.get_matchups(do_return=True)
    assert len(matches) == 1
    assert [row['status'] for row in pipeline.scene_ledger] == [
        'matched', 'unmatched'
    ]


def test_same_day_scenes_use_unique_timestamped_masks(monkeypatch, tmp_path):
    pipeline, _ = make_pipeline(monkeypatch, tmp_path)
    records = [
        scene(f'scene_{hour}.nc', datetime(2024, 1, 1, hour))
        for hour in (10, 11, 12)
    ]
    masks = [
        Path(f'WD_20240101T{hour:02d}0000_T20LLQ_water_mask.tif')
        for hour in (10, 11, 12)
    ]
    monkeypatch.setattr(pipeline, 'discover_input_files', lambda: records)
    monkeypatch.setattr(
        'getpak.automation.m.get_waterdetect_masks',
        lambda input_folder: (['20240101'] * 3, masks),
    )
    matches, _, _, _ = pipeline.get_matchups(do_return=True)
    assert len(matches) == 3
    assert len(set(matches)) == 3
    assert all(row['status'] == 'matched' for row in pipeline.scene_ledger)


def test_single_same_day_mask_is_reused_and_all_scenes_are_ledgered(
        monkeypatch, tmp_path):
    pipeline, _ = make_pipeline(monkeypatch, tmp_path)
    records = [scene(f'scene_{hour}.nc', datetime(2024, 1, 1, hour)) for hour in (10, 11, 12)]
    monkeypatch.setattr(pipeline, 'discover_input_files', lambda: records)
    monkeypatch.setattr('getpak.automation.m.get_waterdetect_masks', lambda input_folder: (['20240101'], [Path('WD_20240101_T20LLQ_water_mask.tif')]))
    matches, _, _, _ = pipeline.get_matchups(do_return=True)
    assert len(matches) == 3
    assert {row['mask_match_type'] for row in pipeline.scene_ledger} == {'same_day_reuse'}
    assert {row['mask_reuse_status'] for row in pipeline.scene_ledger} == {'reused'}


def test_two_masks_for_one_scene_are_explicitly_ambiguous(monkeypatch, tmp_path):
    pipeline, _ = make_pipeline(monkeypatch, tmp_path)
    records = [scene('one.nc', datetime(2024, 1, 1, 10))]
    monkeypatch.setattr(pipeline, 'discover_input_files', lambda: records)
    monkeypatch.setattr(
        'getpak.automation.m.get_waterdetect_masks',
        lambda input_folder: (
            ['20240101', '20240101'],
            [
                Path('A_20240101_T20LLQ_water_mask.tif'),
                Path('B_20240101_T20LLQ_water_mask.tif'),
            ],
        ),
    )
    with pytest.raises(ValueError, match='No ACOLITE/WaterDetect matchups'):
        pipeline.get_matchups()
    assert pipeline.scene_ledger[0]['status'] == 'ambiguous_mask'


def test_static_mode_is_explicit(monkeypatch, tmp_path):
    pipeline, cfg = make_pipeline(monkeypatch, tmp_path)
    cfg['processing']['mask_mode'] = 'static'
    with pytest.raises(ValueError, match='requires an explicit static_mask'):
        _ = pipeline.static_mask

    static_mask = tmp_path / 'reference.tif'
    static_mask.write_bytes(b'placeholder')
    cfg['processing']['static_mask'] = str(static_mask)
    records = [
        scene('one.nc', datetime(2024, 1, 1, 10)),
        scene('two.nc', datetime(2024, 1, 1, 11)),
    ]
    monkeypatch.setattr(pipeline, 'discover_input_files', lambda: records)
    matches, _, _, _ = pipeline.get_matchups(do_return=True)
    assert len(matches) == 2
    assert all(row['mask_mode'] == 'static' for row in pipeline.scene_ledger)


def test_mask_reprojection_and_explicit_spatial_failures(tmp_path):
    rrs = rrs_dataset()
    coarse = write_mask(
        tmp_path / 'coarse.tif',
        np.ones((2, 2)),
        Affine(30, 0, 0, 0, -20, 40),
    )
    masked, status = Methods.intersect_watermask(
        rrs, coarse, require_full_coverage=True, return_status=True
    )
    assert status == 'matched'
    assert masked['Red'].notnull().all().compute().item()

    partial = write_mask(
        tmp_path / 'partial.tif',
        np.ones((2, 1)),
        Affine(20, 0, 0, 0, -20, 40),
    )
    assert Methods.intersect_watermask(
        rrs, partial, require_full_coverage=True, return_status=True
    )[1] == 'static_mask_incompatible'

    empty = write_mask(
        tmp_path / 'empty.tif',
        np.zeros((2, 3)),
        Affine(20, 0, 0, 0, -20, 40),
    )
    assert Methods.intersect_watermask(
        rrs, empty, return_status=True
    )[1] == 'empty_mask'

    distant = write_mask(
        tmp_path / 'distant.tif',
        np.ones((2, 3)),
        Affine(20, 0, 1000, 0, -20, 1040),
    )
    assert Methods.intersect_watermask(
        rrs, distant, return_status=True
    )[1] == 'no_overlap'


def test_roi_empty_and_partial_statuses(monkeypatch):
    monkeypatch.setattr('getpak.methods.zonal_stats', lambda *args, **kwargs: [])
    assert Methods.shp_stats('x.tif', 'x.shp')['roi_status'] == 'empty_roi'

    monkeypatch.setattr(
        'getpak.methods.zonal_stats',
        lambda *args, **kwargs: [
            {'count': 2, 'min': 1, 'max': 3, 'mean': 2, 'median': 2, 'std': 1},
            {'count': 0, 'min': None, 'max': None, 'mean': None,
             'median': None, 'std': None},
        ],
    )
    stats = Methods.shp_stats('x.tif', 'x.shp')
    assert stats['roi_status'] == 'partial_overlap'
    assert stats['roi_features_with_data'] == 1


def test_report_without_rrs_and_with_missing_rasters_is_nonfatal(
        monkeypatch, tmp_path):
    pipeline, cfg = make_pipeline(monkeypatch, tmp_path, report_rrs=False)
    cfg['roi_vectors'] = [str(tmp_path / 'missing_roi.shp')]
    root = tmp_path / 'output' / '20LLQ'
    for product in ('npix', 'OWT', 'OWTSPM', 'Chla', 'Turb', 'HySPM'):
        (root / product).mkdir(parents=True, exist_ok=True)
    (root / 'npix' / 'npixels_TEST.txt').write_text(
        'Water_pixels;4\nNeg_Rrs_B4;0\nLow_Rrs;0\nOWT_1;1\n'
    )
    for product in ('OWT', 'OWTSPM', 'Chla', 'Turb', 'HySPM'):
        (root / product / f'{product}_TEST.tif').write_bytes(b'not-a-raster')

    outputs = pipeline.build_report()
    assert len(outputs) == 1
    assert Path(outputs[0]).is_file()
    assert not any((root / band).exists() for band in ('Aerosol', 'Blue'))


def test_missing_product_parser_is_explicit():
    parsed = Pipelines._parse_tifs(None, 'roi.shp', prefix='Chla')
    assert parsed['Chla_status'] == 'missing_product'
    assert parsed['Chla_count'] is None


def test_stable_identity_overwrite_protection_and_timing_schema(
        monkeypatch, tmp_path):
    pipeline, _ = make_pipeline(monkeypatch, tmp_path)
    info = scene('same_name.nc', datetime(2024, 1, 1, 10, tzinfo=timezone.utc))[1]
    uid = pipeline._scene_uid('ACOLITE', info)
    assert uid.startswith('ACOLITE_S2A_20240101T100000_T20LLQ_')
    assert uid == pipeline._scene_uid('ACOLITE', info)

    target_dir = tmp_path / 'output' / '20LLQ' / 'Chla'
    target_dir.mkdir(parents=True)
    pipeline.ledger_by_uid = {uid: {
        'scene_uid': uid, 'record_id': 'record123', 'processor': 'ACOLITE',
        'processor_version': 'unknown', 'tile': '20LLQ',
        'acquisition_datetime_utc': '2024-01-01 10:00:00',
    }}
    target = Path(pipeline._output_filename('Chla', 'Chla_', uid, '.tif'))
    target.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(target, 'w', driver='GTiff', height=1, width=1, count=1,
                       dtype='uint16', crs='EPSG:4326', transform=Affine.identity(),
                       nodata=65535) as dst:
        dst.write(np.array([[1]], dtype='uint16'), 1)
        dst.update_tags(SCENE_UID=uid, RECORD_ID='record123', PROCESSOR='ACOLITE',
                        PROCESSOR_VERSION='unknown',
                        GETPAK_ENCODING_VERSION='GETPAK-ENC-2')
    assert pipeline._output_target('Chla', 'Chla_', uid, '.tif') == str(target)
    assert pipeline._target_was_skipped(target)

    required = {
        'discovery', 'read', 'mask_reproject', 'filtering_owt_inversion',
        'raster_write', 'roi_report', 'total',
    }
    assert set(pipeline._timing_schema()) == required
    assert all(value >= 0 for value in pipeline._timing_schema().values())
    assert pipeline.timing_manifest['execution_mode'] == 'serial'


def test_invalid_rrs_stays_invalid_and_zero_stays_valid():
    values = xr.Dataset({"Red": (("y", "x"), [[np.nan, -0.1, 0.0, 1.1, 0.01]])})
    cleaned = Methods._quick_rrs(values, "Red")
    assert np.isnan(cleaned[0, 0])
    assert np.isnan(cleaned[0, 1])
    assert cleaned[0, 2] == 0.0
    assert np.isnan(cleaned[0, 3])
    assert cleaned[0, 4] == pytest.approx(0.01)


def test_uint16_rejects_nonreserved_nodata():
    with pytest.raises(ValueError, match="nodata=65535"):
        Utils.to_uint16_scaled(np.array([0.0, np.nan]), nodata=0)


def test_static_mask_path_resolves_from_settings_directory(monkeypatch, tmp_path):
    pipeline, config = make_pipeline(monkeypatch, tmp_path)
    mask = tmp_path / "reference.tif"
    mask.write_bytes(b"mask")
    config["_config_dir"] = str(tmp_path)
    config["processing"].update(static_mask_mode=True, static_mask_path="reference.tif")
    monkeypatch.setattr(pipeline, "discover_input_files", lambda: [scene("one.nc", datetime(2024, 1, 1, 10))])
    matches, _, _, _ = pipeline.get_matchups(do_return=True)
    assert len(matches) == 1
    assert pipeline.scene_ledger[0]["mask_match_type"] == "static"


def test_retired_raw_runner_fails_explicitly(monkeypatch, tmp_path):
    pipeline, _ = make_pipeline(monkeypatch, tmp_path)
    with pytest.raises(NotImplementedError, match="retired and unsupported"):
        pipeline.run_l2b_raw()


def test_per_band_negative_diagnostics_use_finite_water_support():
    data = xr.Dataset({
        "Blue": (("y", "x"), [[-0.1, 0.0, np.nan, 0.2]]),
        "Red": (("y", "x"), [[0.1, -0.2, np.inf, np.nan]]),
    })
    diagnostics = Pipelines._rrs_diagnostics(data, ("Blue", "Red", "Missing"))
    assert diagnostics["Blue"] == {"neg_count": 1, "finite_count": 3, "neg_fraction": pytest.approx(1 / 3), "reason": None, "unit": "sr-1"}
    assert diagnostics["Red"] == {"neg_count": 1, "finite_count": 2, "neg_fraction": pytest.approx(0.5), "reason": None, "unit": "sr-1"}
    assert diagnostics["Missing"]["neg_count"] is None
    assert diagnostics["Missing"]["reason"] == "missing_band"
    assert diagnostics["Missing"]["neg_fraction"] is None


def test_pipeline_persists_pre_qc_diagnostics_and_invalid_outputs(monkeypatch, tmp_path):
    pipeline, config = make_pipeline(monkeypatch, tmp_path, report_rrs=False)
    data = rrs_dataset()
    for band in ("Aerosol", "Blue", "Green", "RedEdge1", "RedEdge2", "RedEdge3", "Nir2"):
        data[band] = data.Red * 0 + 0.01
    data["Red"].data = np.array([[0.01, -0.01, 0.0], [0.01, np.nan, 0.01]])
    data = data.chunk({"y": 2, "x": 3})
    mask = write_mask(tmp_path / "water.tif", np.array([[0, 1, 1], [1, 1, 1]]), data.attrs["trans"])
    config["processing"].update(mask_mode="static", static_mask=str(mask))
    monkeypatch.setattr(pipeline, "discover_input_files", lambda: [scene("fixture.nc", datetime(2024, 1, 1, 10))])
    monkeypatch.setattr("getpak.automation.i.get_input_nc", lambda **kwargs: data)
    classes = np.array([[0, 1, 2], [2, 2, 2]], dtype="uint8")
    dominant = np.stack([classes, np.zeros_like(classes), np.zeros_like(classes)])
    monkeypatch.setattr("getpak.automation.m.classify_owt_chla_px", lambda **kwargs: (classes.copy(), None))
    monkeypatch.setattr("getpak.automation.m.classify_owt_spm_px", lambda **kwargs: (classes.copy(), None))
    monkeypatch.setattr("getpak.automation.m.classify_owt_chla_weights", lambda **kwargs: (dominant.copy(), (dominant > 0).astype(float)))
    monkeypatch.setattr("getpak.automation.m.blended_chla", lambda **kwargs: np.full((2, 3), 10.0))
    monkeypatch.setattr("getpak.automation.m.turb", lambda **kwargs: np.full((2, 3), 10.0))
    Utils.tic()
    pipeline.get_matchups()
    result = next(iter(pipeline.matchups_to_l2b().values()))
    assert result["status"] == "success"
    assert result["rrs_diagnostics"]["Red"]["neg_count"] == 1
    assert result["rrs_diagnostics"]["Red"]["finite_count"] == 4
    with rasterio.open(result["Chla"]) as source:
        assert source.nodata == 65535
        assert source.read_masks(1)[0, 0] == 0
        assert source.read_masks(1)[0, 1] == 0
    assert "scene_rrs_Red_neg_count;1" in Path(result["npix"]).read_text()
    assert "scene_rrs_Red_finite_count;4" in Path(result["npix"]).read_text()


def test_conflicting_static_mask_aliases_fail(monkeypatch, tmp_path):
    pipeline, config = make_pipeline(monkeypatch, tmp_path)
    config["processing"].update(mask_mode="static", static_mask="old.tif", static_mask_path="new.tif")
    with pytest.raises(ValueError, match="conflicts"):
        _ = pipeline.static_mask
