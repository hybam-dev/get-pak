import os
import sys
import ast
import json
import inspect
import hashlib
import platform
import time
import re
from importlib import metadata as package_metadata
import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path
from datetime import datetime, timezone
from getpak import inversion_functions as ifunc
from getpak.input import Input
from getpak.input import GRS as g
from getpak.input import ACOLITE_S2
from getpak.output import Raster as r
from getpak.commons import Utils as u
from getpak.methods import Methods

grs = g()
i = Input()
m = Methods()

class Pipelines:
    
    def __init__(self, config_path=None):
        # GET-Pak settings
        self.settings = u.read_config(config_path=config_path)
        self.settings['_config_dir'] = str(Path(config_path).resolve().parent) if config_path else str(Path(__file__).resolve().parent.parent)
        self.INSTANCE_TIME_TAG = datetime.now().strftime('%Y%m%dT%H%M%S')
        self._run_started_perf = time.perf_counter()
        self.run_started_utc = datetime.now(timezone.utc).isoformat()
        self.scene_ledger = []
        self.timing_manifest = {
            'started_utc': self.run_started_utc,
            'execution_mode': 'serial',
            'configured_parallel': self.configured_parallel,
            'software_versions': self._runtime_versions(),
            'stages_s': self._timing_schema(),
            'scenes': {},
        }


    @property
    def input_folder(self):
        return self.settings.get('client_folder', 'inputs')['inputs']

    @property
    def output_folder(self):
        return self.settings.get('client_folder', 'output')['output']
    
    @property
    def wmask_folder(self):
        return self.settings.get('client_folder', 'wmask_folder')['wmask_folder']
    
    @property
    def roi_vectors(self):
        return self.settings['roi_vectors']
    
    @property
    def compute_l2b(self):
        return self._as_bool(
            self.settings.get('processing', {}).get('compute_l2b', False)
        )
        
    @property
    def make_report(self):
        return self._as_bool(
            self.settings.get('processing', {}).get('make_report', False)
        )
    
    @property
    def report_rrs(self):
        return self._as_bool(
            self.settings.get('processing', {}).get('report_rrs', False)
            )

    @property
    def tile_id(self):
        return self.settings.get('processing', 's2_tile')['s2_tile']

    @property
    def ac_processor(self):
        processor = self.settings.get('processing', {}).get('ac_processor', 'GRS')
        processor = str(processor).strip().upper()
        supported = ('GRS', 'ACOLITE')
        if processor not in supported:
            raise ValueError(
                f"Unsupported ac_processor {processor!r}. Supported batch processors: "
                f"{', '.join(supported)}."
            )
        return processor
    
    # @property
    # def grs_files(self):
    #     grs_file_list = u.walktalk(self.input_folder, unwanted_string='*_anc*')
    #     return grs_file_list
    
    @property
    def grs_file_version(self):
        if self.ac_processor != 'GRS':
            return None
        try:
            return self.settings['processing']['grs_version']
        except KeyError as exc:
            raise ValueError("grs_version is required when ac_processor=GRS.") from exc

    @staticmethod
    def _normalize_tile(tile):
        return str(tile).strip().upper().removeprefix('T')

    def discover_input_files(self):
        """Discover validated input products for the selected processor and tile."""
        tile = self._normalize_tile(self.tile_id)
        if self.ac_processor == 'GRS':
            search_root = os.path.join(self.input_folder, self.tile_id)
            input_files = u.walktalk(search_root, unwanted_string='*_anc*')
            metadata_reader = g.metadata
            filename_rule = '*.nc excluding *_anc*'
        else:
            search_root = self.input_folder
            filename_rule = '*_L2R.nc with ACOLITE/L2R metadata'
            candidates = sorted(Path(search_root).rglob('*_L2R.nc'))
            input_files = []
            for candidate in candidates:
                info = ACOLITE_S2.metadata(candidate, require_l2r=True)
                if self._normalize_tile(info['tile']) == tile:
                    input_files.append(candidate)
            metadata_reader = lambda path: ACOLITE_S2.metadata(path, require_l2r=True)

        records = [(path, metadata_reader(path)) for path in input_files]
        records.sort(key=lambda item: (item[1]['pydate'], str(item[0])))
        if not records:
            raise ValueError(
                f"No valid {self.ac_processor} inputs found under {search_root!r}; "
                f"expected {filename_rule} for tile {tile}."
            )
        return records

    @property
    def l2b_functions(self):
        l2b_algos = ast.literal_eval(self.settings['timeseries']['l2b_algorithms'])
        return l2b_algos

    @property
    def l2b_fx_required_bands(self):
        """
        Returns a dictionary of L2B algorithms from inversion_functions.py 
        as defined by the user in the settings.ini and their required bands.
        """
        fx_req_bands = {}
        for algo in self.l2b_functions:
            if algo not in ifunc.functions:
                raise ValueError(
                    f'Unknown L2B algorithm {algo!r}; register it in '
                    'getpak.inversion_functions.functions first.'
                )
            function = ifunc.functions[algo]['function']
            signature = inspect.signature(function)
            required_bands = [name for name, param in signature.parameters.items() if param.default == inspect.Parameter.empty]
            fx_req_bands[algo] = required_bands
        return fx_req_bands         

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


    @property
    def mask_mode(self):
        configured_static = self._as_bool(self.settings.get('processing', {}).get('static_mask_mode', False))
        mode = str(
            self.settings.get('processing', {}).get('mask_mode', 'scene_specific')
        ).strip().lower()
        if mode not in {'scene_specific', 'static'}:
            raise ValueError("mask_mode must be 'scene_specific' or 'static'.")
        if configured_static and mode == 'scene_specific':
            return 'static'
        return mode

    @property
    def static_mask(self):
        processing = self.settings.get('processing', {})
        modern = processing.get('static_mask_path')
        legacy = processing.get('static_mask')
        if modern and legacy and Path(modern) != Path(legacy):
            raise ValueError('static_mask_path conflicts with legacy static_mask.')
        value = modern or legacy
        if self.mask_mode == 'static' and not value:
            raise ValueError('mask_mode=static requires an explicit static_mask path; static_mask_mode=True requires static_mask_path.')
        return value

    @property
    def overwrite_outputs(self):
        return self._as_bool(
            self.settings.get('processing', {}).get('overwrite_outputs', False)
        )

    @property
    def configured_parallel(self):
        return self._as_bool(
            self.settings.get('processing', {}).get('parallel', False)
        )

    @staticmethod
    def _scene_uid(processor, info):
        acquired = info['pydate']
        if getattr(acquired, 'tzinfo', None) is not None:
            acquired = acquired.astimezone(timezone.utc).replace(tzinfo=None)
        timestamp = acquired.strftime('%Y%m%dT%H%M%S')
        mission = str(info.get('mission', 'UNKNOWN')).strip().upper()
        tile = Pipelines._normalize_tile(info.get('tile', 'UNKNOWN'))
        identity = '|'.join((str(info.get('input_file')), str(info.get('product_id', '')), str(info.get('processor', processor))))
        source_tag = hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]
        return f'{processor}_{mission}_{timestamp}_T{tile}_{source_tag}'

    @staticmethod
    def _timestamp_from_name(path):
        match = re.search(
            r'(20\d{6}T\d{6})', os.path.basename(os.fspath(path))
        )
        return match.group(1) if match else None

    @staticmethod
    def _rrs_diagnostics(rrs_dict, bands):
        diagnostics = {}
        for band in bands:
            if band not in rrs_dict:
                diagnostics[band] = {'neg_count': None, 'finite_count': None, 'reason': 'missing_band', 'unit': 'sr-1'}
                continue
            values = np.asarray(rrs_dict[band].values, dtype=float)
            finite = np.isfinite(values)
            diagnostics[band] = {'neg_count': int(np.count_nonzero(finite & (values < 0))), 'finite_count': int(np.count_nonzero(finite)), 'reason': None, 'unit': 'sr-1'}
        return diagnostics

    @staticmethod
    def _runtime_versions():
        versions = {'python': platform.python_version()}
        for distribution in (
            'getpak', 'numpy', 'pandas', 'xarray', 'dask', 'rasterio',
            'rioxarray', 'rasterstats', 'GDAL', 'scikit-learn',
        ):
            try:
                versions[distribution] = package_metadata.version(distribution)
            except package_metadata.PackageNotFoundError:
                versions[distribution] = 'not-installed'
        return versions

    @staticmethod
    def _timing_schema():
        return {
            name: 0.0 for name in (
                'discovery', 'read', 'mask_reproject',
                'filtering_owt_inversion', 'raster_write',
                'roi_report', 'total',
            )
        }

    def _ledger_target(self):
        return os.path.join(
            self.output_folder, self.tile_id,
            f'{self.INSTANCE_TIME_TAG}_scene_ledger.json',
        )

    def _timing_target(self):
        return os.path.join(
            self.output_folder, self.tile_id,
            f'{self.INSTANCE_TIME_TAG}_timing_manifest.json',
        )

    @staticmethod
    def _write_json(path, payload):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        temporary = f'{path}.tmp'
        with open(temporary, 'w', encoding='utf-8') as target:
            json.dump(payload, target, indent=2, sort_keys=True)
        os.replace(temporary, path)

    def _write_scene_ledger(self):
        self.scene_ledger_path = self._ledger_target()
        self._write_json(self.scene_ledger_path, self.scene_ledger)

    def _write_timing_manifest(self):
        self.timing_manifest_path = self._timing_target()
        self._write_json(self.timing_manifest_path, self.timing_manifest)

    def _output_target(self, folder, prefix, scene_uid, suffix):
        target = os.path.join(
            self.output_folder, self.tile_id, folder,
            f'{prefix}{scene_uid}{suffix}',
        )
        if os.path.exists(target) and not self.overwrite_outputs:
            raise FileExistsError(
                f'Refusing to overwrite existing output: {target}. '
                'Set overwrite_outputs=True only after reviewing the collision.'
            )
        return target

    def _write_scaled_raster(self, values, folder, prefix, scene_uid, *,
                             scale, unit, transform, projection):
        target = self._output_target(folder, prefix, scene_uid, '.tif')
        encoded, metadata = u.to_uint16_scaled(
            values, scale=scale, nodata=65535, unit=unit,
            product=prefix.rstrip('_'), return_metadata=True,
        )
        r.array2tiff(
            ndarray_data=encoded,
            str_output_file=target,
            transform=transform,
            projection=projection,
            no_data=65535,
            metadata=metadata,
        )
        return target, metadata

    def _write_categorical_raster(self, values, folder, prefix, scene_uid, *,
                                  transform, projection):
        target = self._output_target(folder, prefix, scene_uid, '.tif')
        r.array2tiff(
            ndarray_data=values,
            str_output_file=target,
            transform=transform,
            projection=projection,
            no_data=0,
        )
        return target
    def run(self, compute_l2b=None, make_report=None):
        """
        Run the GET-Pak processing flow.

        If compute_l2b or make_report are None, the values are read from settings.ini.
        """
        compute_l2b = self.compute_l2b if compute_l2b is None else compute_l2b
        make_report = self.make_report if make_report is None else make_report

        if compute_l2b:
            print('Compute L2B set to True.')
            self.get_matchups()
            self.matchups_to_l2b()
        
        if make_report:
            print('Generating report.')
            self.build_report()

        return self

    def get_matchups(self, do_return=False):
        """Match inputs to masks while preserving one ledger row per input."""
        u.set_gdal_driver_path()
        sep_trace = u.repeat_to_length('-', 22)
        print(sep_trace)
        print(f'Running L2B algorithms for {self.ac_processor} with WD intersection...')
        print(f'Processing tile: {self.tile_id}')
        print(f'Input {self.ac_processor} root: {self.input_folder}')

        discovery_start = time.perf_counter()
        records = self.discover_input_files()
        tile = self._normalize_tile(self.tile_id)

        if self.mask_mode == 'static':
            static_mask = Path(self.static_mask)
            if not static_mask.is_absolute():
                static_mask = Path(self.settings['_config_dir']) / static_mask
            if not static_mask.is_file():
                raise ValueError(f'Configured static_mask does not exist: {static_mask}')
            mask_records = [{'date': None, 'path': static_mask, 'tile': tile}]
        else:
            wd_dates, wd_masks = m.get_waterdetect_masks(
                input_folder=self.wmask_folder
            )
            mask_records = [
                {
                    'date': date,
                    'path': path,
                    'tile': m._tile_from_name(path),
                    'timestamp': self._timestamp_from_name(path),
                }
                for date, path in zip(wd_dates, wd_masks)
            ]

        ledger = []
        provisional = {}
        for source_path, info in records:
            scene_uid = self._scene_uid(self.ac_processor, info)
            record_id = hashlib.sha256(
                f'{scene_uid}|{Path(source_path)}'.encode('utf-8')
            ).hexdigest()[:20]
            entry = {
                'scene_uid': scene_uid, 'record_id': record_id,
                'processor': self.ac_processor,
                'mission': str(info.get('mission', 'UNKNOWN')),
                'acquisition_time': info['pydate'].isoformat(),
                'tile': tile,
                'source_path': str(source_path),
                'mask_path': None,
                'mask_mode': self.mask_mode,
                'status': 'unmatched',
                'reason': 'no matching date/tile mask',
                'outputs': {},
                'error': None,
            }
            if self.mask_mode == 'static':
                candidates = mask_records
            else:
                candidates = [
                    item for item in mask_records
                    if item['date'] == info['str_date']
                    and item['tile'] in {None, tile}
                ]
                if candidates:
                    acquired = info['pydate']
                    if getattr(acquired, 'tzinfo', None) is not None:
                        acquired = acquired.astimezone(timezone.utc)
                    scene_timestamp = acquired.strftime('%Y%m%dT%H%M%S')
                    exact = [item for item in candidates if item['timestamp'] == scene_timestamp]
                    if exact:
                        candidates = exact
                        entry['mask_match_type'] = 'exact'
                    elif len(candidates) == 1:
                        entry['mask_match_type'] = 'same_day_reuse'
                    elif all(item['timestamp'] for item in candidates):
                        scene_time = datetime.strptime(scene_timestamp, '%Y%m%dT%H%M%S')
                        distances = [abs((datetime.strptime(item['timestamp'], '%Y%m%dT%H%M%S') - scene_time).total_seconds()) for item in candidates]
                        nearest = min(distances)
                        candidates = [item for item, distance in zip(candidates, distances) if distance == nearest]
                        entry['mask_match_type'] = 'nearest_same_day'
                    else:
                        entry['mask_match_type'] = 'ambiguous_same_day'

            if len(candidates) > 1:
                entry.update({
                    'status': 'ambiguous_mask',
                    'reason': 'multiple masks match this scene date/tile',
                    'candidate_masks': [str(item['path']) for item in candidates],
                })
            elif len(candidates) == 1:
                mask_path = str(candidates[0]['path'])
                entry.update({
                    'mask_path': mask_path,
                    'status': 'matched',
                    'reason': None,
                })
                entry.setdefault('mask_match_type', 'static' if self.mask_mode == 'static' else 'same_day_reuse')
                provisional.setdefault(mask_path, []).append(entry)
            ledger.append(entry)

        for entries in provisional.values():
            for entry in entries:
                entry['mask_reuse_status'] = 'reused' if len(entries) > 1 else 'single_use'

        matches = {}
        meta = {}
        for entry, (source_path, info) in zip(ledger, records):
            if entry['status'] != 'matched':
                continue
            scene_uid = entry['record_id']
            if scene_uid in matches:
                entry.update({
                    'status': 'identity_collision',
                    'reason': 'stable scene identity is not unique',
                })
                continue
            matches[scene_uid] = {
                'IMG': source_path,
                'WM': Path(entry['mask_path']),
            }
            meta[scene_uid] = info

        self.scene_ledger = ledger
        self.ledger_by_uid = {item['record_id']: item for item in ledger}
        self.matches = matches
        self.str_matches = {
            key: {'IMG': str(value['IMG']), 'WM': str(value['WM'])}
            for key, value in matches.items()
        }
        self.dates = list(matches)
        self.meta = meta
        self.timing_manifest['stages_s']['discovery'] = (
            time.perf_counter() - discovery_start
        )
        self._write_scene_ledger()

        if not matches:
            raise ValueError(
                f'No {self.ac_processor}/WaterDetect matchups found for tile {tile}; '
                f'see {self.scene_ledger_path}.'
            )

        print(f'get_matchups: {len(matches)} matched of {len(records)} discovered.')
        if do_return:
            return matches, self.str_matches, self.dates, meta
        return None
    
    def matchups_to_l2b(self):
        """
        TO-DO
        """
        report_rrs = self.report_rrs
        matches = self.matches
        str_matches = self.str_matches

        results = {}
        tot = len(self.matches)
        sep_trace = u.repeat_to_length('-', 22)
        imgs_out = os.path.join(self.output_folder, self.tile_id)
        # Creating output folder structure
        Path(os.path.join(imgs_out, "npix")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(imgs_out, "OWT")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(imgs_out, "OWTSPM")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(imgs_out, "Chla")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(imgs_out, "Turb")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(imgs_out, "HySPM")).mkdir(parents=True, exist_ok=True)
        # Rrs bands
        if report_rrs:
            Path(os.path.join(imgs_out, "Aerosol")).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(imgs_out, "Blue")).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(imgs_out, "Green")).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(imgs_out, "Red")).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(imgs_out, "RedEdge1")).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(imgs_out, "RedEdge2")).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(imgs_out, "RedEdge3")).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(imgs_out, "Nir2")).mkdir(parents=True, exist_ok=True)
        
        for n, key in enumerate(matches):
            print(sep_trace)
            print(f'Processing: {n+1}/{tot} - {key}')
            
            scene_start = time.perf_counter()
            scene_timing = {
                'stages_s': self._timing_schema(),
                'scene_dimensions': None,
                'valid_water_pixels': 0,
                'roi_count': len(self.roi_vectors),
            }
            results[key] = {
                'IMG': str_matches[key]['IMG'],
                'WM': str_matches[key]['WM'],
                'scene_uid': key,
                'processor': self.ac_processor,
                'source_path': str_matches[key]['IMG'],
                'status': 'processing',
                'error': None,
                'scaling': {},
            }
            ledger_entry = self.ledger_by_uid[key]
            ledger_entry['status'] = 'processing'
            
            results[key].update({'npix': 'empty'})
            results[key].update({'OWT': 'empty'})
            results[key].update({'OWTSPM': 'empty'})   
            results[key].update({'Chla': 'empty'})
            results[key].update({'Turb': 'empty'})
            results[key].update({'HySPM': 'empty'})
            # Rrs bands
            if report_rrs:
                results[key].update({'Aerosol': 'empty'})  # 443
                results[key].update({'Blue': 'empty'})     # 490
                results[key].update({'Green': 'empty'})    # 560
                results[key].update({'Red': 'empty'})      # 665
                results[key].update({'RedEdge1': 'empty'}) # 705
                results[key].update({'RedEdge2': 'empty'}) # 740
                results[key].update({'RedEdge3': 'empty'}) # 783
                results[key].update({'Nir2': 'empty'})     # 865

            rrs_source = None
            try:
                read_start = time.perf_counter()
                grs_ver = self.grs_file_version
                version_message = f" using grs_version={grs_ver}" if grs_ver else ""
                print(f'Loading {self.ac_processor} data{version_message}...')
                rrs_source = i.get_input_nc(
                    file=str_matches[key]['IMG'], sensor='S2MSI',
                    AC_processor=self.ac_processor, grs_version=grs_ver,
                )
                scene_timing['stages_s']['read'] = time.perf_counter() - read_start
                scene_timing['scene_dimensions'] = [
                    int(value) for value in rrs_source['Red'].shape
                ]

                mask_start = time.perf_counter()
                print(f'Intersecting image with water mask...')
                grs, mask_status = m.intersect_watermask(
                    rrs_dict=rrs_source,
                    water_mask_dir=str_matches[key]['WM'],
                    require_full_coverage=self.mask_mode == 'static',
                    return_status=True,
                )
                scene_timing['stages_s']['mask_reproject'] = (
                    time.perf_counter() - mask_start
                )
                if grs is None:
                    ledger_entry['status'] = mask_status
                    raise ValueError(
                        f'Water mask compatibility failure: {mask_status}.'
                    )

                # Per-band diagnostics are measured after source decoding and water masking, before QC.
                rrs_bands = ('Aerosol', 'Blue', 'Green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3', 'Nir2')
                rrs_diagnostics = self._rrs_diagnostics(grs, rrs_bands)
                ledger_entry['rrs_diagnostics'] = rrs_diagnostics
                results[key]['rrs_diagnostics'] = rrs_diagnostics

                #### Before using the filters, creating a matrix to store the number of pixels
                pixels = np.array([
                ['Water_pixels', '0'],
                ['Neg_Rrs_B4', '0'],
                ['Low_Rrs', '0'],
                ['OWT_1', '0'],])    
                
                # filtering bad pixels
                processing_start = time.perf_counter()
                raster_elapsed = 0.0
                print(f'Filtering bad quality pixels...')
                grs = m.filter_pixels(rrs_dict=grs, neg_rrs='Red', low_rrs=True, low_rrs_thresh=0.002, low_rrs_bands=['Aerosol', 'Blue', 'Green', 'Red', 'RedEdge1', 'RedEdge2'])

                # Clear and get Rrs
                print(f'Obtaining Rrs from Aerosol:443, Blue:490, Green:560, Red:665, RedEdge1:705, RedEdge2:740, RedEdge3:783, and Nir2:865nm...')
                aerosol = m._quick_rrs(rrs_dict=grs, bname='Aerosol')
                blue = m._quick_rrs(rrs_dict=grs, bname='Blue')
                green = m._quick_rrs(rrs_dict=grs, bname='Green')
                red = m._quick_rrs(rrs_dict=grs, bname='Red')
                rededge1 = m._quick_rrs(rrs_dict=grs, bname='RedEdge1')
                rededge2 = m._quick_rrs(rrs_dict=grs, bname='RedEdge2')
                rededge3 = m._quick_rrs(rrs_dict=grs, bname='RedEdge3')
                nir2 = m._quick_rrs(rrs_dict=grs, bname='Nir2')

                # number of pixels
                pixels[0,1] = m.npix
                pixels[1,1] = m.negpix
                pixels[2,1] = m.lowrrs

                # classifying all valid pixels
                print(f'Classifying the OWT of each pixel...')
                class_px, angles = m.classify_owt_chla_px(rrs_dict=grs, sensor='S2MSI', B1=True)
                
                if class_px.sum()<=0:
                    ledger_entry['status'] = 'no_valid_pixel'
                    raise ValueError("No valid pixels remain after masking and quality filters.")
                else:
                    
                    # classifying the OWT of each reservoir
                    print(f'Calculating the OWT weights for each pixel and writing the raster file...')
                    owt_classes, owt_weights = m.classify_owt_chla_weights(class_px=class_px, angles=angles, n=3)

                    # OWT classes for turbidity
                    classes_turb, angles_turb = m.classify_owt_spm_px(rrs_dict=grs, sensor='S2MSI', B1=True)

                    # Different classes for low Rrs pixels
                    # mask for OWT 14
                    bands = ['Aerosol', 'Blue', 'Green', 'Red', 'RedEdge1', 'RedEdge2']
                    stacked = xr.concat([grs[var] for var in bands], dim='variable')
                    max_values = stacked.max(dim='variable')
                    mask = max_values < 0.005
                    # first checking if there are any pixels in the mask
                    if np.where(mask.values)[0].size > 0:
                        owt_classes[0,mask] = 14
                        owt_classes[1,mask] = 0
                        owt_classes[2,mask] = 0
                        owt_weights[0,mask] = 1
                        owt_weights[1,mask] = 0
                        owt_weights[2,mask] = 0

                    # n pixels    
                    pixels[3,1] = np.count_nonzero(owt_classes[0,:,:] == 1)

                    pixels = np.vstack((pixels, np.array([
                        [f'scene_rrs_{band}_{field}', '' if values[field] is None else str(values[field])]
                        for band, values in rrs_diagnostics.items()
                        for field in ('neg_count', 'finite_count')
                    ], dtype=object)))
                    write_start = time.perf_counter()
                    str_output_file = self._output_target(
                        'npix', 'npixels_', key, '.txt'
                    )
                    np.savetxt(str_output_file, pixels, fmt='%s', delimiter=';')
                    results[key]['npix'] = str_output_file

                    str_output_file = self._write_categorical_raster(
                        owt_classes[0, :, :].astype('uint8'),
                        'OWT', 'OWTs_', key,
                        transform=grs.attrs['trans'],
                        projection=grs.attrs['proj'],
                    )
                    results[key]['OWT'] = str_output_file

                    str_output_file = self._write_categorical_raster(
                        classes_turb.astype('uint8'),
                        'OWTSPM', 'OWTSPM_', key,
                        transform=grs.attrs['trans'],
                        projection=grs.attrs['proj'],
                    )
                    results[key]['OWTSPM'] = str_output_file
                    raster_elapsed += time.perf_counter() - write_start

                    # generating the chla product from these classes and weights
                    print(f'Calculating the chla for each dominant OWT and then the blended chla product...')
                    chla = m.blended_chla(rrs_dict=grs, owt_classes=owt_classes, owt_weights=owt_weights, limits=True)

                    # calculating turbidity
                    print(f'Calculating turbidity...')
                    turb = m.turb(rrs_dict=grs, class_owt_spt=classes_turb, alg='owt', limits=True)

                    # calculating SPM_S3
                    print(f'Calculating Hybrid-SPM...')
                    hyspm = m.turb(rrs_dict=grs, class_owt_spt=classes_turb, alg='Hybrid', limits=True)

                    # Preserve explicit post-QC water validity for every derived product.
                    valid_product = np.isfinite(red)
                    chla = np.where(valid_product, chla, np.nan)
                    turb = np.where(valid_product, turb, np.nan)
                    hyspm = np.where(valid_product, hyspm, np.nan)

                    # removing values for OWT1
                    chla[np.where(owt_classes[0,:,:]==1)] = np.nan
                    turb[np.where(owt_classes[0,:,:]==1)] = np.nan
                    hyspm[np.where(owt_classes[0,:,:]==1)] = np.nan
                    
                    scene_timing['valid_water_pixels'] = int(
                        np.count_nonzero(np.isfinite(red))
                    )
                    scene_timing['stages_s']['filtering_owt_inversion'] = max(
                        0.0, time.perf_counter() - processing_start - raster_elapsed
                    )

                    write_start = time.perf_counter()
                    print(f'Parameter report_rrs set to {report_rrs}')
                    if report_rrs:
                        print('Writing Rrs rasters...')
                        rrs_products = {
                            'Aerosol': aerosol, 'Blue': blue, 'Green': green,
                            'Red': red, 'RedEdge1': rededge1,
                            'RedEdge2': rededge2, 'RedEdge3': rededge3,
                            'Nir2': nir2,
                        }
                        for product, values in rrs_products.items():
                            path, scale_metadata = self._write_scaled_raster(
                                values, product, f'{product}_', key,
                                scale=10000, unit='sr-1',
                                transform=grs.attrs['trans'],
                                projection=grs.attrs['proj'],
                            )
                            results[key][product] = path
                            results[key]['scaling'][product] = scale_metadata

                    print('Writing the water-quality rasters...')
                    for product, values, unit in (
                        ('Chla', chla, 'mg m-3'),
                        ('Turb', turb, 'NTU'),
                        ('HySPM', hyspm, 'mg L-1'),
                    ):
                        path, scale_metadata = self._write_scaled_raster(
                            values, product, f'{product}_', key,
                            scale=100, unit=unit,
                            transform=grs.attrs['trans'],
                            projection=grs.attrs['proj'],
                        )
                        results[key][product] = path
                        results[key]['scaling'][product] = scale_metadata
                    scene_timing['stages_s']['raster_write'] = (
                        raster_elapsed + time.perf_counter() - write_start
                    )
               
                stacked = None
                grs.close()
                rrs_source.close()
                rrs_source = None
                results[key]['status'] = 'success'
                ledger_entry['status'] = 'success'
                ledger_entry['reason'] = None
                ledger_entry['outputs'] = {
                    name: value for name, value in results[key].items()
                    if isinstance(value, str) and os.path.exists(value)
                }

                t_hour, t_min, t_sec,_ = u.tac()
                print(f'Done processing: {n+1}/{tot} - {key} \nExecution time: {t_hour}h : {t_min}m : {t_sec}s')
            except Exception as e:
                if rrs_source is not None:
                    rrs_source.close()
                    rrs_source = None
                print(f'Error processing {key}: {e}')
                results[key]['status'] = 'error'
                results[key]['error'] = str(e)
                if ledger_entry['status'] in {'matched', 'processing'}:
                    ledger_entry['status'] = 'processing_error'
                ledger_entry['error'] = str(e)
                t_hour, t_min, t_sec,_ = u.tac()
                print(f'Execution time: {t_hour}h : {t_min}m : {t_sec}s')
            finally:
                scene_timing['stages_s']['total'] = time.perf_counter() - scene_start
                scene_timing['status'] = ledger_entry['status']
                self.timing_manifest['scenes'][key] = scene_timing
                self._write_scene_ledger()
                self._write_timing_manifest()
        
        for stage in (
            'read', 'mask_reproject', 'filtering_owt_inversion',
            'raster_write', 'roi_report',
        ):
            self.timing_manifest['stages_s'][stage] = sum(
                scene['stages_s'][stage]
                for scene in self.timing_manifest['scenes'].values()
            )
        self.timing_manifest['stages_s']['total'] = (
            time.perf_counter() - self._run_started_perf
        )
        self.timing_manifest['ended_utc'] = datetime.now(timezone.utc).isoformat()
        self._write_scene_ledger()
        self._write_timing_manifest()

        res_file_out = os.path.join(imgs_out, self.INSTANCE_TIME_TAG + '.json')
        self._write_json(res_file_out, results)
        return results

    def run_l2b_raw(self):
        raise NotImplementedError('run_l2b_raw() is retired and unsupported; use getpak -c settings.ini run.')
        """
        Run L2B algorithm defined in the settings.ini over all GRS files
        inside the client input folder.
        """
        grs_file_list = self.grs_files
        grs_ver = self.grs_file_version
        for grs_file in grs_file_list:  #TODO: vectorize this loop
            print(f'Processing GRS using version: {grs_ver} file: {grs_file}')
            t_id = u.get_s2_tile_id(grs_file)
        
            print(f'Extracting S2-MSI band data from GRS.nc file...')
            img = grs.get_grs_dict(grs_nc_file=grs_file, grs_version=grs_ver)  #TODO: automate GRS version check
            img_base_name = os.path.basename(grs_file).split('.')[0]
            
            for algo in self.l2b_fx_required_bands.keys():
                print(f'Running {algo}...')
                required_bands = self.l2b_fx_required_bands[algo]  # ex: ['Red', 'Nir2']
                # Build a dictionary mapping each required band to the corresponding image data.
                band_data = {band: img[0][band].values for band in required_bands if band in img[0]}
                # Unpack data and run the L2B algorithm
                l2b_array = ifunc.functions[algo]['function'](**band_data)
                # Define output path/file
                output_tif = os.path.join(self.output_folder, f'{img_base_name}-{algo}.tif')
                print(f'Saving L2B array to {algo}.tif')
                r.s2_to_tiff(
                    ndarray_data=l2b_array,
                    output_img=output_tif,            
                    tile_id=t_id
                )

        print('Done.')
        pass
    
    @staticmethod
    def get_uid(fname):
        name = os.path.basename(fname)
        stem = os.path.splitext(name)[0]
        return stem[len('npixels_'):] if stem.startswith('npixels_') else stem
    
    @staticmethod
    def _search_uid(uid, path):
        result = [os.path.join(path,file) for file in os.listdir(path) if uid in file]
        return result
    
    def match_file_uid(self, out_folders_path, uid):
        products = ['npix', 'OWT', 'OWTSPM', 'Chla', 'Turb', 'HySPM']
        if self.report_rrs:
            products.extend([
                'Aerosol', 'Blue', 'Green', 'Red', 'RedEdge1',
                'RedEdge2', 'RedEdge3', 'Nir2',
            ])

        matched = {}
        missing = []
        ambiguous = []
        for product in products:
            folder = os.path.join(out_folders_path, product)
            if not os.path.isdir(folder):
                matched[product] = None
                missing.append(product)
                continue
            candidates = sorted(
                os.path.join(folder, name)
                for name in os.listdir(folder)
                if uid in name
            )
            if len(candidates) == 1:
                matched[product] = candidates[0]
            elif not candidates:
                matched[product] = None
                missing.append(product)
            else:
                matched[product] = None
                ambiguous.append(product)

        matched['missing_products'] = ';'.join(missing)
        matched['ambiguous_products'] = ';'.join(ambiguous)
        matched['report_file_status'] = (
            'ambiguous_files' if ambiguous
            else ('missing_products' if missing else 'complete')
        )
        return matched

    def line_builder(self):
        output_root = os.path.join(self.output_folder, self.tile_id)
        npix_folder = os.path.join(output_root, 'npix')
        if not os.path.isdir(npix_folder):
            raise ValueError('No npix output folder is available to build a report.')
        uids = sorted(
            self.get_uid(name) for name in os.listdir(npix_folder)
            if name.startswith('npixels_') and name.endswith('.txt')
        )
        if not uids:
            raise ValueError('No processed scenes are available to build a report.')
        return {uid: self.match_file_uid(output_root, uid) for uid in uids}

    @staticmethod
    def build_excel(itermediary_dict, file_to_save):
        df = pd.DataFrame(itermediary_dict).T
        df.drop(columns=['npix', 'OWT', 'OWTSPM', 'Chla', 'Turb', 'HySPM', 'Aerosol', 'Blue', 'Green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3', 'Nir2'], inplace=True, errors='ignore')
        ## ALTERNATIVE: Move coumns to end of DF
        # df = df[[c for c in df if c not in cols_to_move] + cols_to_move]
        df.sort_index(inplace=True)
        df.to_excel(file_to_save)
        pass
    
    # ,---------,
    # | PARSERS |
    # '---------'
    
    @staticmethod
    def _parse_npix(path_to_npix):
        if not path_to_npix or not os.path.isfile(path_to_npix):
            return {'npix_status': 'missing_product'}
        try:
            df = pd.read_csv(path_to_npix, sep=';', header=None).T
            df.columns = df.iloc[0]
            df.drop(0, axis=0, inplace=True)
            values = {key: value[1] for key, value in df.to_dict().items()}
            values['npix_status'] = 'success'
            return values
        except Exception as error:
            return {'npix_status': 'error', 'npix_error': str(error)}
    
    @staticmethod
    def _parse_tifs(path_to_tif, shp_file, prefix='var', scale_factor=100):
        empty = {
            f'{prefix}_{name}': None
            for name in ('min', 'max', 'mean', 'count', 'std', 'median')
        }
        if not path_to_tif or not os.path.isfile(path_to_tif):
            empty[f'{prefix}_status'] = 'missing_product'
            return empty
        try:
            stats = m.shp_stats(tif_file=path_to_tif, shp_poly=shp_file)
        except Exception as error:
            empty[f'{prefix}_status'] = 'error'
            empty[f'{prefix}_error'] = str(error)
            return empty

        def _fix_scale(value, factor=100, digits=6):
            return None if value is None else round(value / factor, digits)

        return {
            f'{prefix}_min': _fix_scale(stats.get('min'), factor=scale_factor),
            f'{prefix}_max': _fix_scale(stats.get('max'), factor=scale_factor),
            f'{prefix}_mean': _fix_scale(stats.get('mean'), factor=scale_factor),
            f'{prefix}_count': stats.get('count', 0),
            f'{prefix}_std': _fix_scale(stats.get('std'), factor=scale_factor),
            f'{prefix}_median': _fix_scale(stats.get('median'), factor=scale_factor),
            f'{prefix}_status': stats.get('roi_status', 'success'),
            f'{prefix}_roi_features': stats.get('roi_features'),
            f'{prefix}_roi_features_with_data': stats.get('roi_features_with_data'),
        }

    def build_report(self):
        report_start = time.perf_counter()
        report_outputs = []
        
        report_rrs = self.report_rrs

        print(f'Building intermediary dictionary with the output folder : {self.output_folder}')
        itermediary_batch_dict = self.line_builder()
        
        for roi_vector in self.roi_vectors:
            print(f'Computing timeseries inside vector: {roi_vector}')
            roi_name = os.path.basename(roi_vector).split('.')[0]

            # ,--------------,
            # | CALL PARSERS |
            # '--------------'        
            
            # One-liners to fetch pixel data inside ROI in a given path of imgs
            if report_rrs:
                print('Fetching Aerosol-443nm data..')
                _ = [itermediary_batch_dict[key].update(self._parse_tifs(itermediary_batch_dict[key]['Aerosol'], roi_vector, prefix='Aerosol', scale_factor=10000)) for key in itermediary_batch_dict.keys()]
                print('Done.')

                print('Fetching Blue-490nm data..')
                _ = [itermediary_batch_dict[key].update(self._parse_tifs(itermediary_batch_dict[key]['Blue'], roi_vector, prefix='Blue', scale_factor=10000)) for key in itermediary_batch_dict.keys()]
                print('Done.')

                print('Fetching Green-560nm data..')
                _ = [itermediary_batch_dict[key].update(self._parse_tifs(itermediary_batch_dict[key]['Green'], roi_vector, prefix='Green', scale_factor=10000)) for key in itermediary_batch_dict.keys()]
                print('Done.')

                print('Fetching Red-665nm data..')
                _ = [itermediary_batch_dict[key].update(self._parse_tifs(itermediary_batch_dict[key]['Red'], roi_vector, prefix='Red', scale_factor=10000)) for key in itermediary_batch_dict.keys()]
                print('Done.')

                print('Fetching RedEdge1-705nm data..')
                _ = [itermediary_batch_dict[key].update(self._parse_tifs(itermediary_batch_dict[key]['RedEdge1'], roi_vector, prefix='RedEdge1', scale_factor=10000)) for key in itermediary_batch_dict.keys()]
                print('Done.')

                print('Fetching RedEdge2-740nm data..')
                _ = [itermediary_batch_dict[key].update(self._parse_tifs(itermediary_batch_dict[key]['RedEdge2'], roi_vector, prefix='RedEdge2', scale_factor=10000)) for key in itermediary_batch_dict.keys()]
                print('Done.')

                print('Fetching RedEdge3-783nm data..')
                _ = [itermediary_batch_dict[key].update(self._parse_tifs(itermediary_batch_dict[key]['RedEdge3'], roi_vector, prefix='RedEdge3', scale_factor=10000)) for key in itermediary_batch_dict.keys()]
                print('Done.')

                print('Fetching Nir2-865nm data..')
                _ = [itermediary_batch_dict[key].update(self._parse_tifs(itermediary_batch_dict[key]['Nir2'], roi_vector, prefix='Nir2', scale_factor=10000)) for key in itermediary_batch_dict.keys()]
                print('Done.')

            print('Fetching SPM L2B data..')
            _ = [itermediary_batch_dict[key].update(self._parse_tifs(itermediary_batch_dict[key]['HySPM'], roi_vector, prefix='HySPM')) for key in itermediary_batch_dict.keys()]
            print('Done.')

            print('Fetching Turbidity L2B data..')
            _ = [itermediary_batch_dict[key].update(self._parse_tifs(itermediary_batch_dict[key]['Turb'], roi_vector, prefix='Turb')) for key in itermediary_batch_dict.keys()]
            print('Done.')

            print('Fetching Chl-a L2B data..')
            _ = [itermediary_batch_dict[key].update(self._parse_tifs(itermediary_batch_dict[key]['Chla'], roi_vector, prefix='Chla')) for key in itermediary_batch_dict.keys()]
            print('Done.')

            print('Fetching L2B pixel metadata..')
            _ = [itermediary_batch_dict[key].update(self._parse_npix(itermediary_batch_dict[key]['npix'])) for key in itermediary_batch_dict.keys()]
            print('Done.')

            print(f'Writing excel file at: {self.output_folder}')
            xlsx_target = os.path.join(self.output_folder, self.tile_id, self.INSTANCE_TIME_TAG + '_' + roi_name + '.xlsx')
            if os.path.exists(xlsx_target) and not self.overwrite_outputs:
                raise FileExistsError(
                    f'Refusing to overwrite existing report: {xlsx_target}'
                )
            self.build_excel(itermediary_batch_dict, file_to_save=xlsx_target)
            pass
            report_outputs.append(xlsx_target)


        report_elapsed = time.perf_counter() - report_start
        self.timing_manifest['stages_s']['roi_report'] += report_elapsed
        self.timing_manifest['stages_s']['total'] = (
            time.perf_counter() - self._run_started_perf
        )
        self.timing_manifest['ended_utc'] = datetime.now(timezone.utc).isoformat()
        self._write_timing_manifest()
        return report_outputs
def main():
    u.tic()
    u.print_logo()
    
    p = Pipelines()
    
    p.run()

    t_hour, t_min, t_sec,_ = u.tac()
    print(f'Done. \nElapsed execution time: {t_hour}h : {t_min}m : {t_sec}s')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
