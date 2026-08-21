import os
import sys
import ast
import json
import inspect
import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path
from datetime import datetime
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
        self.INSTANCE_TIME_TAG = datetime.now().strftime('%Y%m%dT%H%M%S')


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
        """Match atmospheric-correction inputs to external WaterDetect masks."""
        
        u.set_gdal_driver_path() # For cluster use
        sep_trace = u.repeat_to_length('-', 22)
        print(sep_trace)
        print(f'Running L2B algorithms for {self.ac_processor} with WD intersection...')
        print(f'Processing tile: {self.tile_id}')
        print(f'Input {self.ac_processor} root: {self.input_folder}')

        records = self.discover_input_files()
        input_files = [record[0] for record in records]
        input_dates = [record[1]['str_date'] for record in records]

        # Location of renamed WaterDetect masks
        wd_dates, wd_masks_list = m.get_waterdetect_masks(input_folder=self.wmask_folder)

        # match-ups
        matches, str_matches, dates = m.sch_date_matchups(
            fst_dates=input_dates,
            snd_dates=wd_dates,
            fst_tile_list=input_files,
            snd_tile_list=wd_masks_list,
            tile_id=self.tile_id,
        )

        if not matches:
            raise ValueError(
                f"No {self.ac_processor}/WaterDetect matchups found for tile "
                f"{self._normalize_tile(self.tile_id)}."
            )

        meta = {}
        occurrences = {}
        for _, info in records:
            date = info['str_date']
            occurrences[date] = occurrences.get(date, 0) + 1
            key = date if occurrences[date] == 1 else f"{date}_{occurrences[date]}"
            if key in matches:
                meta[key] = info

        self.matches = matches
        self.str_matches = str_matches
        self.dates = dates
        self.meta = meta
        
        print('get_matchups: Done.')

        if do_return:
            return matches, str_matches, dates, meta
        pass
    
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
            
            results[key] = {
                'IMG': str_matches[key]['IMG'],
                'WM': str_matches[key]['WM'],
                'processor': self.ac_processor,
                'source_path': str_matches[key]['IMG'],
                'status': 'processing',
                'error': None,
            }
            
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
                grs_ver = self.grs_file_version
                version_message = f" using grs_version={grs_ver}" if grs_ver else ""
                print(f'Loading {self.ac_processor} data{version_message}...')
                rrs_source = i.get_input_nc(
                    file=str_matches[key]['IMG'], sensor='S2MSI',
                    AC_processor=self.ac_processor, grs_version=grs_ver,
                )

                print(f'Intersecting image with water mask...')
                grs = m.intersect_watermask(rrs_dict=rrs_source, water_mask_dir=str_matches[key]['WM'])
                if grs is None:
                    raise ValueError("Water mask produced no valid overlapping water pixels.")

                #### Before using the filters, creating a matrix to store the number of pixels
                pixels = np.array([
                ['Water_pixels', '0'],
                ['Neg_Rrs_B4', '0'],
                ['Low_Rrs', '0'],
                ['OWT_1', '0'],])    
                
                # filtering bad pixels
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

                    # writing the file
                    str_output_file = os.path.join(imgs_out, "npix/npixels_" + key + ".txt")
                    np.savetxt(str_output_file, pixels, fmt='%s', delimiter=';')
                    results[key].update({'npix': str_output_file})

                    # writing
                    no_data = 0
                    # Name of the output file: 
                    str_output_file = os.path.join(imgs_out, "OWT/OWTs_" + key + ".tif")
                    # Saving as GeoTIFF
                    r.array2tiff(ndarray_data=owt_classes[0,:,:].astype('uint8'), str_output_file=str_output_file, transform=grs.attrs['trans'], projection=grs.attrs['proj'], no_data=no_data)
                    results[key].update({'OWT': str_output_file})

                    str_output_file = os.path.join(imgs_out, "OWTSPM/OWTSPM_" + key + ".tif")
                    r.array2tiff(ndarray_data=classes_turb.astype('uint8'), str_output_file=str_output_file, transform=grs.attrs['trans'], projection=grs.attrs['proj'], no_data=no_data)
                    results[key].update({'OWTSPM': str_output_file})

                    # generating the chla product from these classes and weights
                    print(f'Calculating the chla for each dominant OWT and then the blended chla product...')
                    chla = m.blended_chla(rrs_dict=grs, owt_classes=owt_classes, owt_weights=owt_weights, limits=True)

                    # calculating turbidity
                    print(f'Calculating turbidity...')
                    turb = m.turb(rrs_dict=grs, class_owt_spt=classes_turb, alg='owt', limits=True)

                    # calculating SPM_S3
                    print(f'Calculating Hybrid-SPM...')
                    hyspm = m.turb(rrs_dict=grs, class_owt_spt=classes_turb, alg='Hybrid', limits=True)

                    # removing values for OWT1
                    chla[np.where(owt_classes[0,:,:]==1)] = 0
                    turb[np.where(owt_classes[0,:,:]==1)] = 0
                    hyspm[np.where(owt_classes[0,:,:]==1)] = 0
                    
                    # writing Rrs bands
                    print(f'Parameter report_rrs set to {report_rrs}')
                    if report_rrs:
                        print(f'Writing Rrs rasters...')
                        str_output_file = os.path.join(imgs_out, "Aerosol/Aerosol_" + key + ".tif")
                        r.array2tiff(ndarray_data=(aerosol*10000).astype('uint16'), str_output_file=str_output_file, transform=grs.attrs['trans'], projection=grs.attrs['proj'], no_data=no_data)
                        results[key].update({'Aerosol': str_output_file})

                        str_output_file = os.path.join(imgs_out, "Blue/Blue_" + key + ".tif")
                        r.array2tiff(ndarray_data=(blue*10000).astype('uint16'), str_output_file=str_output_file, transform=grs.attrs['trans'], projection=grs.attrs['proj'], no_data=no_data)
                        results[key].update({'Blue': str_output_file})

                        str_output_file = os.path.join(imgs_out, "Green/Green_" + key + ".tif")
                        r.array2tiff(ndarray_data=(green*10000).astype('uint16'), str_output_file=str_output_file, transform=grs.attrs['trans'], projection=grs.attrs['proj'], no_data=no_data)
                        results[key].update({'Green': str_output_file})

                        str_output_file = os.path.join(imgs_out, "Red/Red_" + key + ".tif")
                        r.array2tiff(ndarray_data=(red*10000).astype('uint16'), str_output_file=str_output_file, transform=grs.attrs['trans'], projection=grs.attrs['proj'], no_data=no_data)
                        results[key].update({'Red': str_output_file})
                        
                        str_output_file = os.path.join(imgs_out, "RedEdge1/RedEdge1_" + key + ".tif")
                        r.array2tiff(ndarray_data=(rededge1*10000).astype('uint16'), str_output_file=str_output_file, transform=grs.attrs['trans'], projection=grs.attrs['proj'], no_data=no_data)
                        results[key].update({'RedEdge1': str_output_file})

                        str_output_file = os.path.join(imgs_out, "RedEdge2/RedEdge2_" + key + ".tif")
                        r.array2tiff(ndarray_data=(rededge2*10000).astype('uint16'), str_output_file=str_output_file, transform=grs.attrs['trans'], projection=grs.attrs['proj'], no_data=no_data)
                        results[key].update({'RedEdge2': str_output_file})

                        str_output_file = os.path.join(imgs_out, "RedEdge3/RedEdge3_" + key + ".tif")
                        r.array2tiff(ndarray_data=(rededge3*10000).astype('uint16'), str_output_file=str_output_file, transform=grs.attrs['trans'], projection=grs.attrs['proj'], no_data=no_data)
                        results[key].update({'RedEdge3': str_output_file})

                        str_output_file = os.path.join(imgs_out, "Nir2/Nir2_" + key + ".tif")
                        r.array2tiff(ndarray_data=(nir2*10000).astype('uint16'), str_output_file=str_output_file, transform=grs.attrs['trans'], projection=grs.attrs['proj'], no_data=no_data)
                        results[key].update({'Nir2': str_output_file})

                    print(f'Writing the rasters of the water quality parameters...')
                    no_data = 0
                    str_output_file = os.path.join(imgs_out, "Chla/Chla_" + key + ".tif")
                    r.array2tiff(ndarray_data=(chla*100).astype('uint16'), str_output_file=str_output_file, transform=grs.attrs['trans'], projection=grs.attrs['proj'], no_data=no_data)
                    results[key].update({'Chla': str_output_file})

                    str_output_file = os.path.join(imgs_out, "Turb/Turb_" + key + ".tif")
                    r.array2tiff(ndarray_data=(turb*100).astype('uint16'), str_output_file=str_output_file, transform=grs.attrs['trans'], projection=grs.attrs['proj'], no_data=no_data)
                    results[key].update({'Turb': str_output_file})

                    str_output_file = os.path.join(imgs_out, "HySPM/HySPM_" + key + ".tif")
                    r.array2tiff(ndarray_data=(hyspm*100).astype('uint16'), str_output_file=str_output_file, transform=grs.attrs['trans'], projection=grs.attrs['proj'], no_data=no_data)
                    results[key].update({'HySPM': str_output_file})
               
                stacked = None
                grs.close()
                rrs_source.close()
                results[key]['status'] = 'success'
            
                t_hour, t_min, t_sec,_ = u.tac()
                print(f'Done processing: {n+1}/{tot} - {key} \nExecution time: {t_hour}h : {t_min}m : {t_sec}s')
                

            except Exception as e:
                if rrs_source is not None:
                    rrs_source.close()
                print(f'Error processing {key}: {e}')
                results[key]['status'] = 'error'
                results[key]['error'] = str(e)
                t_hour, t_min, t_sec,_ = u.tac()
                print(f'Execution time: {t_hour}h : {t_min}m : {t_sec}s')
                continue
        
        # Saving metadata json file with resulting file paths
        res_file_out = os.path.join(imgs_out, self.INSTANCE_TIME_TAG + ".json")
        with open(res_file_out, 'w') as f:
            json.dump(results, f)
        pass

    def run_l2b_raw(self):
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
        fragments = fname.split('_')
        if len(fragments) < 3:
            uid = fragments[1].split('.')[0] + '.'
        elif len(fragments) > 2:
            uid = fragments[1] + '_' + fragments[2].split('.')[0] + '.'
        return uid
    
    @staticmethod
    def _search_uid(uid, path):
        result = [os.path.join(path,file) for file in os.listdir(path) if uid in file]
        return result
    
    @staticmethod
    def match_file_uid(out_folders_path, uid):

        # Build dict of paths to each parameter
        params = {keys : os.path.join(out_folders_path , keys) for keys in os.listdir(out_folders_path) if os.path.isdir(os.path.join(out_folders_path, keys))}

        # Internal function, search uid presence in file name for a given path and return it.
        def _search_uid(uid, fpath):
            result = [os.path.join(fpath,file) for file in os.listdir(fpath) if uid in file]
            if len(result) > 1:
                print('Inconsistent matchup > 1.')
            else:
                # pop the element out of the list.
                result = result[0]
            return result

        # call the search function for each uid and L2B parameter
        match_results = {par : _search_uid(uid, mpath) for par, mpath in params.items()}

        return match_results

    def line_builder(self):
        # Get all UIDs from npix in the output folder
        uids_list = [self.get_uid(f) for f in os.listdir(os.path.join(self.output_folder, self.tile_id, 'npix'))]
        l_size = len(uids_list)
        if l_size >= 1:
            sheet = { uid.split('.')[0] : self.match_file_uid(os.path.join(self.output_folder, self.tile_id), uid) for uid in uids_list }
        else:
            raise ValueError('No processed scenes are available to build a report.')
        
        # # Clear the trailing dot at the end of each UID
        # uids_list = [uid.split('.')[0] for uid in uids_list]
        return sheet

    @staticmethod
    def build_excel(itermediary_dict, file_to_save):
        df = pd.DataFrame(itermediary_dict).T
        df.drop(columns=['npix', 'OWT', 'OWTSPM', 'Chla', 'Turb', 'HySPM', 'Aerosol', 'Blue', 'Green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3', 'Nir2'], inplace=True)
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
        df = pd.read_csv(path_to_npix, sep=";", header=None).T
        df.columns = df.iloc[0]  # get column names from the first line
        df.drop(0, axis=0, inplace=True)  # drop the first row 
        dict_df = df.to_dict()
        dict_df = {key:val[1] for key,val in dict_df.items()}
        return dict_df
    
    @staticmethod
    def _parse_tifs(path_to_tif, shp_file, prefix='var', scale_factor=100):
        stats = m.shp_stats(tif_file=path_to_tif, shp_poly=shp_file)

        def _fix_scale(value,factor=100,digits=6):
            if value is not None:
                value = round(value/factor,digits)
            return value
            
        results = {
            prefix + '_min' : _fix_scale(stats['min'], factor=scale_factor),
            prefix + '_max' : _fix_scale(stats['max'], factor=scale_factor),
            prefix + '_mean' : _fix_scale(stats['mean'], factor=scale_factor),
            prefix + '_count' : stats['count'],
            prefix + '_std' : _fix_scale(stats['std'], factor=scale_factor),
            prefix + '_median' : _fix_scale(stats['median'], factor=scale_factor),
        }
        return results

    def build_report(self):
        
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
            self.build_excel(itermediary_batch_dict, file_to_save=xlsx_target)
            pass

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
