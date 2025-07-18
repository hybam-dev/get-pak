import os
import ast
import inspect
import numpy as np
import xarray as xr

from pathlib import Path
from datetime import datetime
from getpak import inversion_functions as ifunc
from getpak.input import Input
from getpak.input import GRS as g
from getpak.output import Raster as r
from getpak.commons import Utils as u
from getpak.methods import Methods

grs = g()
i = Input()
m = Methods()

class Pipelines:
    
    def __init__(self):
        # GET-Pak settings
        self.settings = u.read_config()
        self.INSTANCE_TIME_TAG = datetime.now().strftime('%Y%m%dT%H%M%S')
        pass
    

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
    def tile_id(self):
        return self.settings.get('sensors', 's2_tile')['s2_tile']

    # @property
    # def grs_files(self):
    #     grs_file_list = u.walktalk(self.input_folder, unwanted_string='*_anc*')
    #     return grs_file_list
    
    @property
    def grs_file_version(self):
        return self.settings.get('sensors', 'grs_version')['grs_version']

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


    def get_matchups(self):
        """ GRS L2B + WD + OWT """
        
        u.set_gdal_driver_path() # For cluster use
        sep_trace = u.repeat_to_length('-', 22)
        print(sep_trace)
        print('Running L2B algorithms with WD intersection...')
        print(f'Processing tile: {self.tile_id}')
        
        # Set input/output folders for the current tile
        tile_input_folder = os.path.join(self.input_folder, self.tile_id)
        print(f'Input GRS folder: {tile_input_folder}')
        
        grs_file_list = u.walktalk(tile_input_folder, unwanted_string='*_anc*')
        
        # Creating a vector of the dates of GRS images
        grs_dates = []
        meta = {}
        for i in grs_file_list:
            info = g.metadata(i)
            date = info['year']+info['month']+info['day']
            grs_dates.append(date)
            meta[date] = info

        # Location of renamed WaterDetect masks
        wd_dates, wd_masks_list = m.get_waterdetect_masks(input_folder=self.wmask_folder)

        # match-ups
        matches, str_matches, dates = m.sch_date_matchups(
            fst_dates=grs_dates,
            snd_dates=wd_dates,
            fst_tile_list=grs_file_list,
            snd_tile_list=wd_masks_list
        )
        return matches, str_matches, dates, meta
    
    def matchups_to_l2b(self, matches, str_matches, dates, meta):
        """
        TO-DO
        """
        results = {}
        tot = len(matches)
        sep_trace = u.repeat_to_length('-', 22)
        imgs_out = os.path.join(self.output_folder, self.tile_id)
        # Creating output folder structure
        Path(os.path.join(imgs_out, "n_pix")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(imgs_out, "OWT")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(imgs_out, "OWT_SPM")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(imgs_out, "Chla")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(imgs_out, "Turb")).mkdir(parents=True, exist_ok=True)
        
        for n, key in enumerate(matches):
            print(sep_trace)
            print(f'Processing: {n+1}/{tot} - {key}')
            
            results[key] = {'IMG': str_matches[key]['IMG'],
                            'WM': str_matches[key]['WM']}
            
            results[key].update({'n_pix': 'empty'})
            results[key].update({'OWT': 'empty'})
            results[key].update({'OWT_SPM': 'empty'})   
            results[key].update({'Chla': 'empty'})
            results[key].update({'Turb': 'empty'})
            
            try:
                print(f'Loading GRS data...')
                grs_t = i.get_input_nc(file=str_matches[key]['IMG'], sensor='S2MSI', AC_processor='GRS', grs_version='v20')
                
                print(f'Intersecting image with water mask...')
                grs = m.intersect_watermask(rrs_dict=grs_t, water_mask_dir=str_matches[key]['WM'])

                #### Before using the filters, creating a matrix to store the number of pixels
                pixels = np.array([
                ['Water_pixels', '0'],
                ['Neg_Rrs_B4', '0'],
                ['Low_Rrs', '0'],
                ['OWT_1', '0'],])    
                
                # filtering bad pixels
                print(f'Filtering bad quality pixels...')
                grs = m.filter_pixels(rrs_dict=grs, neg_rrs='Red', low_rrs=True, low_rrs_thresh=0.002, low_rrs_bands=['Aerosol', 'Blue', 'Green', 'Red', 'RedEdge1', 'RedEdge2'])

                # number of pixels
                pixels[0,1] = m.npix
                pixels[1,1] = m.negpix
                pixels[2,1] = m.lowrrs

                # classifying all valid pixels
                print(f'Classifying the OWT of each pixel...')
                class_px, angles = m.classify_owt_chla_px(rrs_dict=grs, sensor='S2MSI', B1=True)
                
                if class_px.sum()<=0:
                    print('No valid pixels in this image, continuing to the next.')
                    continue
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
                    pixels[3,1] = len(owt_classes[0,:,:]==1)
                    # writing the file
                    str_output_file = os.path.join(imgs_out, "n_pix/n_pixels_" + key + ".txt")
                    np.savetxt(str_output_file, pixels, fmt='%s', delimiter=';')
                    results[key].update({'n_pix': str_output_file})

                    # writing
                    no_data = 0
                    # Name of the output file: 
                    str_output_file = os.path.join(imgs_out, "OWT/OWTs_" + key + ".tif")
                    # Saving as GeoTIFF
                    r.array2tiff(ndarray_data=owt_classes[0,:,:].astype('uint8'), str_output_file=str_output_file, transform=grs.attrs['trans'], projection=grs.attrs['proj'], no_data=no_data)
                    results[key].update({'OWT': str_output_file})

                    str_output_file = os.path.join(imgs_out, "OWT_SPM/OWT_SPM_" + key + ".tif")
                    r.array2tiff(ndarray_data=classes_turb.astype('uint8'), str_output_file=str_output_file, transform=grs.attrs['trans'], projection=grs.attrs['proj'], no_data=no_data)
                    results[key].update({'OWT_SPM': str_output_file})

                    # generating the chla product from these classes and weights
                    print(f'Calculating the chla for each dominant OWT and then the blended chla product...')
                    chla = m.blended_chla(rrs_dict=grs, owt_classes=owt_classes, owt_weights=owt_weights, limits=True)

                    # calculating turbidity
                    print(f'Calculating turbidity...')
                    turb = m.turb(rrs_dict=grs, class_owt_spt=classes_turb, alg='owt', limits=True)

                    # removing values for OWT1
                    chla[np.where(owt_classes[0,:,:]==1)] = 0
                    turb[np.where(owt_classes[0,:,:]==1)] = 0
                    
                    # writing
                    print(f'Writing the rasters of the water quality parameters...')
                    no_data = 0
                    str_output_file = os.path.join(imgs_out, "Chla/Chla_" + key + ".tif")
                    r.array2tiff(ndarray_data=(chla*100).astype('uint16'), str_output_file=str_output_file, transform=grs.attrs['trans'], projection=grs.attrs['proj'], no_data=no_data)
                    results[key].update({'Chla': str_output_file})

                    str_output_file = os.path.join(imgs_out, "Turb/Turb_" + key + ".tif")
                    r.array2tiff(ndarray_data=(turb*100).astype('uint16'), str_output_file=str_output_file, transform=grs.attrs['trans'], projection=grs.attrs['proj'], no_data=no_data)
                    results[key].update({'Turb': str_output_file})
                
                stacked = None
                grs.close()
                grs_t.close()
            
            except Exception as e:
                print(f'Error processing {key}: {e}')
                continue

        return results

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

if __name__=='__main__':
    st_time = u.tic()

    p = Pipelines()
    
    p.run_l2b_from_settings()
    
    t_hour, t_min, t_sec,_ = u.tac()
    print(f'Done. \nElapsed execution time: {t_hour}h : {t_min}m : {t_sec}s')
    
    pass