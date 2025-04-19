import os
import ast
import inspect

from getpak import inversion_functions as ifunc
from getpak.input import GRS as g
from getpak.output import Raster as r
from getpak.commons import Utils as u

grs = g()

class Pipelines:
    
    def __init__(self):
        # GET-Pak settings
        self.settings = u.read_config()
        pass
    
    @property
    def output_folder(self):
        return self.settings.get('client_folder', 'output')['output']

    @property
    def grs_files(self):
        grs_file_list = u.walktalk(self.settings.get('client_folder', 'inputs')['inputs'])
        return grs_file_list
    
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


    def run_l2b_from_settings(self):
        """
        Run L2B algorithm defined in the settings.ini over all GRS files
        inside the client input folder.
        """
        grs_file_list = self.grs_files
        for grs_file in grs_file_list:  #TODO: vectorize this loop
            print(f'Processing GRS file: {grs_file}')
            t_id = u.get_s2_tile_id(grs_file)
        
            print(f'Extracting S2-MSI band data from GRS.nc file...')
            img = grs.get_grs_dict(grs_nc_file=grs_file, grs_version='v20')  #TODO: automate GRS version check
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