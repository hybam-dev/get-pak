import getpak.inversion_functions as ifunc
from getpak.raster import GRS as GG

grs = GG()

class Pipelines:
    # def __int__(self):
    #     print(f'mock: starting logger.')  # TODO: call logger from utils

    def run_l2b_algo(self, user_args):
        try:
            input      = user_args['input']
            output_tif = user_args['output']
            t_id       = user_args['tileid']
        except KeyError as e:
            print(f'Error: missing argument: {e}')
            return

        print(f'Extracting Sentinel-2 MSI band data from GRS nc file: {input}')
        img = grs.get_grs_dict(grs_nc_file=input, grs_version='v20')

        print(f'Running L2B algorithm for tile ID: {t_id}') # TODO: dynamic call to L2B algorithm
        l2b_array = ifunc.spm_severo(b665=img['Rrs_B4'].values, b865=img['Rrs_B8A'].values)

        print(f'Saving L2B array to tiff file: {output_tif}')
        grs.internal_ref_param2tiff(
            ndarray_data=l2b_array,
            tile_id=t_id,
            output_img=output_tif
        )
        print('Done.')
        pass
