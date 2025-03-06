import getpak.inversion_functions as ifunc
from getpak.raster import GRS as g
from getpak.commons import Utils as u

grs = g()

class Pipelines:
    # def __int__(self):
    #     print(f'mock: starting logger.')  # TODO: call logger from utils

    def run_l2b_algo(self, user_args):
        try:
            input      = user_args['input']
            output_tif = user_args['output']
            t_id       = u.get_s2_tile_id(input)
        except KeyError as e:
            print(f'Error: missing argument: {e}')
            return

        print(f'Extracting Sentinel-2 MSI band data from GRS nc file: {input}')
        img = grs.get_grs_dict(grs_nc_file=input, grs_version='v20')

        print(f'Running L2B algorithm for tile ID: {t_id}') # TODO: dynamic call to L2B algorithm
        red, nir2 = img[0]['Red'].values, img[0]['Nir2'].values
        l2b_array = ifunc.spm_severo(Red=red, Nir2=nir2)

        print(f'Saving L2B array to tiff file: {output_tif}')
        grs.param2tiff(
            ndarray_data=l2b_array,
            img_ref=None,
            output_img=output_tif,
            resolve_internal_tile=t_id
        )

        print('Done.')
        pass
