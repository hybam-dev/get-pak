import os
import sys
import time
import json
import logging
import zipfile
import subprocess
import numpy as np
import fnmatch

from PIL import Image
from pathlib import Path
from configparser import ConfigParser
from osgeo import gdal, ogr, osr
from getpak import s2projgrid


try:
    from osgeo import gdal, ogr, osr
except:
    print("Unable to import osgeo! GETpak can still operate but critical functions may fail.")


class Utils:

    def __init__(self):
        pass

    @staticmethod
    def print_logo():
        print('''
            _..._
          .'     '.      _
         /    .-""-\   _/ \ 
       .-|   /:.   |  |   | 
       |  \  |:.   /.-'-./ 
       | .-'-;:__.'    =/  ,ad8888ba,  88888888888 888888888888                          88
       .'=  *=|CNES _.='  d8"'    `"8b 88               88                               88
      /   _.  |    ;     d8'           88               88                               88
     ;-.-'|    \   |     88            88aaaaa          88        8b,dPPYba,  ,adPPYYba, 88   ,d8
    /   | \    _\  _\    88      88888 88"""""          88 aaaaaa 88P'    "8a ""     `Y8 88 ,a8"
    \__/'._;.  ==' ==\   Y8,        88 88               88 """""" 88       d8 ,adPPPPP88 8888[
    /|\  /|\ \    \   |   Y8a.    .a88 88               88        88b,   ,a8" 88,    ,88 88`"Yba,
   / | \/ | \/    /   /    `"Y88888P"  88888888888      88        88`YbbdP"'  `"8bbdP"Y8 88   `Y8a
  /  | || |  /-._/-._/                                            88
             \   `\  \                                            88
              `-._/._/
                        ''')

        pass
    
    @staticmethod
    def read_config():
        config = ConfigParser()
        # get the path to config.ini in the root folder
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'settings.ini')
        print(f'Atempting to read config file: {config_path}')
        # check if the path points to a valid file
        if not os.path.isfile(config_path):
            print(f'Error: config file not found. Exiting...')
            sys.exit(1)

        # read the config file
        config.read(config_path)

        config_dict = {}
        for section in config.sections():
            # Convert section to a dictionary
            config_dict[section] = dict(config[section])
        print('Done.')
        return config_dict

    @staticmethod
    def parse_config_option(section, key):
        config = Utils.read_config()
        try:
            return config[section][key]
        except KeyError as e:
            print(f'Error: missing argument: {e}')
            return

    @staticmethod
    def sort_l2b_by_date(input_directory_path):
        """
        Creates a python list containing the Posixpath from all the files inside the directory and sort them by date.
        """
        # convert input string to Posix
        in_path_obj = Path(input_directory_path)
        # get only the '20160425T134227' from the file name and use it to sort the list by date
        sorted_output_files = sorted(os.listdir(in_path_obj), key=lambda s: s.split('_')[2])
        sorted_output_files_fullpath = [os.path.join(in_path_obj, img) for img in sorted_output_files]

        return sorted_output_files_fullpath

    @staticmethod
    def create_log_handler(fname):
        # https://stackoverflow.com/questions/62835466/create-a-separate-logger-for-each-process-when-using-concurrent-futures-processp
        logger = logging.getLogger(name=fname)
        logger.setLevel(logging.INFO)

        fileHandler = logging.FileHandler(fname)
        fileHandler.setLevel(logging.INFO)

        logger.addHandler(fileHandler)
        logger.addHandler(logging.StreamHandler())

        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

        fileHandler.setFormatter(formatter)

        return logger

    def get_tile_s2_projection(self, tile_id):
        """
        return: A dictionary with proj data for tile_id
        usage: Type the tile_id without the leading 'T'
        example: tile_id=12ABC and not T12ABC
        """
        return s2projgrid[tile_id]

    @staticmethod
    def tic():
        global _start_time
        _start_time = time.time()
        return _start_time

    @staticmethod
    def tac():
        t_raw = time.time()
        t_sec = round(t_raw - _start_time)
        (t_min, t_sec) = divmod(t_sec, 60)
        (t_hour, t_min) = divmod(t_min, 60)
        return t_hour, t_min, t_sec, t_raw

    @staticmethod
    def get_s2_tile_id(img_full_path):
        """
        Given the complete path to a GRS images.nc, return the tile ID
        """
        if os.path.isfile(img_full_path):
            img_parent_name = os.path.basename(Path(img_full_path))
            sliced_ipn = img_parent_name.split('_')
            tile_id = sliced_ipn[5][1:]
            return tile_id
        else:
            print(f'File not found: {img_full_path}')
            print(f'Returning None..')
            return None

    @staticmethod
    def s2proj_ref_builder(img_path_str, renamed_mask=False, skip_id=False):
        """
        Given a WaterDetect output .tif image
        return GDAL information metadata

        Parameters
        ----------
        @param img_path_str: path to the image file

        @return: tile_id (str) and ref (dict) containing the GDAL information
        """
        if renamed_mask:
            img_name = os.path.basename(img_path_str)
        else:
            img_name = os.path.basename(Path(img_path_str).parents[1])
        if not skip_id:
            sliced_ipn = img_name.split('_')
            tile_id = sliced_ipn[5][1:]
        else:
            tile_id = None    
        # Get GDAL information from the template file
        ref_data = gdal.Open(img_path_str)
        mtx = ref_data.ReadAsArray()
        trans = ref_data.GetGeoTransform()
        proj = ref_data.GetProjection()
        rows, cols = mtx.shape  # return Y / X
        ref = {
            'trans': trans,
            'proj': proj,
            'rows': rows,
            'cols': cols
        }
        # close GDAL image
        del ref_data
        return tile_id, ref

    @staticmethod
    def repeat_to_length(s, wanted):
        return (s * (wanted // len(s) + 1))[:wanted]

    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    @staticmethod
    def walktalk(path2walk, fpattern='.nc', dir_is_file=False, unwanted_string=None):
        '''

        Parameters
        ----------
        path2walk: folder to scan for files
        fpattern: the file pattern to look for inside the folders
        dir_is_file: option to include the folders in the search, defaults to False
        unwanted_string: a string to search for and remove from the list of files if present ex: '*_anc*' to remove grs auxiliary NetCDFs

        Returns
        -------
        output_flist: list of files (and folders) matching the fpattern found inside the path2walk folder
        '''
        output_flist = []
        for root, dirs, files in os.walk(path2walk, topdown=False):
            for name in (dirs if dir_is_file else files):
                # check the file format
                if name.endswith(fpattern):
                    if isinstance(unwanted_string, str):
                    # check if it is not the file you want
                        if not fnmatch.fnmatch(name, unwanted_string):
                            f = Path(os.path.join(root, name))
                            output_flist.append(f)
                    else:
                        f = Path(os.path.join(root, name))
                        output_flist.append(f)
        print(f'{len(output_flist)} files found.')
        return output_flist

    # https://stackoverflow.com/questions/6039103/counting-depth-or-the-deepest-level-a-nested-list-goes-to
    @staticmethod
    def depth(somelist): return isinstance(somelist, list) and max(map(Utils.depth, somelist)) + 1

    def get_available_cores(self):
        cpu_count = os.cpu_count() - 1
        if cpu_count <= 0:
            self.log.info(f'Invalid number of CPU cores available: {os.cpu_count()}.')
            sys.exit(1)
        elif cpu_count > 61:
            cpu_count = 61

        return cpu_count

    @staticmethod
    def pil_grid(images, max_horiz=np.iinfo(int).max):
        """
        Combine several images
        https://stackoverflow.com/a/46877433/2238624
        """
        n_images = len(images)
        n_horiz = min(n_images, max_horiz)
        #     h_sizes, v_sizes = [0] * n_horiz, [0] * (n_images // n_horiz)
        h_sizes, v_sizes = [0] * n_horiz, [0] * ((n_images // n_horiz) + (1 if n_images % n_horiz > 0 else 0))
        for i, im in enumerate(images):
            h, v = i % n_horiz, i // n_horiz
            h_sizes[h] = max(h_sizes[h], im.size[0])
            v_sizes[v] = max(v_sizes[v], im.size[1])
        h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
        im_grid = Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')
        for i, im in enumerate(images):
            im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
        return im_grid

    @staticmethod
    def geojson_to_polygon(geojson_path):
        """
        Transform a given input .geojson file into a list of coordinates
        poly_path: string (Path to .geojson file)
        return: dict (Containing one or several polygons depending on the complexity of the input .json)
        """
        # TODO: use isinstance(s, str) to avoid breaking when pathlib.Path is passed.
        with open(geojson_path) as f:
            data = json.load(f)

        inDataSource = ogr.GetDriverByName("GeoJSON").Open(geojson_path, 0)
        inLayer = inDataSource.GetLayer()

        vertices = []
        for layer in inDataSource:
            # this is the one where featureindex may not start at 0
            layer.ResetReading()
            for feature in layer:
                geometry = feature.geometry()
                json_geometry = json.loads(geometry.ExportToJson())
                poly = json_geometry['coordinates'][0]
                while Utils.depth(poly) > 2:
                    poly = poly[0]
                poly = [p[0:2] for p in poly]
                poly = np.array(poly)
                vertices.append(poly)

        return vertices

    @staticmethod
    def shp2geojson_geopandas(shp_file_path, json_out_path):
        import geopandas
        shpfile = geopandas.read_file(shp_file_path)
        f_name = os.path.basename(shp_file_path).split('.')[0]
        shpfile.to_file(os.path.join(json_out_path, f_name + '.geojson'), driver='GeoJSON')

    @staticmethod
    def shp2json_pyshp(shp_file_path):
        import shapefile
        with shapefile.Reader(shp_file_path) as shp:
            geojson_data = shp.__geo_interface__
        return geojson_data

    @staticmethod
    def shp2json_gdal(shp_file_path):

        pass

    @staticmethod
    # KML -> GEOJson : https://github.com/mrcagney/kml2geojson
    def kml2json_gdal(input_kml_path, output_json_path=None):
        """
        Converts a given google earth engine KML file to SHP.
        """
        if output_json_path:
            filename = os.path.basename(input_kml_path).split('.')[0] + '.geojson'
            output_json = os.path.join(output_json_path, filename)
        else:
            output_json = input_kml_path.split('.')[0] + '.geojson'

        logging.info('Opening subprocess call to GDAL:ogr2ogr...')
        subprocess.call(f'ogr2ogr -f "GeoJSON" {output_json} {input_kml_path}', shell=True)
        logging.info(f'{output_json} created.')

        return output_json

    @staticmethod
    # KML -> GEOJson : https://github.com/mrcagney/kml2geojson
    def kmz2kml_unzip(input_kmz_path, output_kml_path=None):
        """
        Converts a given google earth engine KMZ file to KML by extracting its content.
        """
        fname = os.path.basename(input_kmz_path).split('.')[0]
        if output_kml_path:
            filename = fname + '.geojson'
            output_kml = os.path.join(output_kml_path, filename)
        else:
            output_json = input_kmz_path.split('.')[0] + '.kml'

        logging.info('Extracting KMZ content...')
        with zipfile.ZipFile(input_kmz_path, 'r') as zip_ref:
            zip_ref.extractall(output_kml_path)
        # Search for everything ended in .kml inside the extracted files folder (expects 1) and remove it from the list.
        kml_file = [os.path.join(output_kml_path, f) for f in os.listdir(output_kml_path) if f.endswith('.kml')].pop()
        # Create a new name for the output file.
        kml_result = os.path.join(output_kml_path, fname + '.kml')
        # Rename whatever came out of the KMZ.zip to fname.kml
        os.rename(kml_file, kml_result)
        logging.info(f'{kml_result} created.')
        return kml_result

    @staticmethod
    def roi2vertex(roi, aux_folder_out=os.getcwd()):
        """
        Test the format of the input vector file and return a list of vertices.

        :param roi: (str) path to input vector file to be tested and processed.
        :param aux_folder_out: (str) path to save ancillary data.
        :return python list of arrays: (list) containing one array for each vector feature.
        """

        def parse_kml(path2vector):
            logging.info('KML file detected. Attempting to parse...')
            vtx_path = Utils.kml2json_gdal(input_kml_path=path2vector, output_json_path=aux_folder_out)
            return Utils.geojson_to_polygon(vtx_path)

        def parse_kmz(path2vector):
            logging.info('KMZ file detected. Attempting to parse...')
            # KMZ -> KML
            vtx_path = Utils.kmz2kml_unzip(input_kmz_path=path2vector, output_kml_path=aux_folder_out)
            # KML -> JSON
            vtx = Utils.kml2json_gdal(input_kml_path=vtx_path, output_json_path=aux_folder_out)
            # JSON -> VTX
            return Utils.geojson_to_polygon(vtx)

        def parse_json(path2vector):
            logging.info('JSON file detected. Attempting to parse...')
            return Utils.geojson_to_polygon(path2vector)

        def parse_shp(path2vector):
            logging.info('SHP file detected. Attempting to parse...')
            # Convert SHP -> JSON
            vtx = Utils.shp2json_pyshp(path2vector)
            # Convert JSON -> vertex
            # return Utils.geojson_to_polygon(vtx, read_file=False)
            return Utils.shp2geojson_geopandas(vtx)

        roi_typename = os.path.basename(roi).split('.')[1]

        roi_options = {'kml': parse_kml,
                       'kmz': parse_kmz,
                       'shp': parse_shp,
                       'json': parse_json,
                       'geojson': parse_json}

        if roi_typename in roi_options:
            vertex = roi_options[roi_typename](roi)
        else:
            logging.info(f'Input ROI {os.path.basename(roi)} not recognized as a valid vector file. '
                         f'Make sure the input file is of type .shp .kml .kmz .json or .geojson')
            sys.exit(1)

        return vertex

    @staticmethod
    def get_x_y_poly(lat_arr, lon_arr, polyline):
        grid = np.concatenate([lat_arr[..., None], lon_arr[..., None]], axis=2)

        # Polyline is a GeoJSON coordinate array
        polyline = polyline.squeeze()

        # loop through each vertice
        vertices = []
        for i in range(polyline.shape[0]):
            vector = np.array([polyline[i, 1], polyline[i, 0]]).reshape(1, 1, -1)
            subtraction = vector - grid
            dist = np.linalg.norm(subtraction, axis=2)
            result = np.where(dist == dist.min())
            target_x_y = [result[0][0], result[1][0]]

            vertices.append(target_x_y)
        return np.array(vertices)

    @staticmethod
    def bbox(vertices):
        """
        Get the bounding box of the vertices. Just for visualization purposes
        """
        vertices = np.vstack(vertices)
        ymin = np.min(vertices[:, 0])
        ymax = np.max(vertices[:, 0])
        xmin = np.min(vertices[:, 1])
        xmax = np.max(vertices[:, 1])
        return xmin, xmax, ymin, ymax

    @staticmethod
    def nan2zero(nd_mtx):
        """
        Transforms numpy NaNs into zeros
        """
        where_are_NaNs = np.isnan(nd_mtx)
        nd_mtx[where_are_NaNs] = -1
        return nd_mtx

    @staticmethod
    def intersect_matrices(mtx_a, mtx_b):
        """
        Intersection between two matrices, where the intersection remains and other values are replaced by 0
        """
        if not (mtx_a.shape == mtx_b.shape):
            return False

        mtx_intersect = np.where((mtx_a == mtx_b), mtx_a, 0)
        return mtx_intersect

    @staticmethod
    def find_outliers_IQR(values):
        """
        Function to remove outliers, using the same methods as in boxplot (interquartile distance)

        Parameters
        ----------
        values: numpy array with data

        Returns
        -------
        outliers: a numpy array of the removed outliers
        clean_array: a numpy array, with the same size, as values, without the outliers
        """
        aux = values[np.where(np.isnan(values) == False)]
        clean_array = values
        q1 = np.quantile(aux, 0.25)
        q3 = np.quantile(aux, 0.75)
        iqr = q3 - q1
        # finding upper and lower whiskers
        upper_bound = q3 + (1.5 * iqr)
        lower_bound = q1 - (1.5 * iqr)
        # array of outliers and decontaminated array
        outliers = aux[(aux <= lower_bound) | (aux >= upper_bound)]
        clean_array[(clean_array <= lower_bound) | (clean_array >= upper_bound)] = np.nan

        return outliers, clean_array


class DefaultDicts:

    grs_v20nc_s2bands = {'Aerosol': 443,    # B1
                        'Blue': 490,        # B2
                        'Green': 560,       # B3
                        'Red': 665,         # B4
                        'RedEdge1': 705,    # B5
                        'RedEdge2': 740,    # B6
                        'RedEdge3': 783,    # B7
                        'Nir1': 842,        # B8
                        'Nir2': 865,        # B8a
                        'Swir1': 1610,      # B11
                        'Swir2': 2190}      # B12

    acolite_nc_s2abands = {'Aerosol': "443",
                         'Blue': "492",
                         'Green': "560",
                         'Red': "665",
                         'RedEdge1': "704",
                         'RedEdge2': "740",
                         'RedEdge3': "783",
                         'Nir1': "833",
                         'Nir2': "865",
                         'Swir1': "1614",
                         'Swir2': "2202"}

    acolite_nc_s2bbands = {'Aerosol': "442",
                         'Blue': "492",
                         'Green': "559",
                         'Red': "665",
                         'RedEdge1': "704",
                         'RedEdge2': "739",
                         'RedEdge3': "780",
                         'Nir1': "833",
                         'Nir2': "864",
                         'Swir1': "1610",
                         'Swir2': "2186"}

    seadas_nc_s2abands = {'Aerosol': "443",
                         'Blue': "492",
                         'Green': "560",
                         'Red': "665",
                         'RedEdge1': "704",
                         'RedEdge2': "740",
                         'RedEdge3': "783",
                         'Nir1': "835",
                         'Nir2': "865",
                         'Swir1': "1613",
                         'Swir2': "2200"}

    seadas_nc_s2bbands = {'Aerosol': "442",
                         'Blue': "492",
                         'Green': "559",
                         'Red': "665",
                         'RedEdge1': "704",
                         'RedEdge2': "739",
                         'RedEdge3': "780",
                         'Nir1': "835",
                         'Nir2': "864",
                         'Swir1': "1611",
                         'Swir2': "2184"}

    clustering_methods = {'M0': ['Oa17_reflectance:float', 'Oa21_reflectance:float'],
                          'M1': ['Oa17_reflectance:float', 'T865:float', 'A865:float'],
                          'M2': ['Oa21_reflectance:float', 'T865:float', 'A865:float'],
                          'M3': ['Oa17_reflectance:float', 'Oa21_reflectance:float', 'T865:float'],
                          'M4': ['Oa08_reflectance:float', 'Oa17_reflectance:float', 'Oa21_reflectance:float'],
                          'M5': ['Oa04_reflectance:float', 'Oa08_reflectance:float', 'Oa21_reflectance:float']}

    wfr_norm_s3_bands = ['Oa01_reflectance:float',
                         'Oa02_reflectance:float',
                         'Oa03_reflectance:float',
                         'Oa04_reflectance:float',
                         'Oa05_reflectance:float',
                         'Oa06_reflectance:float',
                         'Oa07_reflectance:float',
                         'Oa08_reflectance:float',
                         'Oa09_reflectance:float',
                         'Oa10_reflectance:float',
                         'Oa11_reflectance:float',
                         'Oa12_reflectance:float',
                         'Oa16_reflectance:float',
                         'Oa17_reflectance:float',
                         'Oa18_reflectance:float',
                         'Oa21_reflectance:float']

    wfr_l2_bnames = {'B1-400': 'Oa01: 400 nm',
                     'B2-412.5': 'Oa02: 412.5 nm',
                     'B3-442.5': 'Oa03: 442.5 nm',
                     'B4-490': 'Oa04: 490 nm',
                     'B5-510': 'Oa05: 510 nm',
                     'B6-560': 'Oa06: 560 nm',
                     'B7-620': 'Oa07: 620 nm',
                     'B8-665': 'Oa08: 665 nm',
                     'B9-673.75': 'Oa09: 673.75 nm',
                     'B10-681.25': 'Oa10: 681.25 nm',
                     'B11-708.75': 'Oa11: 708.75 nm',
                     'B12-753.75': 'Oa12: 753.75 nm',
                     'B16-778.75': 'Oa16: 778.75 nm',
                     'B17-865': 'Oa17: 865 nm',
                     'B18-885': 'Oa18: 885 nm',
                     'B21-1020': 'Oa21: 1020 nm'}

    wfr_int_flags = [1, 2, 4, 8, 8388608, 16777216, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
                     32768, 65536, 131072, 262144, 524288, 2097152, 33554432, 67108864, 134217728, 268435456,
                     4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944,
                     549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208, 17592186044416,
                     35184372088832, 70368744177664, 140737488355328, 281474976710656, 562949953421312,
                     1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992,
                     18014398509481984, 36028797018963968]

    wfr_str_flags = ['INVALID', 'WATER', 'LAND', 'CLOUD', 'CLOUD_AMBIGUOUS', 'CLOUD_MARGIN', 'SNOW_ICE',
                     'INLAND_WATER', 'TIDAL', 'COSMETIC', 'SUSPECT', 'HISOLZEN', 'SATURATED', 'MEGLINT', 'HIGHGLINT',
                     'WHITECAPS', 'ADJAC', 'WV_FAIL', 'PAR_FAIL', 'AC_FAIL', 'OC4ME_FAIL', 'OCNN_FAIL', 'KDM_FAIL',
                     'BPAC_ON', 'WHITE_SCATT', 'LOWRW', 'HIGHRW', 'ANNOT_ANGSTROM', 'ANNOT_AERO_B', 'ANNOT_ABSO_D',
                     'ANNOT_ACLIM', 'ANNOT_ABSOA', 'ANNOT_MIXR1', 'ANNOT_DROUT', 'ANNOT_TAU06', 'RWNEG_O1', 'RWNEG_O2',
                     'RWNEG_O3', 'RWNEG_O4', 'RWNEG_O5', 'RWNEG_O6', 'RWNEG_O7', 'RWNEG_O8', 'RWNEG_O9', 'RWNEG_O10',
                     'RWNEG_O11', 'RWNEG_O12', 'RWNEG_O16', 'RWNEG_O17', 'RWNEG_O18', 'RWNEG_O21']

    # https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-3-olci/level-2/quality-and-science-flags-op
    wfr_bin2flags = {0: 'INVALID',
                     1: 'WATER',
                     2: 'LAND',
                     3: 'CLOUD',
                     23: 'CLOUD_AMBIGUOUS',
                     24: 'CLOUD_MARGIN',
                     4: 'SNOW_ICE',
                     5: 'INLAND_WATER',
                     6: 'TIDAL',
                     7: 'COSMETIC',
                     8: 'SUSPECT',
                     9: 'HISOLZEN',
                     10: 'SATURATED',
                     11: 'MEGLINT',
                     12: 'HIGHGLINT',
                     13: 'WHITECAPS',
                     14: 'ADJAC',
                     15: 'WV_FAIL',
                     16: 'PAR_FAIL',
                     17: 'AC_FAIL',
                     18: 'OC4ME_FAIL',
                     19: 'OCNN_FAIL',
                     21: 'KDM_FAIL',
                     25: 'BPAC_ON',
                     26: 'WHITE_SCATT',
                     27: 'LOWRW',
                     28: 'HIGHRW',
                     32: 'ANNOT_ANGSTROM',
                     33: 'ANNOT_AERO_B',
                     34: 'ANNOT_ABSO_D',
                     35: 'ANNOT_ACLIM',
                     36: 'ANNOT_ABSOA',
                     37: 'ANNOT_MIXR1',
                     38: 'ANNOT_DROUT',
                     39: 'ANNOT_TAU06',
                     40: 'RWNEG_O1',
                     41: 'RWNEG_O2',
                     42: 'RWNEG_O3',
                     43: 'RWNEG_O4',
                     44: 'RWNEG_O5',
                     45: 'RWNEG_O6',
                     46: 'RWNEG_O7',
                     47: 'RWNEG_O8',
                     48: 'RWNEG_O9',
                     49: 'RWNEG_O10',
                     50: 'RWNEG_O11',
                     51: 'RWNEG_O12',
                     52: 'RWNEG_O16',
                     53: 'RWNEG_O17',
                     54: 'RWNEG_O18',
                     55: 'RWNEG_O21'}

    # WFR pixels with these flags will be removed:
    wfr_remove = ['INVALID', 'CLOUD', 'CLOUD_AMBIGUOUS', 'CLOUD_MARGIN', 'SNOW_ICE',
                  'SUSPECT', 'SATURATED', 'AC_FAIL', 'MEGLINT', 'HIGHGLINT', 'LOWRW']

    # Pixels must include these flags:
    wfr_keep = ['INLAND_WATER']

    # TODO: used in parallel only (verify).
    wfr_files_p = (('w_aer.nc', 'A865'),
                   ('w_aer.nc', 'T865'),
                   ('Oa01_reflectance.nc', 'Oa01_reflectance'),
                   ('Oa02_reflectance.nc', 'Oa02_reflectance'),
                   ('Oa03_reflectance.nc', 'Oa03_reflectance'),
                   ('Oa04_reflectance.nc', 'Oa04_reflectance'),
                   ('Oa05_reflectance.nc', 'Oa05_reflectance'),
                   ('Oa06_reflectance.nc', 'Oa06_reflectance'),
                   ('Oa07_reflectance.nc', 'Oa07_reflectance'),
                   ('Oa08_reflectance.nc', 'Oa08_reflectance'),
                   ('Oa09_reflectance.nc', 'Oa09_reflectance'),
                   ('Oa10_reflectance.nc', 'Oa10_reflectance'),
                   ('Oa11_reflectance.nc', 'Oa11_reflectance'),
                   ('Oa12_reflectance.nc', 'Oa12_reflectance'),
                   ('Oa16_reflectance.nc', 'Oa16_reflectance'),
                   ('Oa17_reflectance.nc', 'Oa17_reflectance'),
                   ('Oa18_reflectance.nc', 'Oa18_reflectance'),
                   ('Oa21_reflectance.nc', 'Oa21_reflectance'),
                   ('wqsf.nc', 'WQSF'),
                   ('tsm_nn.nc', 'TSM_NN'))

    syn_files = {
        'Syn_AOT550.nc': ['T550'],
        'Syn_Angstrom_exp550.nc': ['A550']
    }

    syn_vld_names = {}  # TODO: populate with the valid Synergy equivalent netcdf files.

    wfr_files = {
        'w_aer.nc': ['A865', 'T865'],
        'Oa01_reflectance.nc': ['Oa01_reflectance'],
        'Oa02_reflectance.nc': ['Oa02_reflectance'],
        'Oa03_reflectance.nc': ['Oa03_reflectance'],
        'Oa04_reflectance.nc': ['Oa04_reflectance'],
        'Oa05_reflectance.nc': ['Oa05_reflectance'],
        'Oa06_reflectance.nc': ['Oa06_reflectance'],
        'Oa07_reflectance.nc': ['Oa07_reflectance'],
        'Oa08_reflectance.nc': ['Oa08_reflectance'],
        'Oa09_reflectance.nc': ['Oa09_reflectance'],
        'Oa10_reflectance.nc': ['Oa10_reflectance'],
        'Oa11_reflectance.nc': ['Oa11_reflectance'],
        'Oa12_reflectance.nc': ['Oa12_reflectance'],
        'Oa16_reflectance.nc': ['Oa16_reflectance'],
        'Oa17_reflectance.nc': ['Oa17_reflectance'],
        'Oa18_reflectance.nc': ['Oa18_reflectance'],
        'Oa21_reflectance.nc': ['Oa21_reflectance'],
        'wqsf.nc': ['WQSF']
    }

    wfr_vld_names = {
        'lon': 'longitude:double',
        'lat': 'latitude:double',
        'Oa01_reflectance': 'Oa01_reflectance:float',
        'Oa02_reflectance': 'Oa02_reflectance:float',
        'Oa03_reflectance': 'Oa03_reflectance:float',
        'Oa04_reflectance': 'Oa04_reflectance:float',
        'Oa05_reflectance': 'Oa05_reflectance:float',
        'Oa06_reflectance': 'Oa06_reflectance:float',
        'Oa07_reflectance': 'Oa07_reflectance:float',
        'Oa08_reflectance': 'Oa08_reflectance:float',
        'Oa09_reflectance': 'Oa09_reflectance:float',
        'Oa10_reflectance': 'Oa10_reflectance:float',
        'Oa11_reflectance': 'Oa11_reflectance:float',
        'Oa12_reflectance': 'Oa12_reflectance:float',
        'Oa16_reflectance': 'Oa16_reflectance:float',
        'Oa17_reflectance': 'Oa17_reflectance:float',
        'Oa18_reflectance': 'Oa18_reflectance:float',
        'Oa21_reflectance': 'Oa21_reflectance:float',
        'OAA': 'OAA:float',
        'OZA': 'OZA:float',
        'SAA': 'SAA:float',
        'SZA': 'SZA:float',
        'WQSF': 'WQSF_lsb:double',
        'A865': 'A865:float',
        'T865': 'T865:float',
        'TSM_NN': 'TSM_NN'
    }

    l1_wave_bands = {
        'Oa01': 400,
        'Oa02': 412.5,
        'Oa03': 442.5,
        'Oa04': 490,
        'Oa05': 510,
        'Oa06': 560,
        'Oa07': 620,
        'Oa08': 665,
        'Oa09': 673.75,
        'Oa10': 681.25,
        'Oa11': 708.75,
        'Oa12': 753.75,
        'Oa13': 761.25,
        'Oa14': 764.375,
        'Oa15': 767.5,
        'Oa16': 778.75,
        'Oa17': 865,
        'Oa18': 885,
        'Oa19': 900,
        'Oa20': 940,
        'Oa21': 1020
    }

    s3_bands_l2 = {
        'Oa01': 400,
        'Oa02': 412.5,
        'Oa03': 442.5,
        'Oa04': 490,
        'Oa05': 510,
        'Oa06': 560,
        'Oa07': 620,
        'Oa08': 665,
        'Oa09': 673.75,
        'Oa10': 681.25,
        'Oa11': 708.75,
        'Oa12': 753.75,
        'Oa16': 778.75,
        'Oa17': 865,
        'Oa18': 885,
        'Oa21': 1020
    }

    dbs_colors = {-1: 'C0', 0: 'C1', 1: 'C2', 2: 'C3', 3: 'C4', 4: 'C5', 5: 'C6', 6: 'C7', 7: 'C8', 8: 'C9', 9: 'k',
                  10: 'k', 11: 'k', 12: 'k', 13: 'k', 14: 'k', 15: 'k', 16: 'k', 17: 'k', 18: 'k', 19: 'k', 20: 'k'}

    dbs_s3b_l2 = {
        'Oa01_reflectance:float': 400,
        'Oa02_reflectance:float': 412.5,
        'Oa03_reflectance:float': 442.5,
        'Oa04_reflectance:float': 490,
        'Oa05_reflectance:float': 510,
        'Oa06_reflectance:float': 560,
        'Oa07_reflectance:float': 620,
        'Oa08_reflectance:float': 665,
        'Oa09_reflectance:float': 673.75,
        'Oa10_reflectance:float': 681.25,
        'Oa11_reflectance:float': 708.75,
        'Oa12_reflectance:float': 753.75,
        'Oa16_reflectance:float': 778.75,
        'Oa17_reflectance:float': 865,
        'Oa18_reflectance:float': 885,
        'Oa21_reflectance:float': 1020
    }

class Footprinter:

    @staticmethod
    def _xml2dict(xfdumanifest):
        """
        Internal function that reads the .SEN3/xfdumanifest.xml and generates a dictionary with the relevant data.
        """
        result = {}
        with open(xfdumanifest) as xmlf:
            # grab the relevant contents and add them to a dict
            for line in xmlf:
                if "<gml:posList>" in line:
                    result['gml_data'] = f'<gml:Polygon xmlns:gml="http://www.opengis.net/gml" ' \
                                    f'srsName="http://www.opengis.net/def/crs/EPSG/0/4326">' \
                                    f'<gml:exterior><gml:LinearRing>{line}</gml:LinearRing>' \
                                    f'</gml:exterior></gml:Polygon>'
                # get only the values between the tags
                if '<sentinel3:rows>' in line:
                    result['rows'] = int(line.split('</')[0].split('>')[1])
                if '<sentinel3:columns>' in line:
                    result['cols'] = int(line.split('</')[0].split('>')[1])
        return result

    @staticmethod
    def _gml2shp(gml_in, shp_out):
        """
        given a .GML file (gml_in), convert it to a .SHP (shp_out).
        """
        # https://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html#create-a-new-layer-from-the-extent-of-an-existing-layer
        # Get the Layer's Geometry
        inGMLfile = gml_in
        inDriver = ogr.GetDriverByName("GML")
        inDataSource = inDriver.Open(inGMLfile, 0)
        inLayer = inDataSource.GetLayer()
        feature = inLayer.GetNextFeature()
        geometry = feature.GetGeometryRef()

        # Get the list of coordinates from the footprint
        json_footprint = json.loads(geometry.ExportToJson())
        footprint = json_footprint['coordinates'][0]

        # Create a Polygon from the extent tuple
        ring = ogr.Geometry(ogr.wkbLinearRing)

        for point in footprint:
            ring.AddPoint(point[0], point[1])

        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

        # Save extent to a new Shapefile
        outShapefile = shp_out
        outDriver = ogr.GetDriverByName("ESRI Shapefile")

        # Remove output shapefile if it already exists
        if os.path.exists(outShapefile):
            outDriver.DeleteDataSource(outShapefile)

        # create the spatial reference, WGS84
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)

        # Create the output shapefile
        outDataSource = outDriver.CreateDataSource(outShapefile)
        outLayer = outDataSource.CreateLayer("s3_footprint", srs, geom_type=ogr.wkbPolygon)

        # Add an ID field
        idField = ogr.FieldDefn("id", ogr.OFTInteger)
        outLayer.CreateField(idField)

        # Create the feature and set values
        featureDefn = outLayer.GetLayerDefn()
        feature = ogr.Feature(featureDefn)
        feature.SetGeometry(poly)
        feature.SetField("id", 1)
        outLayer.CreateFeature(feature)
        feature = None

        # Save and close DataSource
        inDataSource = None
        outDataSource = None
        pass

    def manifest2shp(self, xfdumanifest, filename):
        """
        Given a .SEN3/xfdumanifest.xml and a filename, generates a .shp and a .gml
        """
        # get the dict
        xmldict = self._xml2dict(xfdumanifest)
        # add path to "to-be-generated" gml and shp files
        xmldict['gml_path'] = filename + '.gml'
        xmldict['shp_path'] = filename + '.shp'
        # write the gml_data from the dict to an actual .gml file
        with open(xmldict['gml_path'], 'w') as gmlfile:
            gmlfile.write(xmldict['gml_data'])
        # DEPRECATED:
        # call the ogr2ogr.py script to generate a .shp from a .gml
        # ogr2ogr.main(["", "-f", "ESRI Shapefile", xmldict['shp_path'], xmldict['gml_path']])

        self._gml2shp(xmldict['gml_path'], xmldict['shp_path'])
        return xmldict

    @staticmethod
    def _shp_extent(shp):
        """
        Reads a ESRI Shapefile and return its extent as a str separated by spaces.
        e.x. output: '-71.6239 -58.4709 -9.36789 0.303954'
        """
        ds = ogr.Open(shp)
        layer = ds.GetLayer()
        feature = layer.GetNextFeature()
        geometry = feature.GetGeometryRef()
        extent = geometry.GetEnvelope()  # ex: (-71.6239, -58.4709, -9.36789, 0.303954)
        # cast the 4-elements tuple into a list of strings
        extent_str = [str(i) for i in extent]
        return ' '.join(extent_str)

    def manifest2tiff(self, xfdumanifest):
        """
        Reads .SEN3/xfdumanifest.xml and generates a .tiff raster.
        """
        # get the complete directory path but not the file base name
        img_path = xfdumanifest.split('/xfdu')[0]
        # get only the date of the img from the complete path, ex: '20190904T133117'
        figdate = os.path.basename(img_path).split('____')[1].split('_')[0]
        # add an img.SEN3/footprint folder
        footprint_dir = os.path.join(img_path, 'footprint')
        Path(footprint_dir).mkdir(parents=True, exist_ok=True)
        # ex: img.SEN3/footprint/20190904T133117_footprint.tiff
        path_file_tiff = os.path.join(footprint_dir, figdate+'_footprint.tiff')
        # only '20190904T133117_footprint' nothing else.
        lyr = os.path.basename(path_file_tiff).split('.')[0]
        # img.SEN3/footprint/20190904T133117_footprint (without file extension)
        fname = path_file_tiff.split('.tif')[0]
        # get the dict + generate .shp and .gml files
        xmldict = self.manifest2shp(xfdumanifest, fname)
        cols = xmldict['cols']
        rows = xmldict['rows']
        shp = xmldict['shp_path']
        extent = self._shp_extent(shp)
        # generate the complete cmd string
        cmd = f'gdal_rasterize -l {lyr} -burn 1.0 -ts {cols}.0 {rows}.0 -a_nodata 0.0 -te {extent} -ot Float32 {shp} {path_file_tiff}'
        # call the cmd
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        proc.wait()
        print(f'{figdate} done.')
        return True

    @staticmethod
    def touch_test(footprint_vector, roi_vector):
        """
        Given two input shapefiles (one Sentinel-3 image footprint and
        the user region of interest), test if the touch each other.
        """
        # Hotfix: GDAL does not open Path objects, they must be cast to strings
        footprint_vector = str(footprint_vector)
        roi_vector = str(roi_vector)

        footprint_vector_name = os.path.basename(footprint_vector).split('.')[1]
        if footprint_vector_name.lower() == 'shp':
            foot_ds = ogr.GetDriverByName("ESRI Shapefile").Open(footprint_vector, 0)
        elif footprint_vector_name.lower() == 'json' or footprint_vector_name.lower() == 'geojson':
            foot_ds = ogr.GetDriverByName("GeoJSON").Open(footprint_vector, 0)
        else:
            logging.info(f'Input ROI {os.path.basename(footprint_vector)} not recognized as a valid vector file. '
                         f'Make sure the input file is of type .shp .json or .geojson and try again.')
            sys.exit(1)

        layer_foot = foot_ds.GetLayer()
        feature_foot = layer_foot.GetNextFeature()
        geometry_foot = feature_foot.GetGeometryRef()

        roi_type_name = os.path.basename(roi_vector).split('.')[1]
        if roi_type_name.lower() == 'shp':
            data = ogr.GetDriverByName("ESRI Shapefile").Open(roi_vector, 0)
        elif roi_type_name.lower() == 'json' or roi_type_name.lower() == 'geojson':
            data = ogr.GetDriverByName("GeoJSON").Open(roi_vector, 0)
        else:
            logging.info(f'Input ROI {os.path.basename(roi_vector)} not recognized as a valid vector file. '
                         f'Make sure the input file is of type .shp .json or .geojson and try again.')
            sys.exit(1)

        # Adapted from https://gist.github.com/CMCDragonkai/e7b15bb6836a7687658ec2bb3abd2927
        for layer in data:
            # this is the one where featureindex may not start at 0
            layer.ResetReading()
            for feature in layer:
                geometry_roi = feature.geometry()
                intersection = geometry_roi.Intersection(geometry_foot)
                if not intersection.IsEmpty():
                    return True

        # This will only run if no geometry from the ROI touched the S3 footprint.
        return False


