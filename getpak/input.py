import os
import sys
import psutil
import rasterio
import rasterio.mask
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

from osgeo import gdal
from datetime import datetime
from dask.distributed import Client as dkClient, LocalCluster

# GET-Pak imports
from getpak.commons import Utils
from getpak.commons import DefaultDicts as dd

u = Utils()


class Input:
    """
    Core function to read any input images for processing with GET-Pak
    """

    def __init__(self, parent_log=None):
        # if parent_log:
        #     self.log = parent_log
        # else:
        #     INSTANCE_TIME_TAG = datetime.now().strftime('%Y%m%dT%H%M%S')
        #     logfile = os.path.join(os.getcwd(), 'getpak_raster_' + INSTANCE_TIME_TAG + '.log')
        #     self.log = u.create_log_handler(logfile)

        # start dask with maximum of 16 GB of RAM
        try:
            dkClient.current()
        except ValueError:
            # total memory available
            mem = int(0.75 * psutil.virtual_memory().total / (1024 * 1024 * 1024))
            # memory limit
            limit = 16 if mem > 16 else mem
            # starting dask
            cluster = LocalCluster(n_workers=4, memory_limit=str(limit / 4) + 'GB')
            client = dkClient(cluster)

    def get_input_dict(self, file, sensor='S2MSI', AC_processor='GRS', grs_version=None):
        """
        Function to open the satellite image, depending on user information of sensor and atmospheric correction
        processor.
        This class is just a wrapper as it uses the different classes for the different ACs to read the input

        Parameters
        ----------
        @file: the path to the image
        @sensor: a string of the satellite mission, one of:
            S2MSI for Sentinel-2 MSI A and B
            S3OLCI for Sentinel-3 OLCI A and B
        @AC_processor: a string of the AC processor, one of ACOLITE, GRS or SeaDAS

        Returns
        -------
        @return img: xarray.DataArray containing the Rrs bands.
        """

        if sensor == 'S2MSI' and AC_processor == 'GRS':
            if grs_version:
                g = GRS()
                img, meta, proj, trans = g.get_grs_dict(grs_nc_file=file, grs_version=grs_version)
                self.meta = meta
                self.proj = proj
                self.trans = trans
            else:
                print("Error: No GRS version!")
                sys.exit(1)
        elif sensor == 'S2MSI' and AC_processor == 'ACOLITE':
            a = ACOLITE_S2()
            img, meta, proj, trans = a.get_aco_dict(aco_nc_file=file)
            self.meta = meta
            self.proj = proj
            self.trans = trans
        else:
            # self.log.error(f'Error: Wrong sensor or AC processor!')
            print("Error: Wrong sensor or AC processor!")
            img = None
            sys.exit(1)

        return img


class GRS:
    """
    Core functionalities to handle GRS files

    Methods
    -------
    metadata(grs_file_entry)
        Given a GRS string element, return file metadata extracted from its name.
    """

    def __init__(self, parent_log=None):
        # if parent_log:
        #     self.log = parent_log
        # else:
        #     INSTANCE_TIME_TAG = datetime.now().strftime('%Y%m%dT%H%M%S')
        #     logfile = os.path.join(os.getcwd(), 'getpak_raster_' + INSTANCE_TIME_TAG + '.log')
        #     self.log = u.create_log_handler(logfile)

        # import band names from Commons/DefaultDicts
        pass

    @staticmethod
    def metadata(grs_file_entry):
        """
        Given a GRS file return metadata extracted from its name:

        Parameters
        ----------
        @param grs_file_entry: str or pathlike obj that leads to the GRS.nc file.

        @return: metadata (dict) containing the extracted info, available keys are:
            input_file, basename, mission, prod_lvl, str_date, pydate, year,
            month, day, baseline_algo_version, relative_orbit, tile,
            product_discriminator, cloud_cover, grs_ver.

        Reference
        ---------
        Given the following file:
        /root/23KMQ/2021/05/21/S2A_MSIL1C_20210521T131241_N0300_R138_T23KMQ_20210521T163353_cc020_v15.nc

        S2A : (MMM) is the mission ID(S2A/S2B)
        MSIL1C : (MSIXXX) Product procesing level
        20210521T131241 : (YYYYMMDDTHHMMSS) Sensing start time
        N0300 : (Nxxyy) Processing Baseline number
        R138 : Relative Orbit number (R001 - R143)
        T23KMQ : (Txxxxx) Tile Number
        20210521T163353 : Product Discriminator
        cc020 : GRS Cloud cover estimation (0-100%)
        v15 : GRS algorithm baseline version

        For GRS version >=2.0, the naming does not include the cloud cover estimation nor the GRS version

        Further reading:
        Sentinel-2 MSI naming convention:
        URL = https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/naming-convention
        """
        metadata = {}
        basefile = os.path.basename(grs_file_entry)
        splt = basefile.split('_')
        if len(splt) == 9:
            mission, proc_level, date_n_time, proc_ver, r_orbit, tile, prod_disc, cc, aux = basefile.split('_')
            ver, _ = aux.split('.')
        else:
            mission, proc_level, date_n_time, proc_ver, r_orbit, tile, aux = basefile.split('_')
            prod_disc, _ = aux.split('.')
            cc, ver = ['NA', 'NA']
        file_event_date = datetime.strptime(date_n_time, '%Y%m%dT%H%M%S')
        yyyy = f"{file_event_date.year:02d}"
        mm = f"{file_event_date.month:02d}"
        dd = f"{file_event_date.day:02d}"

        metadata['input_file'] = grs_file_entry
        metadata['basename'] = basefile
        metadata['mission'] = mission
        metadata['prod_lvl'] = proc_level
        metadata['str_date'] = yyyy + mm + dd
        metadata['pydate'] = file_event_date
        metadata['year'] = yyyy
        metadata['month'] = mm
        metadata['day'] = dd
        metadata['baseline_algo_version'] = proc_ver
        metadata['relative_orbit'] = r_orbit
        metadata['tile'] = tile
        metadata['product_discriminator'] = prod_disc
        metadata['cloud_cover'] = cc
        metadata['grs_ver'] = ver

        return metadata

    def get_grs_dict(self, grs_nc_file, grs_version='v20'):
        """
        Open GRS netCDF files using xarray and dask, and return
        a DataArray containing only the Rrs bands.

        Parameters
        ----------
        @grs_nc_file: the path to the GRS file
        @grs_version: a string with GRS version ('v15', 'v20' for version 2.0.5, 'v21' for version 2.1.6+)

        Returns
        -------
        @return grs: xarray.DataArray containing 11 Rrs bands.
        The band names can be found at getpak.commons.DefaultDicts.grs_v20nc_s2bands
        """
        meta = self.metadata(grs_nc_file)
        # list of bands
        bands = self.grs_v20nc_s2bands
        # self.log.info(f'Opening GRS version {grs_version} file {grs_nc_file}')
        if grs_version == 'v15':
            ds = xr.open_dataset(grs_nc_file, engine="h5netcdf", decode_coords='all', chunks={'y': -1, 'x': -1})
            trans = ds.rio.transform()
            proj = rasterio.crs.CRS.from_wkt(ds['spatial_ref'].attrs.get('crs_wkt'))
            # List of variables to keep
            if 'Rrs_B1' in ds.variables:
                variables_to_keep = bands
                # Drop the variables you don't want
                variables_to_drop = [var for var in ds.variables if var not in variables_to_keep]
                grs = ds.drop_vars(variables_to_drop)
                grs.attrs["proj"] = proj
                grs.attrs["trans"] = trans
        elif grs_version == 'v20':
            ds = xr.open_dataset(grs_nc_file, chunks={'y': -1, 'x': -1}, engine="h5netcdf")
            trans = ds.rio.transform()
            proj = rasterio.crs.CRS.from_wkt(ds['spatial_ref'].attrs.get('crs_wkt'))
            subset_dict = {band: ds['Rrs'].sel(wl=wave).drop_vars(['wl', 'time']) for band, wave in bands.items()}
            grs = xr.Dataset(subset_dict)
            grs.attrs["proj"] = proj
            grs.attrs["trans"] = trans
        elif grs_version == 'v21':
            ds = xr.open_dataset(grs_nc_file, chunks={'y': -1, 'x': -1}, engine="h5netcdf")
            trans = ds.rio.transform()
            proj = rasterio.crs.CRS.from_wkt(ds['spatial_ref'].attrs.get('crs_wkt'))
            subset_dict = {band: ds['Rrs'].sel(wl=wave).drop_vars(['wl', 'time', 'band', 'central_wavelength']) for
                           band, wave in bands.items()}
            grs = xr.Dataset(subset_dict)
            grs.attrs["proj"] = proj
            grs.attrs["trans"] = trans
        else:
            # self.log.error(f'GRS version {grs_version} not supported.')
            grs = None
            sys.exit(1)

        ds.close()

        return grs, meta, proj, trans

    def _get_shp_features(self, shp_file, unique_key='id', grs_crs='EPSG:32720'):
        '''
        INTERNAL FUNCTION
        Given a shp_file.shp, read each feature by unique 'id' and save it in a dict.
        OBS1: Shp must have a unique key to identify each feature (call it 'id').
        OBS2: The crs of the shapefile must be the same as the crs of the raster (EPSG:32720).
        '''
        try:
            # self.log.info(f'Reading shapefile: {shp_file}')
            shp = gpd.read_file(shp_file)
            # self.log.info(f'CRS: {str(shp.crs)}')
            # if str(shp.crs) != grs_crs:
            # self.log.info(f'CRS mismatch, expected {grs_crs}.')
            # self.log.info(f'Program may work incorrectly.')
            # Initialize a dict to handle all the geodataframes by id
            feature_dict = {}
            # Iterate over all features in the shapefile and populate the dict
            for id in shp.id.values:
                shp_feature = shp[shp[unique_key] == id]
                feature_dict[id] = shp_feature

            # self.log.info(f'Done. {len(feature_dict)} features found.')

        except Exception as e:
            # self.log.error(f'Error: {e}')
            sys.exit(1)

        return feature_dict

    # Internal function to paralelize the process of each band and point
    # TODO: improve documentation
    def _process_band_point(self, band, shp_feature, crs, pt_id):
        """
        INTERNAL FUNCTION
        """
        Rrs_raster_band = self.grs_dict[band]
        Rrs_raster_band.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        clipped_rrs = Rrs_raster_band.rio.clip(shp_feature.geometry.values, crs)
        global counter
        if counter % 10 == 0:
            print(f'> {counter}')
        else:
            print('>', end='')
        counter += 1
        mean_rrs = clipped_rrs.mean().compute().data.item()
        return (band, pt_id, mean_rrs)


class ACOLITE_S2:
    """
    Core functionalities to handle ACOLITE S2 files

    Methods
    -------
    metadata(grs_file_entry)
        Given a ACOLITE string element, return file metadata extracted from its name.
    """

    def __init__(self, parent_log=None):
        # if parent_log:
        #     self.log = parent_log
        # else:
        #     INSTANCE_TIME_TAG = datetime.now().strftime('%Y%m%dT%H%M%S')
        #     logfile = os.path.join(os.getcwd(), 'getpak_raster_' + INSTANCE_TIME_TAG + '.log')
        #     self.log = u.create_log_handler(logfile)

        # import band names from Commons/DefaultDicts
        self.acolite_nc_s2abands = dd.acolite_nc_s2abands
        self.acolite_nc_s2bbands = dd.acolite_nc_s2bbands

    @staticmethod
    def metadata(aco_file_entry):
        """
        Given a ACOLITE file return metadata extracted from its name:

        Parameters
        ----------
        @param aco_file_entry: str or pathlike obj that leads to the .nc file.

        @return: metadata (dict) containing the extracted info, available keys are:
            input_file, basename, mission, str_date, pydate, year, month, day, tile, product_type

        Reference
        ---------
        Given the following file:
        S2B_MSI_2024_06_06_14_45_27_T20LLQ_L2R.nc

        S2A : (MMM) is the mission ID(S2A/S2B)
        MSI : (MSI) is the sensor
        2024_06_06 : (YYYY_MM_DD) Sensing date
        14_45_27 : (??????) who knows?
        T20LLQ : (Txxxxx) Tile Number
        L2R : ACOLITE product type

        Further reading:
        Sentinel-2 MSI naming convention:
        URL = https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/naming-convention
        """
        metadata = {}
        basefile = os.path.basename(aco_file_entry)
        splt = basefile.split('_')
        if len(splt) == 10:
            mission, sensor, yyyy, mm, dd, _, _, _, tile, aux = basefile.split('_')
            prod_type, _ = aux.split('.')

        date = yyyy + mm + dd
        file_event_date = datetime.strptime(date, '%Y%m%d')

        metadata['input_file'] = aco_file_entry
        metadata['basename'] = basefile
        metadata['mission'] = mission
        metadata['str_date'] = date
        metadata['pydate'] = file_event_date
        metadata['year'] = yyyy
        metadata['month'] = mm
        metadata['day'] = dd
        metadata['tile'] = tile
        metadata['prod_type'] = prod_type

        return metadata

    def get_aco_dict(self, aco_nc_file):
        """
        Open ACOLITE netCDF files using xarray and dask, and return
        a DataArray containing only the Rrs bands.

        Parameters
        ----------
        @grs_nc_file: the path to the ACOLITE file

        Returns
        -------
        @return aco: xarray.DataArray containing 11 Rrs bands.
        The band names can be found at getpak.commons.DefaultDicts.aco_nc_s2*bands
        """
        # list of bands
        meta = self.metadata(aco_nc_file)
        # checking if Sentinel-2 A or B
        if meta['mission'] == 'S2A':
            bands = self.acolite_nc_s2abands
        elif meta['mission'] == 'S2B':
            bands = self.acolite_nc_s2bbands

        # Opening the ACOLITE file
        # self.log.info(f'Opening ACOLITE file {aco_nc_file}')
        ds = xr.open_dataset(aco_nc_file, chunks={'y': -1, 'x': -1}, engine="h5netcdf")
        # Getting spatial information from the Dataset
        proj = ds.attrs.get("proj4_string")
        trans = ds.rio.transform()
        # Subsetting only the bands and renaming them to the naming convention, and dividing by pi (to Rrs) if L2R
        if meta['prod_type'] == 'L2R':
            adjbands = {key: f"rhos_{value}" for key, value in bands.items()}
            subset_dict = {new_name: ds[var_name] / np.pi for new_name, var_name in adjbands.items()}
        elif meta['prod_type'] == 'L2W':
            if "Rrs_833" in ds.variables:
                adjbands = {key: f"Rrs_{value}" for key, value in bands.items()}
                subset_dict = {new_name: ds[var_name] for new_name, var_name in adjbands.items()}
                meta['prod_type'] = 'Rrs'
            else:
                adjbands = {key: f"rhow_{value}" for key, value in bands.items()}
                subset_dict = {new_name: ds[var_name] / np.pi for new_name, var_name in adjbands.items()}
        else:
            # self.log.error(f'ACOLITE product not supported.')
            aco = None
            sys.exit(1)

        aco = xr.Dataset(subset_dict)
        aco.attrs["proj"] = proj
        aco.attrs["trans"] = trans
        ds.close()

        return aco, meta, proj, trans


