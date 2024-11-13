import os
import re
import sys
import json
import dask
import logging
import rasterio
import rasterio.mask
import concurrent.futures
import importlib_resources

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
import geopandas as gpd

from osgeo import gdal
from dask import delayed
from pathlib import Path
from datetime import datetime
from skimage.draw import polygon
from skimage.transform import resize
from rasterstats import zonal_stats
from rasterio.warp import calculate_default_transform, reproject, Resampling

from getpak.commons import Utils as u
from getpak.commons import DefaultDicts as d


class Raster:
    """
    Generic class containing methods for matricial manipulations
    
    Methods
    -------
    array2tiff(ndarray_data, str_output_file, transform, projection, no_data=-1, compression='COMPRESS=PACKBITS')
        Given an input ndarray and the desired projection parameters, create a raster.tif using rasterio.

    array2tiff_gdal(ndarray_data, str_output_file, transform, projection, no_data=-1, compression='COMPRESS=PACKBITS')
        Given an input ndarray and the desired projection parameters, create a raster.tif using GDT_Float32.

    reproj(in_raster, out_raster, target_crs='EPSG:4326')
        Given an input raster.tif reproject it to reprojected.tif using @target_crs

    s2proj_ref_builder(wd_image_tif)
        Given an input WD output water_mask.tif over the desires Sentinel-2 tile-grid system (ex: 20LLQ),
        output the GDAL transformation, projection, rows and columns of the input image.        

    shp_stats(tif_file, shp_poly, keep_spatial=True, statistics='count min mean max median std')
        Given a single-band GeoTIFF file and a vector.shp return statistics inside the polygon.

    extract_px(rasterio_rast, shapefile, rrs_dict, bands)
        Given a dict of Rrs and a polygon, to extract the values of pixels from each band

    sam(self, values, single=False)
        Given a a set of pixels, uses the Spectral Angle Mapper to generate angle between the Rrs and the OWTs

    classify_owt_px(self, rrs_dict, bands)
        Function to classify the OWT of each pixel

    classify_owt(self, rasterio_rast, shapefiles, rrs_dict, bands, min_px=6)
        Function to classify the the OWT of pixels inside a shapefile (or a set of shapefiles)

    """

    def __init__(self, parent_log=None):
        if parent_log:
            self.log = parent_log
        else:
            INSTANCE_TIME_TAG = datetime.now().strftime('%Y%m%dT%H%M%S')
            logfile = os.path.join(os.getcwd(), 'getpak_raster_' + INSTANCE_TIME_TAG + '.log')
        # Import CRS projection information from /data/s2_proj_ref.json
        s2projdata = importlib_resources.files(__name__).joinpath('data/s2_proj_ref.json')
        with s2projdata.open('rb') as fp:
            byte_content = fp.read()
        self.s2projgrid = json.loads(byte_content)

        # Import OWT means for S2 MSI from /data/means_OWT_Spyrakos_S2A_B2-7.json
        means_owt = importlib_resources.files(__name__).joinpath('data/means_OWT_Spyrakos_S2A_B1-7.json')
        with means_owt.open('rb') as fp:
            byte_content = fp.read()
        self.owts_B1_7 = dict(json.loads(byte_content))

        means_owt = importlib_resources.files(__name__).joinpath('data/means_OWT_Spyrakos_S2A_B2-7.json')
        with means_owt.open('rb') as fp:
            byte_content = fp.read()
        self.owts_B2_7 = dict(json.loads(byte_content))

        means_owt = importlib_resources.files(__name__).joinpath('data/Means_OWT_Cordeiro_S2A_SPM.json')
        with means_owt.open('rb') as fp:
            byte_content = fp.read()
        self.owts_spm_B1_8A = dict(json.loads(byte_content))

        means_owt = importlib_resources.files(__name__).joinpath('data/Means_OWT_Cordeiro_S2A_SPM_B2-8A.json')
        with means_owt.open('rb') as fp:
            byte_content = fp.read()
        self.owts_spm_B2_8A = dict(json.loads(byte_content))

        # raster

    @staticmethod
    def array2tiff(ndarray_data, str_output_file, transform, projection, no_data=-1, compression='COMPRESS=PACKBITS'):
        """
        Given an input ndarray and the desired projection parameters, create a raster.tif using GDT_Float32.
        
        Parameters
        ----------
        @param ndarray_data: Inform if the index should be saved as array in the output folder
        @param str_output_file: string of the path of the file to be written
        @param transform: rasterio affine transformation matrix (resolution and "upper left" coordinate)
        @param projection: projection CRS
        @param no_data: the value for no data
        @param compression: type of file compression

        @return: None (If all goes well, array2tiff should pass and generate a file inside @str_output_file)
        """
        with rasterio.open(fp=str_output_file,
                           mode='w',
                           driver='GTiff',
                           height=ndarray_data.shape[0],
                           width=ndarray_data.shape[1],
                           count=1,
                           dtype=ndarray_data.dtype,
                           crs=projection,
                           transform=transform,
                           nodata=no_data,
                           options=[compression]) as dst:
            dst.write(ndarray_data, 1)

        pass

    @staticmethod
    def array2tiff_gdal(ndarray_data, str_output_file, transform, projection, no_data=-1,
                        compression='COMPRESS=PACKBITS'):
        """
        Given an input ndarray and the desired projection parameters, create a raster.tif using GDT_Float32.

        Parameters
        ----------
        @param ndarray_data: Inform if the index should be saved as array in the output folder
        @param str_output_file:
        @param transform:
        @param projection: projection CRS
        @param no_data: the value for no data
        @param compression: type of file compression

        @return: None (If all goes well, array2tiff should pass and generate a file inside @str_output_file)
        """
        # Create file using information from the template
        outdriver = gdal.GetDriverByName("GTiff")  # http://www.gdal.org/gdal_8h.html
        # imgs_out = /work/scratch/guimard/grs2spm/
        [cols, rows] = ndarray_data.shape
        # GDT_Byte = 1, GDT_UInt16 = 2, GDT_UInt32 = 4, GDT_Int32 = 5, GDT_Float32 = 6,
        # options=['COMPRESS=PACKBITS'] -> https://gdal.org/drivers/raster/gtiff.html#creation-options
        outdata = outdriver.Create(str_output_file, rows, cols, 1, gdal.GDT_Float32, options=[compression])
        # Write the array to the file, which is the original array in this example
        outdata.GetRasterBand(1).WriteArray(ndarray_data)
        # Set a no data value if required
        outdata.GetRasterBand(1).SetNoDataValue(no_data)
        # Georeference the image
        outdata.SetGeoTransform(transform)
        # Write projection information
        outdata.SetProjection(projection)
        # Close the file https://gdal.org/tutorials/raster_api_tut.html#using-create
        outdata = None
        pass

    @staticmethod
    def reproj(in_raster, out_raster, target_crs='EPSG:4326'):
        """
        Given an input raster.tif reproject it to reprojected.tif using @target_crs (default = 'EPSG:4326').
        
        Parameters
        ----------
        @param in_raster: Inform if the index should be saved as array in the output folder
        @param out_raster:
        @param target_crs:
        
        @return: None (If all goes well, reproj should pass and generate a reprojected file inside @out_raster)
        """
        # Open the input raster file
        with rasterio.open(in_raster) as src:
            # Calculate the transformation parameters
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds)

            # Define the output file metadata
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            # Create the output raster file
            with rasterio.open(out_raster, 'w', **kwargs) as dst:
                # Reproject the data
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest)
        print(f'Done: {out_raster}')
        pass

    @staticmethod
    def s2proj_ref_builder(img_path_str):
        """
        Given a WaterDetect output .tif image
        return GDAL information metadata
        
        Parameters
        ----------
        @param img_path_str: path to the image file

        @return: tile_id (str) and ref (dict) containing the GDAL information
        """
        # img_parent_name = os.path.basename(Path(img_path_str).parents[1])
        # sliced_ipn = img_parent_name.split('_') 
        # tile_id = sliced_ipn[5][1:]
        match = re.search(r'T\d{2}[A-Z]{3}', img_path_str)
        tile_id = match.group(0)[1:]
        
        # Get GDAL information from the template file
        ref_data = gdal.Open(img_path_str)
        mtx = ref_data.ReadAsArray()
        trans = ref_data.GetGeoTransform()
        proj = ref_data.GetProjection()
        rows, cols = mtx.shape # return Y / X
        ref = {
            'trans': trans,
            'proj': proj,
            'rows': rows,
            'cols': cols
        }
        #close GDAL image
        del ref_data
        return tile_id, ref
    
    @staticmethod
    def shp_stats(tif_file, shp_poly, keep_spatial=False, statistics='count min mean max median std'):
        """
        Given a single-band GeoTIFF file and a vector.shp return statistics inside the polygon.
        
        Parameters
        ----------
        @param tif_file: path to raster.tif file.
        @param shp_poly: path to the polygon.shp file.
        @param keep_spatial (bool): 
            True = include the input shp_poly in the output as GeoJSON 
            False (default) = get only the mini_raster and statistics
        @param statistics: what to extract from the shapes, available values are:
        
        min, max, mean, count, sum, std, median, majority,
        minority, unique, range, nodata, percentile.
        https://pythonhosted.org/rasterstats/manual.html#zonal-statistics        
        
        @return: roi_stats (dict) containing the extracted statistics inside the region of interest.
        """
        # with fiona.open(shp_poly) as src:
        #     roi_stats = zonal_stats(src,
        #                             tif_file,
        #                             stats=statistics,
        #                             raster_out=True,
        #                             all_touched=True,
        #                             geojson_out=keep_spatial,
        #                             band=1)
        # # Original output comes inside a list containing only the output dict:
        # return roi_stats[0]
        roi_stats = zonal_stats(shp_poly,
                                tif_file,
                                stats=statistics,
                                raster_out=True,
                                all_touched=True,
                                geojson_out=keep_spatial,
                                band=1)
        # Original output comes inside a list containing only the output dict:
        return roi_stats[0]

    @staticmethod
    def extract_px(rasterio_rast, shapefile, rrs_dict, bands):
        """
        Given a dict of Rrs and a polygon, extract the values of pixels from each band

        Parameters
        ----------
        rasterio_rast: a rasterio raster open with rasterio.open
        shapefile: a polygon opened as geometry using fiona
        rrs_dict: a dict containing the Rrs bands
        bands: an array containing the bands of the rrs_dict to be extracted

        Returns
        -------
        values: an array of dimension n, where n is the number of bands, containing the values inside the polygon
        slice: the slice of the polygon (from the rasterio window)
        mask_image: the rasterio mask
        """
        # rast = rasterio_rast.read(1)
        mask_image, _, window_image = rasterio.mask.raster_geometry_mask(rasterio_rast, [shapefile], crop=True)
        slices = window_image.toslices()
        values = []
        for band in bands:
            # subsetting the xarray dataset
            subset_data = rrs_dict[band].isel(x=slices[1], y=slices[0])
            # Extract values where mask_image is False
            values.append(subset_data.where(~mask_image).values.flatten())

        return values, slices, mask_image

    @staticmethod
    def extract_function_px(rasterio_rast, shapefiles, data_matrix, fun='median', min_px=6):
        """
        Given a numpy matrix of data with the size and projection of a TIFF file opened with rasterio and a polygon,
        to extract the values of pixels for each shapefile and returns the values for the desired function

        Parameters
        ----------
        @rasterio_rast: a rasterio raster open with rasterio.open
        @shapefiles: set of polygons opened as geometry using fiona
        @data_matrix: a numpy array with the size and projection of rasterio_rast, with the values to extract
        @fun: the function to calculate over the data_matrix. Can be one of min, mean, max, median, and std
        @min_px: minimum number of pixels in each polygon to operate the classification

        Returns
        -------
        @return values_shp: an array with the same length as the shapefiles, with the calculated function for each polygon
        """

        if fun == 'median':
            calc = lambda x: np.nanmedian(x)
        elif fun == 'mean':
            calc = lambda x: np.nanmean(x)
        values_shp = np.zeros((len(shapefiles)), dtype='float32')
        for i, shape in enumerate(shapefiles):
            # extracting the data_matrix by the shapefile
            mask_image, _, window_image = rasterio.mask.raster_geometry_mask(rasterio_rast, [shape], crop=True)
            slices = window_image.toslices()
            values = data_matrix[slices[0], slices[1]][~mask_image]
            # Verifying if there are enough pixels to calculate
            valid_pixels = np.isnan(values) == False
            if np.count_nonzero(valid_pixels) >= min_px:
                values_shp[i] = calc(values)
            else:
                values_shp[i] = np.nan

        return values_shp

    def _sam(self, rrs, single=False, mode='B1'):
        """
        Spectral Angle Mapper for OWT classification for a set of pixels
        It calculates the angle between the Rrs of a set of pixels and those of the 13 OWT of inland waters
            (Spyrakos et al., 2018)
        Input values are the values of the pixels from B1 or B2 (depends on mode) to B7, the dict of the OWTs is already
            stored
        Returns the spectral angle between the Rrs of the pixels and each OWT
        To classify pixels individually, set single=True
        ----------
        """
        if single:
            E = rrs / rrs.sum()
        else:
            med = np.nanmedian(rrs, axis=1)
            E = med / med.sum()
        # norm of the vector
        nE = np.linalg.norm(E)

        # Convert OWT values to numpy array for vectorized computations
        if mode == 'B1':
            M = np.array([list(val.values()) for val in self.owts_B1_7.values()])
        else:
            M = np.array([list(val.values()) for val in self.owts_B2_7.values()])
        nM = np.linalg.norm(M, axis=1)

        # scalar product
        num = np.dot(M, E)
        den = nM * nE

        angles = np.arccos(num / den)

        return angles

    def _euclid_dist(self, rrs_px, rrs_owt, mode='B1'):
        """
        Spectral Angle Mapper for OWT classification for a set of pixels
        It calculates the angle between the Rrs of a set of pixels and those of the 13 OWT of inland waters
            (Spyrakos et al., 2018)
        Input values are the values of the pixels from B1 or B2 (depends on mode) to B7, the dict of the OWTs is already
            stored
        Returns the spectral angle between the Rrs of the pixels and each OWT
        To classify pixels individually, set single=True
        ----------
        """
        # normalising the pixel reflectance
        nE = rrs_px / rrs_px.sum()

        # normalising the OWT reflectance
        nM = rrs_owt / rrs_owt.sum()

        # Euclidean distance
        angle = np.linalg.norm(nE - nM)

        return angle

    def classify_owt_px(self, rrs_dict, B1=True):
        """
        Classify the OWT of each pixel

        Parameters
        ----------
        rrs_dict: a xarray Dataset containing the Rrs bands
        B1: boolean to whether or not use Band 1 when using Sentinel-2 data

        Returns
        -------
        class_px: an array, with the same size as the input bands, with the pixels classified with the smallest SAM
        angles: a 2-dimensional array, where the first dimension is the number of valid pixels in the rrs_dict, and
        the second dimension is the number of OWTs. The values are the spectral angles between the Rrs and the mean Rrs
        of each OWT, in each pixel

        """
        if B1:
            bands = ['Rrs_B1', 'Rrs_B2', 'Rrs_B3', 'Rrs_B4', 'Rrs_B5', 'Rrs_B6', 'Rrs_B7']
            mode = 'B1'
        else:
            bands = ['Rrs_B2', 'Rrs_B3', 'Rrs_B4', 'Rrs_B5', 'Rrs_B6', 'Rrs_B7']
            mode = 'B2'
        # Drop the variables that won't be used in the classification
        variables_to_drop = [var for var in rrs_dict.variables if var not in bands + ['x', 'y']]
        rrs = rrs_dict.drop_vars(variables_to_drop)
        # Find non-NaN values across all bands
        nzero = np.where(~np.any([np.isnan(rrs[var]) for var in bands], axis=0))
        # array of OWT class for each pixel
        class_px = np.zeros_like(rrs[bands[0]], dtype='uint8')
        # array of angles to limit the loop
        angles = np.zeros((len(nzero[0]), len(self.owts_B1_7)), dtype='float16')

        # creating a new Band 1 by undoing the upsampling of GRS, keeping only the pixels entirely inside water
        if B1:
            aux = rrs['Rrs_B1'].coarsen(x=3, y=3).mean(skipna=False).interp(x=rrs.x, y=rrs.y, method='nearest').values
            rrs['Rrs_B160m'] = (('x', 'y'), aux)
            bands = ['Rrs_B160m', 'Rrs_B2', 'Rrs_B3', 'Rrs_B4', 'Rrs_B5', 'Rrs_B6', 'Rrs_B7']

        # loop over each nonzero value in the rrs_dict
        pix = np.zeros((len(nzero[0]), len(bands)))
        for i in range(len(bands)):
            pix[:, i] = rrs[bands[i]].values[nzero]
        for i in range(len(nzero[0])):
            angles[i, :] = self._sam(rrs=pix[i, :], mode=mode, single=True)

        # if B1 is being used, there won't be any classification for the pixels without values in B1, so they have to be
        # classified using only bands 2 to 7
        if B1:
            nodata = np.where(np.isnan(angles[:, 0]))[0]
            for i in range(len(nodata)):
                angles[nodata[i], :] = self._sam(rrs=pix[nodata[i], 1:], mode='B2', single=True)
        class_px[nzero] = np.nanargmin(angles, axis=1) + 1

        return class_px, angles

    def classify_owt_spm_px(self, rrs_dict, B1=True):
        """
        Classify the OWT of each pixel based on the SPM optical water classes (Codeiro, 2022)
        It is based on the minimum Euclidean distance between the spectra of each pixel and the 4 classes

        Parameters
        ----------
        rrs_dict: a xarray Dataset containing the Rrs bands
        B1: boolean to whether or not use Band 1 when using Sentinel-2 data

        Returns
        -------
        class_px: an array, with the same size as the input bands, with the pixels classified with the smallest
        Euclidean distance
        angles: a 2-dimensional array, where the first dimension is the number of valid pixels in the rrs_dict, and
        the second dimension is the number of OWTs. The values are the Euclidean distances between the Rrs and the mean
        Rrs of each OWT, in each pixel

        """
        if B1:
            bands = ['Rrs_B1', 'Rrs_B2', 'Rrs_B3', 'Rrs_B4', 'Rrs_B5', 'Rrs_B6', 'Rrs_B7', 'Rrs_B8', 'Rrs_B8A']
            mode = 'B1'
        else:
            bands = ['Rrs_B2', 'Rrs_B3', 'Rrs_B4', 'Rrs_B5', 'Rrs_B6', 'Rrs_B7', 'Rrs_B8', 'Rrs_B8A']
            mode = 'B2'
        # Drop the variables that won't be used in the classification
        variables_to_drop = [var for var in rrs_dict.variables if var not in bands + ['x', 'y']]
        rrs = rrs_dict.drop_vars(variables_to_drop)
        # Find non-NaN values across all bands
        nzero = np.where(~np.any([np.isnan(rrs[var]) for var in bands], axis=0))
        # array of OWT class for each pixel
        class_px = np.zeros_like(rrs[bands[0]], dtype='uint8')
        # array of angles to limit the loop
        angles = np.zeros((len(nzero[0]), len(self.owts_spm_B1_8A)), dtype='float16')

        # creating a new Band 1 by undoing the upsampling of GRS, keeping only the pixels entirely inside water
        if B1:
            aux = rrs['Rrs_B1'].coarsen(x=3, y=3).mean(skipna=False).interp(x=rrs.x, y=rrs.y, method='nearest').values
            rrs['Rrs_B160m'] = (('x', 'y'), aux)
            bands = ['Rrs_B160m', 'Rrs_B2', 'Rrs_B3', 'Rrs_B4', 'Rrs_B5', 'Rrs_B6', 'Rrs_B7', 'Rrs_B8', 'Rrs_B8A']

        # loop over each nonzero value in the rrs_dict
        pix = np.zeros((len(nzero[0]), len(bands)))
        for i in range(len(bands)):
            pix[:, i] = rrs[bands[i]].values[nzero]
        # array of values of the OWTs
        if B1:
            M = np.array([list(val.values()) for val in self.owts_spm_B1_8A.values()])
        else:
            M = np.array([list(val.values()) for val in self.owts_spm_B2_8A.values()])
        for i in range(len(nzero[0])):
            for j in range(len(M)):
               # angles[i, j] = self._euclid_dist(pix[i, :], mode=mode)
               angles[i, j] = np.linalg.norm(pix[i, :] - M[j])

        # if B1 is being used, there won't be any classification for the pixels without values in B1, so they have to be
        # classified using only bands 2 to 7
        if B1:
            nodata = np.where(np.isnan(angles[:, 0]))[0]
            M = np.array([list(val.values()) for val in self.owts_spm_B2_8A.values()])
            for i in range(len(nodata)):
                for j in range(len(M)):
                    # angles[nodata[i], j] = self._euclid_dist(pix[i, :], mode='B2')
                    angles[nodata[i], j] = np.linalg.norm(pix[nodata[i], 1:] - M[j])
        class_px[nzero] = np.nanargmin(angles, axis=1) + 1

        return class_px, angles

    def classify_owt_shp(self, rasterio_rast, shapefiles, rrs_dict, B1=True, min_px=9):
        """
        Classify the OWT of pixels inside a shapefile (or a set of shapefiles)

        Parameters
        ----------
        rasterio_rast: a rasterio raster with the same configuration as the bands, open with rasterio.open
        shapefiles: a polygon (or set of polygons), usually of waterbodies to be classified, opened as geometry
            using fiona
        rrs_dict: a xarray Dataset containing the Rrs bands
        B1: boolean to use Band 1 when using Sentinel-2 data
        min_px: minimum number of pixels in each polygon to operate the classification

        Returns
        -------
        class_spt: an array, with the same size as the input bands, with the classified pixels
        class_shp: an array with the same length as the shapefiles, with a OWT class for each polygon
        """
        # checking if B1 will be used in the classification
        if B1:
            bands = ['Rrs_B1', 'Rrs_B2', 'Rrs_B3', 'Rrs_B4', 'Rrs_B5', 'Rrs_B6', 'Rrs_B7']
            mode = 'B1'
        else:
            bands = ['Rrs_B2', 'Rrs_B3', 'Rrs_B4', 'Rrs_B5', 'Rrs_B6', 'Rrs_B7']
            mode = 'B2'
        class_spt = np.zeros(rrs_dict[bands[0]].shape, dtype='int32')
        class_shp = np.zeros((len(shapefiles)), dtype='int32')
        for i, shape in enumerate(shapefiles):
            values, slices, mask = self.extract_px(rasterio_rast, shape, rrs_dict, bands)
            # Verifying if there are more pixels than the minimum
            valid_pixels = np.isnan(values[0]) == False
            if np.count_nonzero(valid_pixels) >= min_px:
                angle = int(np.argmin(self._sam(values, mode=mode)) + 1)
            else:
                angle = int(0)

            # classifying only the valid pixels inside the polygon
            values = np.where(valid_pixels, angle, 0)
            # adding to avoid replacing values of cropping by other polygons
            class_spt[slices[0], slices[1]] += values.reshape(mask.shape)
            # classification by polygon
            class_shp[i] = angle

        return class_spt.astype('uint8'), class_shp.astype('uint8')

    def classify_owt_weights(self, class_px, angles, n=3, remove_classes=1):
        """
        Attribute weights to the n-th most important OWTs, based on the spectral angle mapper
        The weights are used for calculating weighted means of the water quality parameters, in order to smooth the
        spatial differences between the pixels, and also to remove possible outliers generated by some models
        For more information on this approach, please refer to Moore et al. (2001) and Liu et al. (2021)

        This function uses the results of classify_owt_px as input data

        Parameters
        ----------
        class_px: an array, with the same size as the input bands, with the pixels classified with the smallest SAM
        angles: a 2-dimensional array, where the first dimension is the number of valid pixels in the rrs_dict, and
        the second dimension is the number of OWTs. The values are the spectral angles between the Rrs and the mean Rrs
        of each OWT, in each pixel
        n = the number of dominant classes to be used to generate the weights
        remove_classes: int or a list of OWT classes to be removed (pixel-wise)

        Returns
        -------
        owt_classes: an array, with the same size as the input bands, with the n classes of pixels (first dimension)
        owt_weights: an array, with the same size as the input bands, with the n weights (first dimension)
        """
        # creating the variables of weights and classes for each pixel, depending on the desired number of classes to
        # be used
        owt_weights = np.zeros((n, *class_px.shape), dtype='float32')
        owt_classes = np.zeros((n, *class_px.shape), dtype='float32')

        # finding where there is no valid pixels in the reflectance data
        nzero = np.where(class_px != 0)

        # Create an array of indices
        indices = np.argsort(angles, axis=1)
        lowest_angles = np.take_along_axis(angles, indices[:,0:(n+1):], axis=1)

        # Calculating the weights based on normalisation of the n+1 lowest spectral angles (parallel of convertion of
        # units) and assigning values to the matrices
        for i in range(n):
            owt_classes[i, nzero[0], nzero[1]] = indices[:, i] + 1  # summing one due to positioning starting in 1
            for j in range(nzero[0].shape[0]):
                owt_weights[i, nzero[0][j], nzero[1][j]] = (lowest_angles[j, i] - lowest_angles[j, -1]) / (
                            lowest_angles[j, 0] - lowest_angles[j, -1])

        # removing undesidered OWTs:
        if isinstance(remove_classes, int):
            ones = np.where(indices[:, 0] == (remove_classes-1))[0]
            owt_weights[:, nzero[0][ones], nzero[1][ones]] = np.nan
        elif isinstance(remove_classes, list):
            for i in range(len(remove_classes)):
                ones = np.where(indices[:, 0] == (remove_classes[i]-1))[0]
                owt_weights[:, nzero[0][ones], nzero[1][ones]] = np.nan

        # # removing the zeros to avoid division by 0
        # owt_weights[np.where(owt_weights == 0)] = np.nan
        # owt_classes[np.where(owt_classes == 0)] = np.nan

        return owt_classes, owt_weights

    def cdom(self, rrs_dict, class_owt_spt, upper_lim=50, lower_lim=0):
        """
        Calculate the coloured dissolved organic matter (CDOM) based on the optical water type (OWT)

        Parameters
        ----------
        rrs_dict: rrs_dict: a xarray Dataset containing the Rrs bands
        class_owt_spt: an array, with the same size as the input bands, with the OWT pixels

        Returns
        -------
        cdom: an array, with the same size as the input bands, with the modeled values
        """
        import getpak.inversion_functions as ifunc
        cdom = np.zeros(rrs_dict['Rrs_B4'].shape, dtype='float32')

        # create a matrix for the outliers
        if not hasattr(self, 'out'):
            self.out = np.zeros(rrs_dict['Rrs_B4'].shape, dtype='uint8')

        # spm functions for each OWT
        classes = [1, 4, 5, 6, 2, 7, 8, 11, 12, 3, 9, 10, 13]
        index = np.where(np.isin(class_owt_spt, classes))
        if len(index[0] > 0):
            cdom[index] = ifunc.cdom_brezonik(Blue=rrs_dict['Rrs_B2'].values[index],
                                                                       RedEdg2=rrs_dict['Rrs_B6'].values[index])

        # removing espurious values
        if isinstance(upper_lim, (int, float)) and isinstance(lower_lim, (int, float)):
            out = np.where((cdom < lower_lim) | (cdom > upper_lim))
            cdom[out] = np.nan
            self.out[out] = 1

        out = np.where((cdom == 0) | np.isinf(cdom))
        cdom[out] = np.nan

        return cdom

    def chlorophylla(self, rrs_dict, class_owt_spt, limits=True, alg='owt'):
        """
        Function to calculate the chlorophyll-a concentration (chla) based on the optical water type (OWT)
        The functions are the ones recomended by Carrea et al. (2023) and Neil et al. (2019, 2020), and are coded in
        inversion_functions.py

        Parameters
        ----------
        rrs_dict: rrs_dict: a xarray Dataset containing the Rrs bands
        class_owt_spt: an array, with the same size as the input bands, with the OWT pixels
        limits: boolean to choose whether to apply the algorithms only in their limits of cal/val
        alg: one the the following algorithms available: owt to use the methodology based on OWT, or gons

        Returns
        -------
        chla: an array, with the same size as the input bands, with the modeled values
        """
        import getpak.inversion_functions as ifunc
        chla = np.zeros(rrs_dict['Rrs_B4'].shape, dtype='float32')

        # create a matrix for the outliers if it doesn't exist
        if not hasattr(self, 'out'):
            self.out = np.zeros(rrs_dict['Rrs_B4'].shape, dtype='uint8')

        if alg == 'owt':
            # chla functions for each OWT
            # for class 1, using the coefficients calibrated by Neil et al. (2020) for the OWT 1
            # classes = [1]
            # index = np.where(np.isin(class_owt_spt, classes))
            # if len(index[0] > 0):
            #     chla[index] = ifunc.chl_gurlin(Red=rrs_dict['Rrs_B4'].values[index],
            #                                                             RedEdg1=rrs_dict['Rrs_B5'].values[index],
            #                                                             a=86.09, b=-517.5, c=886.7)
            #     if limits:
            #         lims = [10, 1000]
            #         out = np.where((chla[index] < lims[0]) | (chla[index] > lims[1]))
            #         chla[index[0][out], index[1][out]] = np.nan
            #         self.out[index[0][out], index[1][out]] += 1

            classes = [1, 6, 10]
            index = np.where(np.isin(class_owt_spt, classes))
            if len(index[0] > 0):
                chla[index] = ifunc.chl_gons(Red=rrs_dict['Rrs_B4'].values[index],
                                                                      RedEdg1=rrs_dict['Rrs_B5'].values[index],
                                                                      RedEdg3=rrs_dict['Rrs_B7'].values[index],
                                                                      aw665=0.425, aw708=0.704)
                if limits:
                    lims = [1, 250]
                    out = np.where((chla[index] < lims[0]) | (chla[index] > lims[1]))
                    chla[index[0][out], index[1][out]] = np.nan
                    self.out[index[0][out], index[1][out]] += 1

            classes = [2, 4, 5, 11, 12]
            index = np.where(np.isin(class_owt_spt, classes))
            if len(index[0] > 0):
                chla[index] = ifunc.chl_ndci(Red=rrs_dict['Rrs_B4'].values[index],
                                                                           RedEdg1=rrs_dict['Rrs_B5'].values[index])
                if limits:
                    lims = [5, 250]
                    out = np.where((chla[index] < lims[0]) | (chla[index] > lims[1]))
                    chla[index[0][out], index[1][out]] = np.nan
                    self.out[index[0][out], index[1][out]] += 1

            # for class 2 and 12, when values of NDCI are >20, use Gilerson instead
            classes = [2, 12]
            conditions = (np.isin(class_owt_spt, classes)) & (chla > 20)
            index = np.where(conditions)
            if len(index[0] > 0):
                chla[index] = ifunc.chl_gilerson2(Red=rrs_dict['Rrs_B4'].values[index],
                                                                      RedEdg1=rrs_dict['Rrs_B5'].values[index])
                if limits:
                    lims = [5, 500]
                    out = np.where((chla[index] < lims[0]) | (chla[index] > lims[1]))
                    chla[index[0][out], index[1][out]] = np.nan
                    self.out[index[0][out], index[1][out]] += 1

            classes = [7, 8]
            index = np.where(np.isin(class_owt_spt, classes))
            if len(index[0] > 0):
                chla[index] = ifunc.chl_gilerson2(Red=rrs_dict['Rrs_B4'].values[index],
                                                                      RedEdg1=rrs_dict['Rrs_B5'].values[index])
                if limits:
                    lims = [5, 500]
                    out = np.where((chla[index] < lims[0]) | (chla[index] > lims[1]))
                    chla[index[0][out], index[1][out]] = np.nan
                    self.out[index[0][out], index[1][out]] += 1

            # classes = []
            # index = np.where(np.isin(class_owt_spt, classes))
            # if len(index[0] > 0):
            #     chla[index] = ifunc.chl_gilerson3(Red=rrs_dict['Rrs_B4'].values[index],
            #                                                                RedEdg1=rrs_dict['Rrs_B5'].values[index],
            #                                                                RedEdg2=rrs_dict['Rrs_B6'].values[index])
            #     if limits:
            #         lims = [10, 1000]
            #         out = np.where((chla[index] < lims[0]) | (chla[index] > lims[1]))
            #         chla[index[0][out], index[1][out]] = np.nan
            #         self.out[index[0][out], index[1][out]] += 1

            classes = [3]
            index = np.where(np.isin(class_owt_spt, classes))
            if len(index[0] > 0):
                chla[index] = ifunc.chl_OC2(Blue=rrs_dict['Rrs_B2'].values[index],
                                            Green=rrs_dict['Rrs_B3'].values[index], a=0.1098, b=-0.755, c=-14.12,
                                            d=-117, e=-17.76)
                if limits:
                    lims = [0.01, 50]
                    out = np.where((chla[index] < lims[0]) | (chla[index] > lims[1]))
                    chla[index[0][out], index[1][out]] = np.nan
                    self.out[index[0][out], index[1][out]] += 1

            classes = [9]
            index = np.where(np.isin(class_owt_spt, classes))
            if len(index[0] > 0):
                chla[index] = ifunc.chl_OC2(Blue=rrs_dict['Rrs_B2'].values[index],
                                            Green=rrs_dict['Rrs_B3'].values[index], a=0.0536, b=7.308, c=116.2,
                                            d=412.4, e=463.5)
                if limits:
                    lims = [0.01, 50]
                    out = np.where((chla[index] < lims[0]) | (chla[index] > lims[1]))
                    chla[index[0][out], index[1][out]] = np.nan
                    self.out[index[0][out], index[1][out]] += 1

            classes = [13]
            index = np.where(np.isin(class_owt_spt, classes))
            if len(index[0] > 0):
                chla[index] = ifunc.chl_OC2(Blue=rrs_dict['Rrs_B2'].values[index],
                                            Green=rrs_dict['Rrs_B3'].values[index], a=-5020, b=2.9e+04, c=-6.1e+04,
                                            d=5.749e+04, e=-2.026e+04)
                if limits:
                    lims = [0.01, 50]
                    out = np.where((chla[index] < lims[0]) | (chla[index] > lims[1]))
                    chla[index[0][out], index[1][out]] = np.nan
                    self.out[index[0][out], index[1][out]] += 1

        else:
            chla = ifunc.chl_gons(Red=rrs_dict['Rrs_B4'].values, RedEdg1=rrs_dict['Rrs_B5'].values,
                                  RedEdg3=rrs_dict['Rrs_B7'].values)
            if limits:
                lims = [1, 250]
                out = np.where((chla < lims[0]) | (chla > lims[1]))
                chla[out] = np.nan
                self.out[out] += 1

        # removing espurious values and zeros
        out = np.where((chla == 0) | np.isinf(chla))
        chla[out] = np.nan

        return chla

    def spm(self, rrs_dict, class_owt_spt, alg='owt', limits=True, mode_Jiang=None, rasterio_rast=None, shapefile=None,
            min_px=9):
        """
        Function to calculate the suspended particulate matter (SPM) based on the optical water type (OWT)

        Parameters
        ----------
        rrs_dict: rrs_dict: a xarray Dataset containing the Rrs bands
        class_owt_spt: an array, with the same size as the input bands, with the OWT pixels
        alg: one of the following algorithms available: owt to use the methodology based on OWT, Hybrid, Nechad,
            NechadGreen, Binding, Zhang, Dogliotti, Condé or different versions of Jiang
        limits: boolean to choose whether to apply the algorithms only in their limits of cal/val
        mode_Jiang: used only for the general Jiang algorithm, to choose from pixel-wise or lake-wise calculation
        rasterio_rast: used only for the general Jiang algorithm, a rasterio raster with the same configuration as the
            bands, open with rasterio.open
        shapefile: used only for the general Jiang algorithm, a polygon (or set of polygons), usually of waterbodies to
            be classified, opened as geometry using fiona
        min_px: used only for the general Jiang algorithm, minimum number of pixels in each polygon to operate the
            inversion

        Returns
        -------
        spm: an array, with the same size as the input bands, with the modeled values
        """
        import getpak.inversion_functions as ifunc
        spm = np.zeros(rrs_dict['Rrs_B4'].shape, dtype='float32')

        # create a matrix for the outliers
        if not hasattr(self, 'out'):
            self.out = np.zeros(rrs_dict['Rrs_B4'].shape, dtype='uint8')

        # spm functions for each OWT
        classes = [1, 2, 3, 4]
        index = np.where(np.isin(class_owt_spt, classes))
        if len(index[0] > 0):
            if alg == 'owt':
                classes = [1]
                index = np.where(np.isin(class_owt_spt, classes))
                if len(index[0] > 0):
                    spm[index] = ifunc.spm_jiang2021_green(Aerosol=rrs_dict['Rrs_B1'].values[index],
                                                           Blue=rrs_dict['Rrs_B2'].values[index],
                                                           Green=rrs_dict['Rrs_B3'].values[index],
                                                           Red=rrs_dict['Rrs_B4'].values[index])
                    if limits:
                        lims = [0, 50]
                        out = np.where((spm[index] < lims[0]) | (spm[index] > lims[1]))
                        spm[index[0][out], index[1][out]] = np.nan
                        self.out[index[0][out], index[1][out]] += 1

                classes = [2]
                index = np.where(np.isin(class_owt_spt, classes))
                if len(index[0] > 0):
                    spm[index] = ifunc.spm_jiang2021_red(Aerosol=rrs_dict['Rrs_B1'].values[index],
                                                         Blue=rrs_dict['Rrs_B2'].values[index],
                                                         Green=rrs_dict['Rrs_B3'].values[index],
                                                         Red=rrs_dict['Rrs_B4'].values[index])
                    if limits:
                        lims = [10, 500]
                        out = np.where((spm[index] < lims[0]) | (spm[index] > lims[1]))
                        spm[index[0][out], index[1][out]] = np.nan
                        self.out[index[0][out], index[1][out]] += 1

                classes = [3]
                index = np.where(np.isin(class_owt_spt, classes))
                if len(index[0] > 0):
                    spm[index] = ifunc.spm_zhang2014(RedEdge1=rrs_dict['Rrs_B5'].values[index])

                    if limits:
                        lims = [20, 1000]
                        out = np.where((spm[index] < lims[0]) | (spm[index] > lims[1]))
                        spm[index[0][out], index[1][out]] = np.nan
                        self.out[index[0][out], index[1][out]] += 1

                classes = [4]
                index = np.where(np.isin(class_owt_spt, classes))
                if len(index[0] > 0):
                    spm[index] = ifunc.spm_binding2010(RedEdge2=rrs_dict['Rrs_B6'].values[index])

                    if limits:
                        lims = [50, 2000]
                        out = np.where((spm[index] < lims[0]) | (spm[index] > lims[1]))
                        spm[index[0][out], index[1][out]] = np.nan
                        self.out[index[0][out], index[1][out]] += 1

            elif alg == 'Hybrid':
                spm[index] = ifunc.spm_s3(Red=rrs_dict['Rrs_B4'].values[index],
                                          Nir2=rrs_dict['Rrs_B8A'].values[index])
            elif alg == 'Nechad':
                spm[index] = ifunc.spm_nechad(Red=rrs_dict['Rrs_B4'].values[index])

            elif alg == 'NechadGreen':
                spm[index] = ifunc.spm_nechad(Red=rrs_dict['Rrs_B3'].values[index], a=228.72, c=0.2200)

            elif alg == 'Binding':
                spm[index] = ifunc.spm_binding2010(RedEdge2=rrs_dict['Rrs_B6'].values[index])

            elif alg == 'Zhang':
                spm[index] = ifunc.spm_zhang2014(RedEdge1=rrs_dict['Rrs_B5'].values[index])

            elif alg == 'Jiang_Green':
                spm[index] = ifunc.spm_jiang2021_green(Aerosol=rrs_dict['Rrs_B1'].values[index],
                                                       Blue=rrs_dict['Rrs_B2'].values[index],
                                                       Green=rrs_dict['Rrs_B3'].values[index],
                                                       Red=rrs_dict['Rrs_B4'].values[index])

            elif alg == 'Jiang_Red':
                spm[index] = ifunc.spm_jiang2021_red(Aerosol=rrs_dict['Rrs_B1'].values[index],
                                                     Blue=rrs_dict['Rrs_B2'].values[index],
                                                     Green=rrs_dict['Rrs_B3'].values[index],
                                                     Red=rrs_dict['Rrs_B4'].values[index])
            elif alg == 'Dogliotti':
                spm[index] = ifunc.spm_dogliotti_S2(Red=rrs_dict['Rrs_B4'].values[index],
                                                     Nir2=rrs_dict['Rrs_B8A'].values[index])
            elif alg == 'Conde':
                spm[index] = ifunc.spm_conde(Red=rrs_dict['Rrs_B4'].values[index])

            elif alg == 'Jiang':
                if mode_Jiang == 'pixel':
                    spm[index] = ifunc.spm_jiang2021(Aerosol=rrs_dict['Rrs_B1'].values[index],
                                                     Blue=rrs_dict['Rrs_B2'].values[index],
                                                     Green=rrs_dict['Rrs_B3'].values[index],
                                                     Red=rrs_dict['Rrs_B4'].values[index],
                                                     RedEdge2=rrs_dict['Rrs_B6'].values[index],
                                                     Nir2=rrs_dict['Rrs_B8A'].values[index], mode=mode)
                elif mode_Jiang == 'polygon':
                    for i, shape in enumerate(shapefile):
                        values, slices, mask = self.extract_px(rasterio_rast=rasterio_rast, shapefile=shape,
                                                               rrs_dict=rrs_dict, bands=['Rrs_B1','Rrs_B2','Rrs_B3',
                                                                                         'Rrs_B4','Rrs_B6','Rrs_B8A'])
                        # Verifying if there are more pixels than the minimum
                        valid_pixels = np.isnan(values[0]) == False
                        if np.count_nonzero(valid_pixels) >= min_px:
                            out = ifunc.spm_jiang2021(Aerosol=values[0].reshape(mask.shape),
                                                      Blue=values[1].reshape(mask.shape),
                                                      Green=values[2].reshape(mask.shape),
                                                      Red=values[3].reshape(mask.shape),
                                                      RedEdge2=values[4].reshape(mask.shape),
                                                      Nir2=values[5].reshape(mask.shape), mode=mode).flatten()
                            # classifying only the valid pixels inside the polygon
                            values = np.where(valid_pixels, out, 0)
                            # adding to avoid replacing values of cropping by other polygons
                            spm[slices[0], slices[1]] += values.reshape(mask.shape)

        # removing espurious values and zeros
        out = np.where((spm == 0) | np.isinf(spm))
        spm[out] = np.nan
        self.out[out] = 1

        return spm

    def secchi_dd(self, rrs_dict, class_owt_spt, upper_lim=50, lower_lim=0):
        """
        Function to calculate the Secchi disk depth (SPM) based on the optical water type (OWT)

        Parameters
        ----------
        rrs_dict: rrs_dict: a xarray Dataset containing the Rrs bands
        class_owt_spt: an array, with the same size as the input bands, with the OWT pixels

        Returns
        -------
        secchi: an array, with the same size as the input bands, with the modeled values
        """
        import getpak.inversion_functions as ifunc
        secchi = np.zeros(rrs_dict['Rrs_B4'].shape, dtype='float32')

        # create a matrix for the outliers
        if not hasattr(self, 'out'):
            self.out = np.zeros(rrs_dict['Rrs_B4'].shape, dtype='uint8')

        # spm functions for each OWT
        classes = [1, 4, 5, 6, 2, 7, 8, 11, 12, 3, 9, 10, 13]
        index = np.where(np.isin(class_owt_spt, classes))
        if len(index[0] > 0):
            secchi[index] = ifunc.functions['SDD_Lee']['function'](Red=rrs_dict['Rrs_B4'].values[index])

        # removing espurious values and zeros
        if isinstance(upper_lim, (int, float)) and isinstance(lower_lim, (int, float)):
            out = np.where((secchi < lower_lim) | (secchi > upper_lim))
            secchi[out] = np.nan
            self.out[out] = 1

        out = np.where((secchi == 0) | np.isinf(secchi))
        secchi[out] = np.nan

        return secchi

    @staticmethod
    def water_colour(rrs_dict, bands=['Rrs_B2', 'Rrs_B3', 'Rrs_B4', 'Rrs_B5']):
        """
        Function to calculate the water colour of each pixel based on the Forel-Ule scale, using the bands of Sentinel-2
        MSI, with coefficients derived by linear correlation by van der Woerd and Wernand (2018). The different
        combinations of S2 bands (10, 20 or 60 m resolution) in the visible spectrum can be used, with the default
        being at 20 m.

        Parameters
        ----------
        rrs_dict: rrs_dict: a xarray Dataset containing the Rrs bands
        bands: an array containing the bands of the rrs_dict to be extracted, in order from blue to red edge

        Returns
        -------
        colour: an array, with the same size as the input bands, with the classified pixels
        """
        # Drop the variables that won't be used in the classification
        variables_to_drop = [var for var in rrs_dict.variables if var not in bands]
        rrs = rrs_dict.drop_vars(variables_to_drop)
        # Find non-NaN values
        nzero = np.where(~np.isnan(rrs[bands[0]].values))
        # array of classes to limit the loop
        classes = np.zeros((len(nzero[0])), dtype='uint8')
        # array of colour class for each pixel
        colour = np.zeros_like(rrs[bands[0]], dtype='uint8')

        # calculation of the CIE tristimulus (no intercepts):
        # B2 to B4
        if len(bands) == 3:
            X = (12.040 * rrs[bands[0]].values[nzero] + 53.696 * rrs[bands[1]].values[nzero] +
                 32.087 * rrs[bands[2]].values[nzero])
            Y = (23.122 * rrs[bands[0]].values[nzero] + 65.702 * rrs[bands[1]].values[nzero] +
                 16.830 * rrs[bands[2]].values[nzero])
            Z = (61.055 * rrs[bands[0]].values[nzero] + 1.778 * rrs[bands[1]].values[nzero] +
                 0.015 * rrs[bands[2]].values[nzero])
            delta = lambda d: -164.83 * (alpha / 100) ** 5 + 1139.90 * (alpha / 100) ** 4 - 3006.04 * (
                    alpha / 100) ** 3 + 3677.75 * (alpha / 100) ** 2 - 1979.71 * (alpha / 100) + 371.38
        # B2 to B5
        elif len(bands) == 4:
            X = (12.040 * rrs[bands[0]].values[nzero] + 53.696 * rrs[bands[1]].values[nzero] +
                 32.028 * rrs[bands[2]].values[nzero] + 0.529 * rrs[bands[3]].values[nzero])
            Y = (23.122 * rrs[bands[0]].values[nzero] + 65.702 * rrs[bands[1]].values[nzero] +
                 16.808 * rrs[bands[2]].values[nzero] + 0.192 * rrs[bands[3]].values[nzero])
            Z = (61.055 * rrs[bands[0]].values[nzero] + 1.778 * rrs[bands[1]].values[nzero] +
                 0.015 * rrs[bands[2]].values[nzero] + 0.000 * rrs[bands[3]].values[nzero])
            delta = lambda d: -161.23 * (alpha / 100) ** 5 + 1117.08 * (alpha / 100) ** 4 - 2950.14 * (
                    alpha / 100) ** 3 + 3612.17 * (alpha / 100) ** 2 - 1943.57 * (alpha / 100) + 364.28
        # B1 to B5
        elif len(bands) == 5:
            X = (11.756 * rrs[bands[0]].values[nzero] + 6.423 * rrs[bands[1]].values[nzero] +
                 53.696 * rrs[bands[2]].values[nzero] + 32.028 * rrs[bands[3]].values[nzero] +
                 0.529 * rrs[bands[4]].values[nzero])
            Y = (1.744 * rrs[bands[0]].values[nzero] + 22.289 * rrs[bands[1]].values[nzero] +
                 65.702 * rrs[bands[2]].values[nzero] + 16.808 * rrs[bands[3]].values[nzero] +
                 0.192 * rrs[bands[4]].values[nzero])
            Z = (62.696 * rrs[bands[0]].values[nzero] + 31.101 * rrs[bands[1]].values[nzero] +
                 1.778 * rrs[bands[2]].values[nzero] + 0.015 * rrs[bands[3]].values[nzero] +
                 0.000 * rrs[bands[4]].values[nzero])
            delta = lambda d: -65.74 * (alpha / 100) ** 5 + 477.16 * (alpha / 100) ** 4 - 1279.99 * (
                    alpha / 100) ** 3 + 1524.96 * (alpha / 100) ** 2 - 751.59 * (alpha / 100) + 116.56
        else:
            print("Error in the number of bands provided")
            colour = None
        # normalisation of the tristimulus in 2 coordinates
        x = X / (X + Y + Z)
        y = Y / (X + Y + Z)
        # hue angle:
        alpha = (np.arctan2(y - 1 / 3, x - 1 / 3)) * 180 / np.pi % 360
        # correction for multispectral information
        alpha_corrected = alpha + delta(alpha)
        for i in range(len(nzero[0])):
            classes[i] = int(Raster._Forel_Ule_scale(alpha_corrected[i]))
        # adding the classes to the matrix
        colour[nzero] = classes

        return colour

    @staticmethod
    def _Forel_Ule_scale(angle):
        """
        Calculates the Forel-Ule water colour scale, depending on the hue angle calculated from the Sentinel-2 MSI data,
        based on the classification by Novoa et al. (2013)
        Parameters
        ----------
        angle: the hue angle, in degrees

        Returns
        -------
        The water colour class (1-21)
        """
        mapping = [(21.0471, 21), (24.4487, 20), (28.2408, 19), (32.6477, 18), (37.1698, 17), (42.3707, 16),
                   (47.8847, 15), (53.4431, 14), (59.4234, 13), (64.9378, 12), (70.9617, 11), (78.1648, 10),
                   (88.5017, 9), (99.5371, 8), (118.5208, 7), (147.4148, 6), (178.7020, 5), (202.8305, 4),
                   (217.1473, 3), (224.8037, 2)]
        score = lambda s: next((L for x, L in mapping if s < x), 1)

        return score(angle)

class GRS:
    """
    Core functionalities to handle GRS files

    Methods
    -------
    metadata(grs_file_entry)
        Given a GRS string element, return file metadata extracted from its name.
    """

    def __init__(self, parent_log=None):
        if parent_log:
            self.log = parent_log
        else:
            INSTANCE_TIME_TAG = datetime.now().strftime('%Y%m%dT%H%M%S')
            logfile = os.path.join(os.getcwd(), 'getpak_raster_' + INSTANCE_TIME_TAG + '.log')
            self.log = u.create_log_handler(logfile)

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
            cc, ver = ['NA', 'v20']
        file_event_date = datetime.strptime(date_n_time, '%Y%m%dT%H%M%S')
        yyyy = f"{file_event_date.year:02d}"
        mm = f"{file_event_date.month:02d}"
        dd = f"{file_event_date.day:02d}"

        metadata['input_file'] = grs_file_entry
        metadata['basename'] = basefile
        metadata['mission'] = mission
        metadata['prod_lvl'] = proc_level
        metadata['str_date'] = date_n_time
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
        @grs_version: a string with GRS version ('v15' or 'v20' for version 2.0.5+)

        Returns
        -------
        @return grs: xarray.DataArray containing 11 Rrs bands.
        The band names can be found at getpak.commons.DefaultDicts.grs_v20nc_s2bands
        """
        # list of bands
        bands = list(d.grs_v20nc_s2bands.keys())
        self.log.info(f'Opening GRS version {grs_version} file {grs_nc_file}')
        if grs_version == 'v15':
            ds = xr.open_dataset(grs_nc_file, engine="netcdf4", decode_coords='all', chunks={'y': 610, 'x': 610})
            # List of variables to keep
            if 'Rrs_B1' in ds.variables:
                variables_to_keep = bands
                # Drop the variables you don't want
                variables_to_drop = [var for var in ds.variables if var not in variables_to_keep]
                grs = ds.drop_vars(variables_to_drop)
        elif grs_version == 'v20':
            ds = xr.open_dataset(grs_nc_file, chunks={'y': -1, 'x': -1}, engine="netcdf4")
            waves = ds['Rrs']['wl']
            subset_dict = {band: ds['Rrs'].sel(wl=waves[i]).drop(['wl']) for i, band in enumerate(bands)}
            grs = xr.Dataset(subset_dict)
        else:
            self.log.error(f'GRS version {grs_version} not supported.')
            grs = None
            sys.exit(1)
        ds.close()
        return grs

    def param2tiff(self, ndarray_data, img_ref, output_img, no_data=0, gdal_driver_name="GTiff"):

        # Gather information from the template file
        ref_data = gdal.Open(img_ref)
        trans = ref_data.GetGeoTransform()
        proj = ref_data.GetProjection()
        # nodatav = 0 #data.GetNoDataValue()
        # Create file using information from the template
        outdriver = gdal.GetDriverByName(gdal_driver_name)  # http://www.gdal.org/gdal_8h.html

        [cols, rows] = ndarray_data.shape

        print(f'Writing output .tiff')
        # GDT_Byte = 1, GDT_UInt16 = 2, GDT_UInt32 = 4, GDT_Int32 = 5, GDT_Float32 = 6,
        # options=['COMPRESS=PACKBITS'] -> https://gdal.org/drivers/raster/gtiff.html#creation-options
        outdata = outdriver.Create(output_img, rows, cols, 1, gdal.GDT_Float32, options=['COMPRESS=PACKBITS'])
        # Write the array to the file, which is the original array in this example
        outdata.GetRasterBand(1).WriteArray(ndarray_data)
        # Set a no data value if required
        outdata.GetRasterBand(1).SetNoDataValue(no_data)
        # Georeference the image
        outdata.SetGeoTransform(trans)
        # Write projection information
        outdata.SetProjection(proj)

        # Closing the files
        # https://gdal.org/tutorials/raster_api_tut.html#using-create
        # data = None
        outdata = None
        self.log.info('')
        pass
    
    def _get_shp_features(self, shp_file, unique_key='id', grs_crs='EPSG:32720'):
        '''
        INTERNAL FUNCTION
        Given a shp_file.shp, read each feature by unique 'id' and save it in a dict.
        OBS1: Shp must have a unique key to identify each feature (call it 'id').
        OBS2: The crs of the shapefile must be the same as the crs of the raster (EPSG:32720).
        '''
        try:
            self.log.info(f'Reading shapefile: {shp_file}')
            shp = gpd.read_file(shp_file)
            self.log.info(f'CRS: {str(shp.crs)}')
            if str(shp.crs) != grs_crs:
                self.log.info(f'CRS mismatch, expected {grs_crs}.')
                self.log.info(f'Program may work incorrectly.')
            # Initialize a dict to handle all the geodataframes by id
            feature_dict = {}
            # Iterate over all features in the shapefile and populate the dict
            for id in shp.id.values:
                shp_feature = shp[shp[unique_key] == id]
                feature_dict[id] = shp_feature
            
            self.log.info(f'Done. {len(feature_dict)} features found.')
        
        except Exception as e:
            self.log.error(f'Error: {e}')
            sys.exit(1)

        return feature_dict

    # Internal function to paralelize the process of each band and point
    def _process_band_point(self, band, shp_feature, crs, pt_id):
        """
        INTERNAL FUNCTION
        """
        Rrs_raster_band = self.grs_dict[band]
        Rrs_raster_band.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        clipped_rrs = Rrs_raster_band.rio.clip(shp_feature.geometry.values, crs)
        global counter
        if counter%10==0:
            print(f'> {counter}')
        else:
            print('>', end='')
        counter += 1
        mean_rrs = clipped_rrs.mean().compute().data.item()
        return (band, pt_id, mean_rrs)
    
    def sample_grs_with_shp(self, grs_nc, vector_shp, grs_version='v20', unique_shp_key='id'):
        '''
        Given a GRS.nc file and a multi-feature vector.shp, extract the Rrs values for all GRS bands that
        fall inside the vector.shp features.
        
        Parameters
        ----------
        @grs_nc: the path to the GRS file
        @vector_shp: the path to the shapefile
        @grs_version: a string with GRS version ('v15' or 'v20' for version 2.0.5+)
        @unique_shp_key: A unique key/column to identify each feature in the shapefile

        Returns
        -------
        @return df: pd.DataFrame containing the mean Rrs values 
        for each band and feature in the input shapefile.
        '''
        try:
            # No need to print or log, the functions should handle it internally.
            self.grs_dict = self.get_grs_dict(grs_nc, grs_version)
            self.gpd_feature_dict = self._get_shp_features(vector_shp, unique_key=unique_shp_key)

        except Exception as e:
            self.log.error(f'Error: {e}')
            sys.exit(1) # Exit program: 0 is success, 1 is failure
        
        # Initialize the dict and fill it with zeros
        # This will be converted to a pd.DataFrame later
        pt_stats = {}
        for band in d.grs_v20nc_s2bands.keys():
            pt_stats[d.grs_v20nc_s2bands[band]] = {id:0.0 for id in self.gpd_feature_dict.keys()}
        
        # Declare a list to hold delayed tasks
        tasks = []

        # Create delayed tasks for each combination of band and shapefile point
        for band in d.grs_v20nc_s2bands.keys():
            for id in self.gpd_feature_dict.keys():
                # Get the shape feature
                shp_feature = self.gpd_feature_dict[id]
                # Create a delayed task
                task = delayed(self._process_band_point)(band, shp_feature, shp_feature.crs, id)
                tasks.append(task)
        
        global counter
        counter = 0
        print(f'Processing {len(tasks)} tasks...')
        # Compute all delayed tasks
        results = dask.compute(*tasks)
        print(f' {counter}')
        print('Done.')

        # Aggregate results into pt_stats
        for band, pt_id, mean_rrs in results:
            pt_stats[d.grs_v20nc_s2bands[band]][pt_id] = mean_rrs
        
        df = pd.DataFrame(pt_stats)
        return df

class S3_engine:
    """
    Provide methods to manipulate NetCDF4 data from Sentinel-3 OLCI products.
    
    Input
    -----
    @input_nc_folder: string path to the .SEN3 folder containing .nc files
    @parent_log: logger object from the parent class (internal use)

    Methods
    -------
    TO-DO

    Parameters
    ----------
    @log : logger object
    @nc_base_name: string containing the base name of the .SEN3 folder
    @product: string containing the product type (WFR/EFR/SYN)
    @netcdf_valid_band_list: list of valid bands in the .SEN3 folder
    @g_lat: 2D array containing the latitude values
    @g_lon: 2D array containing the longitude values
    @t_lat: 2D array containing the tie latitude values
    @t_lon: 2D array containing the tie longitude values
    @OAA: 2D array containing the OAA values
    @OZA: 2D array containing the OZA values
    @SAA: 2D array containing the SAA values
    @SZA: 2D array containing the SZA values
    """

    def __init__(self, input_nc_folder=None, parent_log=None, product='wfr'):
        self.log = parent_log
        self.nc_folder = Path(input_nc_folder)
        self.nc_base_name = os.path.basename(input_nc_folder).split('.')[0]
        self.product = product.lower()
        self.netcdf_valid_band_list = self.get_valid_band_files(rad_only=False)

        if parent_log:
            self.log = parent_log
        else:
            INSTANCE_TIME_TAG = datetime.now().strftime('%Y%m%dT%H%M%S')
            logfile = os.path.join(os.getcwd(), 'getpak_raster_' + INSTANCE_TIME_TAG + '.log')
            self.log = u.create_log_handler(logfile)

        if self.product.lower() == 'wfr':
            self.log.info(f'{os.getpid()} - Initializing geometries for: {self.nc_base_name}')
            geo_coord = nc.Dataset(self.nc_folder / 'geo_coordinates.nc')
            self.g_lat = geo_coord['latitude'][:]
            self.g_lon = geo_coord['longitude'][:]

            # Load and resize tie LON/LAT Bands using the geo_coordinates.nc file dimensions: (4091, 4865)
            tie_geo = nc.Dataset(self.nc_folder / 'tie_geo_coordinates.nc')

            self.t_lat = tie_geo['latitude'][:]
            self.t_lat = resize(self.t_lat, (self.g_lat.shape[0], self.g_lat.shape[1]), anti_aliasing=False)
            self.t_lon = tie_geo['longitude'][:]
            self.t_lon = resize(self.t_lon, (self.g_lon.shape[0], self.g_lon.shape[1]), anti_aliasing=False)

            # Load and resize Sun Geometry Angle Bands using the geo_coordinates.nc file dimensions: (4091, 4865)
            t_geometries = nc.Dataset(self.nc_folder / 'tie_geometries.nc')

            self.OAA = t_geometries['OAA'][:]
            self.OAA = resize(self.OAA, (self.g_lon.shape[0], self.g_lon.shape[1]), anti_aliasing=False)
            self.OZA = t_geometries['OZA'][:]
            self.OZA = resize(self.OZA, (self.g_lon.shape[0], self.g_lon.shape[1]), anti_aliasing=False)
            self.SAA = t_geometries['SAA'][:]
            self.SAA = resize(self.SAA, (self.g_lon.shape[0], self.g_lon.shape[1]), anti_aliasing=False)
            self.SZA = t_geometries['SZA'][:]
            self.SZA = resize(self.SZA, (self.g_lon.shape[0], self.g_lon.shape[1]), anti_aliasing=False)

        elif self.product.lower() == 'syn':
            dsgeo = nc.Dataset(self.nc_folder / 'geolocation.nc')
            self.g_lat = dsgeo['lat'][:]
            self.g_lon = dsgeo['lon'][:]

        else:
            self.log.info(f'Invalid product: {self.product.upper()}.')
            sys.exit(1)

    def __repr__(self):
        return f'{type(self.t_lat)}, ' \
               f'{type(self.t_lon)}, ' \
               f'{type(self.g_lat)}, ' \
               f'{type(self.g_lon)}, ' \
               f'{type(self.OAA)}, ' \
               f'{type(self.OZA)}, ' \
               f'{type(self.SAA)},' \
               f'{type(self.SZA)},' \
               f'nc_base_name:{self.nc_base_name}'

    def get_valid_band_files(self, rad_only=True):
        """
        Search inside the .SEN3 image folder for files ended with .nc
        Default behavior is to return only the radiometric bands.
        If rad_only=False it will return everything ended in .nc extension.


        Returns
        -------
        rad_only=True
        @return nc_bands: list of files containing valid S3 .nc bands available.
        
        rad_only=False
        @return nc_files: list of files containing every .nc file available.
        """
        if self.nc_folder is None:
            self.log.info(f'Unable to find files. NetCDF image folder is not defined during {__name__} class instance.')
            sys.exit(1)

        sentinel_images_path = self.nc_folder

        # retrieve all files in folder
        files = os.listdir(sentinel_images_path)

        # extract only NetCDFs from the file list
        nc_files = [f for f in files if f.endswith('.nc')]

        # extract only the radiometric bands from the NetCDF list
        nc_bands = [b for b in nc_files if b.startswith('Oa')]

        return nc_bands if rad_only else nc_files

    def latlon_2_xy_poly(self, poly_path, go_parallel=False):
        """
        Given an input polygon and image, return a dataframe containing
        the data of the image that falls inside the polygon.
        """
        self.log.info(f'Converting the polygon coordinates into a matrix x,y poly...')
        # I) Convert the lon/lat polygon into a x/y poly:
        xy_vert, ll_vert = self._lat_lon_2_xy(poly_path=poly_path, parallel=go_parallel)

        return xy_vert, ll_vert

    def _lat_lon_2_xy(self, poly_path, parallel=True):
        """
        Takes in a polygon file and return a dataframe containing
        the data in each band that falls inside the polygon.
        """
        # self._test_initialized()

        if parallel:
            gpc = ParallelCoord()

            xy_vertices = [gpc.parallel_get_xy_poly(self.g_lat, self.g_lon, vert) for vert in poly_path]
        else:
            xy_vertices = [u.get_x_y_poly(self.g_lat, self.g_lon, vert) for vert in poly_path]

        return xy_vertices, poly_path

    def get_raster_mask(self, xy_vertices):
        """
        Creates a boolean mask of 0 and 1 with the polygons using the nc resolution.
        """
        # self._test_initialized()
        # Generate extraction mask

        img = np.zeros(self.g_lon.shape)
        cc = np.ndarray(shape=(0,), dtype='int64')
        rr = np.ndarray(shape=(0,), dtype='int64')

        for vert in xy_vertices:
            t_rr, t_cc = polygon(vert[:, 0], vert[:, 1], self.g_lon.shape)
            img[t_rr, t_cc] = 1
            cc = np.append(cc, t_cc)
            rr = np.append(rr, t_rr)

        return img, cc, rr

    def get_rgb_from_poly(self, xy_vertices):

        # II) Get the bounding box:
        xmin, xmax, ymin, ymax = u.bbox(xy_vertices)

        # III) Get only the RGB bands:
        if self.product.lower() == 'wfr':
            ds = nc.Dataset(self.nc_folder / 'Oa08_reflectance.nc')
            red = ds['Oa08_reflectance'][:]
            ds = nc.Dataset(self.nc_folder / 'Oa06_reflectance.nc')
            green = ds['Oa06_reflectance'][:]
            ds = nc.Dataset(self.nc_folder / 'Oa03_reflectance.nc')
            blue = ds['Oa03_reflectance'][:]

        elif self.product.lower() == 'syn':
            ds = nc.Dataset(self.nc_folder / 'Syn_Oa08_reflectance.nc')
            red = ds['SDR_Oa08'][:]
            ds = nc.Dataset(self.nc_folder / 'Syn_Oa06_reflectance.nc')
            green = ds['SDR_Oa06'][:]
            ds = nc.Dataset(self.nc_folder / 'Syn_Oa03_reflectance.nc')
            blue = ds['SDR_Oa03'][:]
        else:
            self.log.info(f'Invalid product: {self.product.upper()}.')
            sys.exit(1)

        # IV) Subset the bands using the bbox:
        red = red[ymin:ymax, xmin:xmax]
        green = green[ymin:ymax, xmin:xmax]
        blue = blue[ymin:ymax, xmin:xmax]

        # V) Stack the bands vertically:
        # https://stackoverflow.com/questions/10443295/combine-3-separate-numpy-arrays-to-an-rgb-image-in-python
        rgb_uint8 = (np.dstack((red, green, blue)) * 255.999).astype(np.uint8)

        return red, green, blue, rgb_uint8


class ParallelCoord:

    @staticmethod
    def vect_dist_subtraction(coord_pair, grid):
        subtraction = coord_pair - grid
        dist = np.linalg.norm(subtraction, axis=2)
        result = np.where(dist == dist.min())
        target_x_y = [result[0][0], result[1][0]]
        return target_x_y

    def parallel_get_xy_poly(self, lat_arr, lon_arr, polyline):
        # Stack LAT and LON in the Z axis
        grid = np.concatenate([lat_arr[..., None], lon_arr[..., None]], axis=2)

        # Polyline is a GeoJSON coordinate array
        polyline = polyline.squeeze()  # squeeze removes one of the dimensions of the array
        # https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html

        # Generate a list containing the lat, lon coordinates for each point of the input poly
        coord_vect_pairs = []
        for i in range(polyline.shape[0]):
            coord_vect_pairs.append(np.array([polyline[i, 1], polyline[i, 0]]).reshape(1, 1, -1))

        # for future reference
        # https://stackoverflow.com/questions/6832554/multiprocessing-how-do-i-share-a-dict-among-multiple-processes
        cores = u.get_available_cores()
        with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
            try:
                result = list(executor.map(self.vect_dist_subtraction, coord_vect_pairs, [grid]*len(coord_vect_pairs)))

            except concurrent.futures.process.BrokenProcessPool as ex:
                print(f"{ex} This might be caused by limited system resources. "
                      f"Try increasing system memory or disable concurrent processing. ")

        return np.array(result)


class ParallelBandExtract:

    def __init__(self, parent_log=None):
        if parent_log:
            self.log = parent_log

    def _get_band_in_nc(self, file_n_band, rr, cc):
        
        print(u"\u2588", end='')
        # logging.info(f'{os.getpid()} | Extracting band: {file_n_band[1]} from file: {file_n_band[0]}.\n')
        # self.log.info(f'{os.getpid()} | Extracting band: {file_n_band[1]} from file: {file_n_band[0]}.\n')
        result = {}
        # load NetCDF folder + nc_file_name
        ds = nc.Dataset(file_n_band[0])
        # load the nc_band_name as a matrix and unmask its values
        band = ds[file_n_band[1]][:].data
        # extract the values of the matrix and return as a dict entry
        result[file_n_band[1]] = [band[x, y] for x, y in zip(rr, cc)]
        return result

    def nc_2_df(self, rr, cc, oaa, oza, saa, sza, lon, lat, nc_folder, wfr_files_p, parent_log=None):
        """
        Given an input polygon and image, return a dataframe containing
        the data of the image that falls inside the polygon.
        """
        if parent_log:
            self.log = logging.getLogger(name=parent_log)

        wfr_files_p = [(os.path.join(nc_folder, nc_file), nc_band) for nc_file, nc_band in wfr_files_p]

        # Generate initial df
        custom_subset = {'x': rr, 'y': cc}
        df = pd.DataFrame(custom_subset)
        df['lat'] = [lat[x, y] for x, y in zip(df['x'], df['y'])]
        df['lon'] = [lon[x, y] for x, y in zip(df['x'], df['y'])]
        df['OAA'] = [oaa[x, y] for x, y in zip(df['x'], df['y'])]
        df['OZA'] = [oza[x, y] for x, y in zip(df['x'], df['y'])]
        df['SAA'] = [saa[x, y] for x, y in zip(df['x'], df['y'])]
        df['SZA'] = [sza[x, y] for x, y in zip(df['x'], df['y'])]

        cores = u.get_available_cores()
        # Populate the initial DF with the output from the other bands
        with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
            try:
                print(u.repeat_to_length('_', len(wfr_files_p)))
                list_of_bands = list(executor.map(
                    self._get_band_in_nc, wfr_files_p,
                    [rr] * len(wfr_files_p),
                    [cc] * len(wfr_files_p)
                ))
                print(' Done.')
            except concurrent.futures.process.BrokenProcessPool as ex:
                print(f"{ex} This might be caused by limited system resources. "
                              f"Try increasing system memory or disable concurrent processing. ")

        # For every returned dict inside the list, grab only the Key and append it at the final DF
        for b in list_of_bands:
            for key, val in b.items():
                df[key] = val

        # DROP NODATA
        idx_names = df[df['Oa08_reflectance'] == 65535.0].index
        df.drop(idx_names, inplace=True)
        return df
