import os
import sys
import rasterio
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr

from shapely.geometry import box
from rasterstats import zonal_stats
from getpak.input import GRS
from getpak.commons import DefaultDicts as dd
from dask import compute
from dask import delayed

from getpak import owts_spy_S2_B1_7
from getpak import owts_spy_S2_B2_7
from getpak import owts_spm_S2_B1_8A
from getpak import owts_spm_S2_B2_8A 


class Methods:
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

    """

    def __init__(self):
        pass
    
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
    def sample_grs_with_shp(grs_nc, vector_shp, grs_version='v20', unique_shp_key='id'):
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
        @return result: bla bla bla
        '''
        try:
            # No need to print or log, the functions should handle it internally.
            GRS.grs_dict = GRS.get_grs_dict(grs_nc, grs_version)
            GRS.gpd_feature_dict = GRS._get_shp_features(vector_shp, unique_key=unique_shp_key)

        except Exception as e:
            # self.log.error(f'Error: {e}')
            sys.exit(1)  # Exit program: 0 is success, 1 is failure

        # Initialize the dict and fill it with zeros
        # This will be converted to a pd.DataFrame later
        pt_stats = {}
        for band in GRS.grs_v20nc_s2bands.keys():
            pt_stats[GRS.grs_v20nc_s2bands[band]] = {id: 0.0 for id in GRS.gpd_feature_dict.keys()}

        # Declare a list to hold delayed tasks
        tasks = []

        # Create delayed tasks for each combination of band and shapefile point
        for band in dd.grs_v20nc_s2bands.keys():
            for id in dd.gpd_feature_dict.keys():
                # Get the shape feature
                shp_feature = dd.gpd_feature_dict[id]
                # Create a delayed task
                task = delayed(GRS._process_band_point)(band, shp_feature, shp_feature.crs, id)
                tasks.append(task)

        global counter
        counter = 0
        print(f'Processing {len(tasks)} tasks...')
        # Compute all delayed dask-tasks
        results = compute(*tasks)
        print(f' {counter}')
        print('Done.')

        # Aggregate results into pt_stats
        for band, pt_id, mean_rrs in results:
            pt_stats[dd.grs_v20nc_s2bands[band]][pt_id] = mean_rrs

        df = pd.DataFrame(pt_stats)
        return df

    def _sam(self, rrs, single=False, sensor='S2MSI', mode='B1'):
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

        # Loading the correct OWTs
        if sensor == 'S2MSI':
            if mode == 'B1':
                owts = owts_spy_S2_B1_7
            else:
                owts = owts_spy_S2_B2_7

        # Convert OWT values to numpy array for vectorized computations
        M = np.array([list(val.values()) for val in owts.values()])
        nM = np.linalg.norm(M, axis=1)

        # scalar product
        num = np.dot(M, E)
        den = nM * nE

        angles = np.arccos(num / den)

        return angles

    def _euclid_dist(self, rrs_px, rrs_owt):
        """
        Euclidean distance between a pixel and the OWT classes
        Returns the distance between the Rrs of the pixels and each OWT
        ----------
        """
        # normalising the pixel reflectance
        nE = rrs_px / rrs_px.sum()

        # normalising the OWT reflectance
        nM = rrs_owt / rrs_owt.sum()

        # Euclidean distance
        angle = np.linalg.norm(nE - nM)

        return angle

    # Filters
    def filter_pixels(self, rrs_dict, neg_rrs='Red', low_rrs=True, low_rrs_thresh=0.002,
                      low_rrs_bands=['Aerosol', 'Blue', 'Green', 'Red', 'RedEdge1', 'RedEdge2']):
        """
        Function to filter bad pixels from the processing
        Removes pixels with negative Rrs, based on the choice of a band (neg_rrs). It can also remove pixels with low
        Rrs, considered those pixels which have maximum Rrs across a range of bands (to be chosen based on low_rrs_bands)
        lower than a threshold (low_rrs_thresh)

        Parameters
        ----------
        @rrs_dict: a xarray Dataset containing the Rrs bands
        @neg_rrs: the name of the band to filter negative Rrs pixels
        @low_rrs: whether to remove low Rrs pixels or not (boolean)
        @low_rrs_thresh: threshold for removing low Rrs pixels
        @low_rrs_bands: the bands to be considered for removing low Rrs pixels

        Returns
        -------
        @img: a xarray Dataset containing the Rrs bands with the pixels filtered

        """
        self.npix = len(np.where(np.isnan(rrs_dict['Red'].values) == False)[0])
        # Removing negative Rrs pixels
        if isinstance(neg_rrs, str):
            self.negpix = len(np.where(rrs_dict[neg_rrs] <= 0)[0])
            mask = rrs_dict[neg_rrs] >= 0
            rrs_dict = rrs_dict.where(mask, np.nan)

        # Removing low Rrs pixels
        if low_rrs:
            # Compute the maximum value across all bands and create a mask
            stacked = xr.concat([rrs_dict[var] for var in low_rrs_bands], dim='variable')
            max_values = stacked.max(dim='variable')
            mask = max_values >= low_rrs_thresh
            n_neg = len(np.where(np.isnan(rrs_dict['Red'].values) == False)[0]) - len(np.where(mask)[0])
            self.lowrrs = n_neg
            if n_neg > 0:
                rrs_dict = rrs_dict.where(mask, np.nan)

        return rrs_dict

    # Classification methods
    def classify_owt_chla_px(self, rrs_dict, sensor='S2MSI', B1=True):
        """
        Classify the OWT of each pixel according to the Spyrakos et al. (2018) classes

        Parameters
        ----------
        @rrs_dict: a xarray Dataset containing the Rrs bands
        @sensor: a string of the satellite mission, one of:
            S2MSI for Sentinel-2 MSI A and B
            S3OLCI for Sentinel-3 OLCI A and B
        @B1: boolean to whether or not use Band 1 when using Sentinel-2 data

        Returns
        -------
        @class_px: an array, with the same size as the input bands, with the pixels classified with the smallest SAM
        @angles: a 2-dimensional array, where the first dimension is the number of valid pixels in the rrs_dict, and
        the second dimension is the number of OWTs. The values are the spectral angles between the Rrs and the mean Rrs
        of each OWT, in each pixel

        """
        if sensor == 'S2MSI':
            if B1:
                bands = ['Aerosol', 'Blue', 'Green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3']
                mode = 'B1'
            else:
                bands = ['Blue', 'Green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3']
                mode = 'B2'
        # Drop the variables that won't be used in the classification
        variables_to_drop = [var for var in rrs_dict.variables if var not in bands + ['x', 'y']]
        rrs = rrs_dict.drop_vars(variables_to_drop)
        # Find non-NaN values across all bands
        nzero = np.where(~np.any([np.isnan(rrs[var]) for var in bands], axis=0))
        # array of OWT class for each pixel
        class_px = np.zeros_like(rrs[bands[0]], dtype='uint8')
        # array of angles to limit the loop
        if sensor == 'S2MSI':
            angles = np.zeros((len(nzero[0]), len(owts_spy_S2_B1_7)), dtype='float16')

        # creating a new Band 1 by undoing the upsampling of GRS, keeping only the pixels entirely inside water
        if sensor == 'S2MSI' and B1:
            aux = rrs['Aerosol'].coarsen(x=3, y=3).mean(skipna=False).interp(x=rrs.x, y=rrs.y, method='nearest').values
            rrs['Aerosol60m'] = (('x', 'y'), aux)
            bands = ['Aerosol60m', 'Blue', 'Green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3']

        # loop over each nonzero value in the rrs_dict
        pix = np.zeros((len(nzero[0]), len(bands)))
        for i in range(len(bands)):
            pix[:, i] = rrs[bands[i]].values[nzero]
        for i in range(len(nzero[0])):
            angles[i, :] = self._sam(rrs=pix[i, :], sensor=sensor, mode=mode, single=True)

        # if B1 is being used, there won't be any classification for the pixels without values in B1, so they have to be
        # classified using only bands 2 to 7
        if B1:
            nodata = np.where(np.isnan(angles[:, 0]))[0]
            for i in range(len(nodata)):
                angles[nodata[i], :] = self._sam(rrs=pix[nodata[i], 1:], sensor=sensor, mode='B2', single=True)
        class_px[nzero] = np.nanargmin(angles, axis=1) + 1

        return class_px, angles

    def classify_owt_chla_shp(self, rasterio_rast, shapefiles, rrs_dict, sensor='S2MSI', B1=True, min_px=9):
        """
        Classify the OWT of pixels inside a shapefile (or a set of shapefiles) according to the Spyrakos et al. (2018)
        classes

        Parameters
        ----------
        @rasterio_rast: a rasterio raster with the same configuration as the bands, open with rasterio.open
        @shapefiles: a polygon (or set of polygons), usually of waterbodies to be classified, opened as geometry
            using fiona
        @rrs_dict: a xarray Dataset containing the Rrs bands
        @sensor: a string of the satellite mission, one of:
            S2MSI for Sentinel-2 MSI A and B
            S3OLCI for Sentinel-3 OLCI A and B
        @B1: boolean to use Band 1 when using Sentinel-2 data
        @min_px: minimum number of pixels in each polygon to operate the classification

        Returns
        -------
        @class_spt: an array, with the same size as the input bands, with the classified pixels
        @class_shp: an array with the same length as the shapefiles, with a OWT class for each polygon
        """
        # checking if B1 will be used in the classification
        if sensor == 'S2MSI':
            if B1:
                bands = ['Aerosol', 'Blue', 'Green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3']
                mode = 'B1'
            else:
                bands = ['Blue', 'Green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3']
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

    def classify_owt_spm_px(self, rrs_dict, sensor='S2MSI', B1=True):
        """
        Classify the OWT of each pixel based on the SPM optical water classes (Codeiro, 2022)
        It is based on the minimum Euclidean distance between the spectra of each pixel and the 4 classes

        Parameters
        ----------
        @rrs_dict: a xarray Dataset containing the Rrs bands
        @sensor: a string of the satellite mission, one of:
            S2MSI for Sentinel-2 MSI A and B
            S3OLCI for Sentinel-3 OLCI A and B
        @B1: boolean to whether or not use Band 1 when using Sentinel-2 data

        Returns
        -------
        @class_px: an array, with the same size as the input bands, with the pixels classified with the smallest
        Euclidean distance
        @angles: a 2-dimensional array, where the first dimension is the number of valid pixels in the rrs_dict, and
        the second dimension is the number of OWTs. The values are the Euclidean distances between the Rrs and the mean
        Rrs of each OWT, in each pixel

        """
        if sensor == 'S2MSI':
            if B1:
                bands = ['Aerosol', 'Blue', 'Green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3', 'Nir1', 'Nir2']
                mode = 'B1'
            else:
                bands = ['Blue', 'Green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3', 'Nir1', 'Nir2']
                mode = 'B2'
        # Drop the variables that won't be used in the classification
        variables_to_drop = [var for var in rrs_dict.variables if var not in bands + ['x', 'y']]
        rrs = rrs_dict.drop_vars(variables_to_drop)
        # Find non-NaN values across all bands
        nzero = np.where(~np.any([np.isnan(rrs[var]) for var in bands], axis=0))
        # array of OWT class for each pixel
        class_px = np.zeros_like(rrs[bands[0]], dtype='uint8')
        # array of angles to limit the loop
        if sensor == 'S2MSI':
            angles = np.zeros((len(nzero[0]), len(owts_spm_S2_B1_8A)), dtype='float16')

        # creating a new Band 1 by undoing the upsampling of GRS, keeping only the pixels entirely inside water
        if sensor == 'S2MSI' and B1:
            aux = rrs['Aerosol'].coarsen(x=3, y=3).mean(skipna=False).interp(x=rrs.x, y=rrs.y, method='nearest').values
            rrs['Aerosol60m'] = (('x', 'y'), aux)
            bands = ['Aerosol60m', 'Blue', 'Green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3', 'Nir1', 'Nir2']

        # loop over each nonzero value in the rrs_dict
        pix = np.zeros((len(nzero[0]), len(bands)))
        for i in range(len(bands)):
            pix[:, i] = rrs[bands[i]].values[nzero]
        # Loading the correct dictionary of OWTs
        if sensor == 'S2MSI':
            if B1:
                owts = owts_spm_S2_B1_8A
            else:
                owts = owts_spm_S2_B2_8A
        # array of values of the OWTs
        M = np.array([list(val.values()) for val in owts.values()])
        for i in range(len(nzero[0])):
            for j in range(len(M)):
                # angles[i, j] = self._euclid_dist(pix[i, :], mode=mode)
                angles[i, j] = np.linalg.norm(pix[i, :] - M[j])

        # if B1 is being used, there won't be any classification for the pixels without values in B1, so they have to be
        # classified using only bands 2 to 7
        if sensor == 'S2MSI' and B1:
            nodata = np.where(np.isnan(angles[:, 0]))[0]
            M = np.array([list(val.values()) for val in owts_spm_S2_B2_8A.values()])
            for i in range(len(nodata)):
                for j in range(len(M)):
                    # angles[nodata[i], j] = self._euclid_dist(pix[i, :], mode='B2')
                    angles[nodata[i], j] = np.linalg.norm(pix[nodata[i], 1:] - M[j])
        class_px[nzero] = np.nanargmin(angles, axis=1) + 1

        return class_px, angles

    def classify_owt_chla_weights(self, class_px, angles, n=3, remove_classes=None):
        """
        Attribute weights to the n-th most important OWTs, based on the spectral angle mapper
        The weights are used for calculating weighted means of the water quality parameters, in order to smooth the
        spatial differences between the pixels, and also to remove possible outliers generated by some models
        For more information on this approach, please refer to Moore et al. (2001) and Liu et al. (2021)

        This function uses the results of classify_owt_chla_px as input data

        Parameters
        ----------
        @class_px: an array, with the same size as the input bands, with the pixels classified with the smallest SAM
        @angles: a 2-dimensional array, where the first dimension is the number of valid pixels in the rrs_dict, and
        the second dimension is the number of OWTs. The values are the spectral angles between the Rrs and the mean Rrs
        of each OWT, in each pixel
        @n = the number of dominant classes to be used to generate the weights
        @remove_classes: int or a list of OWT classes to be removed (pixel-wise)

        Returns
        -------
        @owt_classes: an array, with the same size as the input bands, with the n classes of pixels (first dimension)
        @owt_weights: an array, with the same size as the input bands, with the n weights (first dimension)
        """
        # creating the variables of weights and classes for each pixel, depending on the desired number of classes to
        # be used
        owt_weights = np.zeros((n, *class_px.shape), dtype='float32')
        owt_classes = np.zeros((n, *class_px.shape), dtype='float32')

        # finding where there is no valid pixels in the reflectance data
        nzero = np.where(class_px != 0)

        # Create an array of indices
        indices = np.argsort(angles, axis=1)
        lowest_angles = np.take_along_axis(angles, indices[:, 0:(n + 1):], axis=1)

        # Calculating the weights based on normalisation of the n+1 lowest spectral angles (parallel of convertion of
        # units) and assigning values to the matrices
        for i in range(n):
            owt_classes[i, nzero[0], nzero[1]] = indices[:, i] + 1  # summing one due to positioning starting in 1
            for j in range(nzero[0].shape[0]):
                owt_weights[i, nzero[0][j], nzero[1][j]] = (lowest_angles[j, i] - lowest_angles[j, -1]) / (
                        lowest_angles[j, 0] - lowest_angles[j, -1])

        # removing undesidered OWTs:
        if isinstance(remove_classes, int):
            ones = np.where(indices[:, 0] == (remove_classes - 1))[0]
            owt_weights[:, nzero[0][ones], nzero[1][ones]] = np.nan
        elif isinstance(remove_classes, list):
            for i in range(len(remove_classes)):
                ones = np.where(indices[:, 0] == (remove_classes[i] - 1))[0]
                owt_weights[:, nzero[0][ones], nzero[1][ones]] = np.nan

        # # removing the zeros to avoid division by 0
        # owt_weights[np.where(owt_weights == 0)] = np.nan
        # owt_classes[np.where(owt_classes == 0)] = np.nan

        return owt_classes, owt_weights

    # Calculation methods
    def cdom(self, rrs_dict, upper_lim=50, lower_lim=0):
        """
        Calculate the coloured dissolved organic matter (CDOM) based on the optical water type (OWT)

        Parameters
        ----------
        @rrs_dict: rrs_dict: a xarray Dataset containing the Rrs bands

        Returns
        -------
        @cdom: an array, with the same size as the input bands, with the modeled values
        """
        import getpak.inversion_functions as ifunc
        cdom = ifunc.cdom_brezonik(Blue=rrs_dict['Blue'].values, RedEdge2=rrs_dict['RedEdge2'].values)

        # removing espurious values
        if isinstance(upper_lim, (int, float)) and isinstance(lower_lim, (int, float)):
            out = np.where((cdom < lower_lim) | (cdom > upper_lim))
            cdom[out] = np.nan

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
        @rrs_dict: rrs_dict: a xarray Dataset containing the Rrs bands
        @class_owt_spt: an array, with the same size as the input bands, with the OWT classes for each pixel
        @limits: boolean to choose whether to apply the algorithms only in their limits of cal/val
        @alg: one the following algorithms available:
            owt: To use the methodology based on OWTs (Tavares et al., 2025)
            gilerson2: 2-band NIR-red ratio by Gilerson et al. (2010)
            gilerson3: 3-band NIR-red ratio by Gilerson et al. (2010)
            gons: 2-band semi-analytical by Gons et al. (2003, 2005)
            gurlin: 2-band squared band ratio by Gurlin et al. (2011)
            ndci: Normalised Difference Chlorophyll Index, Mishra and Mishra (2012)
            oc2: NASA Ocean Colour 2-band algorithm, O'Reilly et al. (1998)

        Returns
        -------
        @chla: an array, with the same size as the input bands, with the modeled values
        """
        import getpak.inversion_functions as ifunc
        chla = np.zeros(rrs_dict['Red'].shape, dtype='float32')

        if alg == 'owt':
            # chla functions for each OWT
            # for class 1, using the coefficients calibrated by Neil et al. (2020) for the OWT 1
            # classes = [1]
            # index = np.where(np.isin(class_owt_spt, classes))
            # if len(index[0] > 0):
            #     chla[index] = ifunc.chl_gurlin(Red=rrs_dict['Red'].values[index],
            #                                                             RedEdge1=rrs_dict['Rrs_B5'].values[index],
            #                                                             a=86.09, b=-517.5, c=886.7)
            #     if limits:
            #         lims = [10, 1000]
            #         out = np.where((chla[index] < lims[0]) | (chla[index] > lims[1]))
            #         chla[index[0][out], index[1][out]] = np.nan

            classes = [1, 6, 10]
            index = np.where(np.isin(class_owt_spt, classes))
            if len(index[0] > 0):
                chla[index] = ifunc.chl_gons(Red=rrs_dict['Red'].values[index],
                                             RedEdge1=rrs_dict['RedEdge1'].values[index],
                                             RedEdge3=rrs_dict['RedEdge3'].values[index],
                                             aw665=0.425, aw708=0.704)
                if limits:
                    lims = [1, 250]
                    out = np.where((chla[index] < lims[0]) | (chla[index] > lims[1]))
                    chla[index[0][out], index[1][out]] = np.nan

            classes = [2, 4, 5, 11, 12]
            index = np.where(np.isin(class_owt_spt, classes))
            if len(index[0] > 0):
                chla[index] = ifunc.chl_ndci(Red=rrs_dict['Red'].values[index],
                                             RedEdge1=rrs_dict['RedEdge1'].values[index])
                if limits:
                    lims = [2.5, 250]
                    out = np.where((chla[index] < lims[0]) | (chla[index] > lims[1]))
                    chla[index[0][out], index[1][out]] = np.nan

            # for class 2 and 12, when values of NDCI are >20, use Gilerson instead
            classes = [2, 12]
            conditions = (np.isin(class_owt_spt, classes)) & (chla > 20)
            index = np.where(conditions)
            if len(index[0] > 0):
                chla[index] = ifunc.chl_gilerson2(Red=rrs_dict['Red'].values[index],
                                                  RedEdge1=rrs_dict['RedEdge1'].values[index])
                if limits:
                    lims = [2.5, 500]
                    out = np.where((chla[index] < lims[0]) | (chla[index] > lims[1]))
                    chla[index[0][out], index[1][out]] = np.nan

            classes = [7, 8]
            index = np.where(np.isin(class_owt_spt, classes))
            if len(index[0] > 0):
                chla[index] = ifunc.chl_gilerson2(Red=rrs_dict['Red'].values[index],
                                                  RedEdge1=rrs_dict['RedEdge1'].values[index])
                if limits:
                    lims = [2.5, 500]
                    out = np.where((chla[index] < lims[0]) | (chla[index] > lims[1]))
                    chla[index[0][out], index[1][out]] = np.nan

            # classes = []
            # index = np.where(np.isin(class_owt_spt, classes))
            # if len(index[0] > 0):
            #     chla[index] = ifunc.chl_gilerson3(Red=rrs_dict['Red'].values[index],
            #                                       RedEdge1=rrs_dict['RedEdge1'].values[index],
            #                                       RedEdge2=rrs_dict['RedEdge2'].values[index])
            #     if limits:
            #         lims = [10, 1000]
            #         out = np.where((chla[index] < lims[0]) | (chla[index] > lims[1]))
            #         chla[index[0][out], index[1][out]] = np.nan

            classes = [3]
            index = np.where(np.isin(class_owt_spt, classes))
            if len(index[0] > 0):
                chla[index] = ifunc.chl_OC2(Blue=rrs_dict['Blue'].values[index],
                                            Green=rrs_dict['Green'].values[index], a=0.1098, b=-0.755, c=-14.12,
                                            d=-117, e=-17.76)
                if limits:
                    lims = [0.01, 50]
                    out = np.where((chla[index] < lims[0]) | (chla[index] > lims[1]))
                    chla[index[0][out], index[1][out]] = np.nan

            classes = [9]
            index = np.where(np.isin(class_owt_spt, classes))
            if len(index[0] > 0):
                chla[index] = ifunc.chl_OC2(Blue=rrs_dict['Blue'].values[index],
                                            Green=rrs_dict['Green'].values[index], a=0.0536, b=7.308, c=116.2,
                                            d=412.4, e=463.5)
                if limits:
                    lims = [0.01, 50]
                    out = np.where((chla[index] < lims[0]) | (chla[index] > lims[1]))
                    chla[index[0][out], index[1][out]] = np.nan

            classes = [13]
            index = np.where(np.isin(class_owt_spt, classes))
            if len(index[0] > 0):
                chla[index] = ifunc.chl_OC2(Blue=rrs_dict['Blue'].values[index],
                                            Green=rrs_dict['Green'].values[index], a=-5020, b=2.9e+04, c=-6.1e+04,
                                            d=5.749e+04, e=-2.026e+04)
                if limits:
                    lims = [0.01, 50]
                    out = np.where((chla[index] < lims[0]) | (chla[index] > lims[1]))
                    chla[index[0][out], index[1][out]] = np.nan

            classes = [14]
            index = np.where(np.isin(class_owt_spt, classes))
            if len(index[0] > 0):
                chla[index] = ifunc.chl_OC2(Blue=rrs_dict['Blue'].values[index],
                                            Green=rrs_dict['Green'].values[index])
                if limits:
                    lims = [0.01, 50]
                    out = np.where((chla[index] < lims[0]) | (chla[index] > lims[1]))
                    chla[index[0][out], index[1][out]] = np.nan

        elif alg == 'gilerson2':
            chla = ifunc.chl_gilerson2(Red=rrs_dict['Red'].values, RedEdge1=rrs_dict['RedEdge1'].values)
            if limits:
                lims = [2.5, 500]
                out = np.where((chla < lims[0]) | (chla > lims[1]))
                chla[out] = np.nan

        elif alg == 'gilerson3':
            chla = ifunc.chl_gilerson3(Red=rrs_dict['Red'].values, RedEdge2=rrs_dict['RedEdge2'].values)
            if limits:
                lims = [10, 2000]
                out = np.where((chla < lims[0]) | (chla > lims[1]))
                chla[out] = np.nan

        elif alg == 'gons':
            chla = ifunc.chl_gons(Red=rrs_dict['Red'].values, RedEdge1=rrs_dict['RedEdge1'].values,
                                  RedEdge3=rrs_dict['RedEdge3'].values)
            if limits:
                lims = [1, 250]
                out = np.where((chla < lims[0]) | (chla > lims[1]))
                chla[out] = np.nan

        elif alg == 'gurlin':
            chla = ifunc.chl_gurlin(Red=rrs_dict['Red'].values, RedEdge1=rrs_dict['RedEdge1'].values)
            if limits:
                lims = [10, 1000]
                out = np.where((chla < lims[0]) | (chla > lims[1]))
                chla[out] = np.nan

        elif alg == 'ndci':
            chla = ifunc.chl_ndci(Red=rrs_dict['Red'].values, RedEdge1=rrs_dict['RedEdge1'].values)
            if limits:
                lims = [2.5, 250]
                out = np.where((chla < lims[0]) | (chla > lims[1]))
                chla[out] = np.nan

        elif alg == 'oc2':
            chla = ifunc.chl_OC2(Blue=rrs_dict['Blue'].values, Green=rrs_dict['Green'].values)
            if limits:
                lims = [0.01, 50]
                out = np.where((chla < lims[0]) | (chla > lims[1]))
                chla[out] = np.nan

        # removing espurious values and zeros
        out = np.where((chla == 0) | np.isinf(chla))
        chla[out] = np.nan

        return chla

    def blended_chla(self, rrs_dict, owt_classes, owt_weights, limits=True):
        """
        Function to calculate a blended chlorophyll-a concentration (chla) product based on chla matrices calculated
        with function chlorophylla in mode 'owt'

        Parameters
        ----------
        @rrs_dict: rrs_dict: a xarray Dataset containing the Rrs bands
        @owt_classes: an array with the different OWT classes calculated with function classify_owt_chla_weights
        @owt_classes: an array with the OWT weights calculated with function classify_owt_chla_weights
        @limits: boolean to choose whether to apply the limits for the different layers of chla products

        Returns
        -------
        @chla: an array, with the same size as the input bands, with the modeled values
        """

        if isinstance(owt_classes, np.ndarray) and len(owt_classes.shape) == 3:
            # calculating chla for each
            aux_chla = np.zeros((owt_classes.shape[0], *rrs_dict['Red'].shape), dtype='float32')
            for i in range(0, owt_classes.shape[0]):
                aux_chla[i, :, :] = Methods.chlorophylla(self, rrs_dict, owt_classes[i, :, :], limits=True, alg='owt')

            # checking the limits
            if limits:
                for i in range(1, owt_classes.shape[0]):
                    over = np.where((np.absolute(aux_chla[i, :, :] - aux_chla[0, :, :]) > (4 * aux_chla[0, :, :])))
                    aux_chla[i, over[0], over[1]] = 0

            # inserting weights = 0
            for i in range(0, owt_classes.shape[0]):
                owt_weights[i, np.where(np.isnan(aux_chla[i, :, :]))[0], np.where(np.isnan(aux_chla[i, :, :]))[1]] = 0
                aux_chla[i, np.where(np.isnan(aux_chla[i, :, :]))[0], np.where(np.isnan(aux_chla[i, :, :]))[1]] = 0

            # calculating the blended product
            num = np.zeros(rrs_dict['Red'].shape, dtype='float32')
            den = np.zeros(rrs_dict['Red'].shape, dtype='float32')
            for i in range(0, owt_classes.shape[0]):
                num = num + (owt_weights[i, :, :] * aux_chla[i, :, :])
                den = den + owt_weights[i, :, :]
            chla = num / den

            # changing from np.nan to 0:
            chla[np.where(np.isnan(chla))] = 0

        elif isinstance(owt_classes, np.ndarray) and len(owt_classes.shape) == 3:
            print('There is only one class of OWT classes, use function chlorophylla instead.')
            sys.exit(1)
        else:
            print('Error: The vector of OWT classes has the wrong number of dimensions.')
            sys.exit(1)

        return chla

    def turb(self, rrs_dict, class_owt_spt, alg='owt', limits=True, mode_Jiang=None, rasterio_rast=None, shapefile=None,
             min_px=9):
        """
        Function to calculate the turbidity based on the optical water type (OWT)

        Parameters
        ----------
        @rrs_dict: rrs_dict: a xarray Dataset containing the Rrs bands
        @class_owt_spt: an array, with the same size as the input bands, with the OWT pixels
        @alg: one of the following algorithms available:
            owt: To use the methodology based on OWTs (Tavares et al., 2025)
            Nechad: Nechad et al. (2010)
            NechadGreen: Nechad but using the green band (Nechad et al., 2016)
            Binding: Binding et al. (2010)
            Zhang: Zhang et al. (2014)
            Dogliotti: hybrid algorithm by Dogliottt et al. (2015)
            Condé: developed for the Paranapanema river (Condé et al., 2019)
            Madeira: developed for Sentinel-2 imagery for the Madeira river (Alves e Santos et al., 2024)
            Hybrid: developed by Cordeiro (2022) for Sentinel-3 imagery
            or different versions of Jiang (Jiang_Green, Jiang_Red, or Jiang in forms 'pixel' or 'polygon')
        @limits: boolean to choose whether to apply the algorithms only in their limits of cal/val
        @mode_Jiang: used only for the general Jiang algorithm, to choose from:
            pixel: pixel-wise calculation
            polygon: lake-wise (based on shapefile) calculation
        @rasterio_rast: used only for the general Jiang algorithm, a rasterio raster with the same configuration as the
            bands, open with rasterio.open
        @shapefile: used only for the general Jiang algorithm, a polygon (or set of polygons), usually of waterbodies to
            be classified, opened as geometry using fiona
        @min_px: used only for the general Jiang algorithm, minimum number of pixels in each polygon to operate the
            inversion

        Returns
        -------
        turb: an array, with the same size as the input bands, with the modeled values
        """
        import getpak.inversion_functions as ifunc
        turb = np.zeros(rrs_dict['Red'].shape, dtype='float32')

        if alg == 'owt':
            # turb functions for each OWT
            classes = [1, 2, 3, 4]
            index = np.where(np.isin(class_owt_spt, classes))
            if len(index[0] > 0):
                classes = [1]
                index = np.where(np.isin(class_owt_spt, classes))
                if len(index[0] > 0):
                    turb[index] = ifunc.spm_jiang2021_green(Aerosol=rrs_dict['Aerosol'].values[index],
                                                            Blue=rrs_dict['Blue'].values[index],
                                                            Green=rrs_dict['Green'].values[index],
                                                            Red=rrs_dict['Red'].values[index])
                    if limits:
                        lims = [0, 50]
                        out = np.where((turb[index] < lims[0]) | (turb[index] > lims[1]))
                        turb[index[0][out], index[1][out]] = np.nan

                classes = [2]
                index = np.where(np.isin(class_owt_spt, classes))
                if len(index[0] > 0):
                    turb[index] = ifunc.spm_jiang2021_red(Aerosol=rrs_dict['Aerosol'].values[index],
                                                          Blue=rrs_dict['Blue'].values[index],
                                                          Green=rrs_dict['Green'].values[index],
                                                          Red=rrs_dict['Red'].values[index])
                    if limits:
                        lims = [10, 500]
                        out = np.where((turb[index] < lims[0]) | (turb[index] > lims[1]))
                        turb[index[0][out], index[1][out]] = np.nan

                classes = [3]
                index = np.where(np.isin(class_owt_spt, classes))
                if len(index[0] > 0):
                    turb[index] = ifunc.spm_zhang2014(RedEdge1=rrs_dict['RedEdge1'].values[index])

                    if limits:
                        lims = [20, 1000]
                        out = np.where((turb[index] < lims[0]) | (turb[index] > lims[1]))
                        turb[index[0][out], index[1][out]] = np.nan

                classes = [4]
                index = np.where(np.isin(class_owt_spt, classes))
                if len(index[0] > 0):
                    turb[index] = ifunc.spm_binding2010(RedEdge2=rrs_dict['RedEdge2'].values[index])

                    if limits:
                        lims = [50, 2000]
                        out = np.where((turb[index] < lims[0]) | (turb[index] > lims[1]))
                        turb[index[0][out], index[1][out]] = np.nan

        elif alg == 'Hybrid':
            turb = ifunc.spm_s3(Red=rrs_dict['Red'].values, Nir2=rrs_dict['Nir2'].values)
            if limits:
                lims = [0.1, 2000]
                out = np.where((turb < lims[0]) | (turb > lims[1]))
                turb[out] = np.nan

        elif alg == 'Nechad':
            turb = ifunc.spm_nechad(Red=rrs_dict['Red'].values)
            if limits:
                lims = [0.1, 1000]
                out = np.where((turb < lims[0]) | (turb > lims[1]))
                turb[out] = np.nan

        elif alg == 'NechadGreen':
            turb = ifunc.spm_nechad(Red=rrs_dict['Green'].values, a=228.72, c=0.2200)
            if limits:
                lims = [0.1, 200]
                out = np.where((turb < lims[0]) | (turb > lims[1]))
                turb[out] = np.nan

        elif alg == 'Binding':
            turb = ifunc.spm_binding2010(RedEdge2=rrs_dict['RedEdge2'].values)
            if limits:
                lims = [0.1, 2000]
                out = np.where((turb < lims[0]) | (turb > lims[1]))
                turb[out] = np.nan

        elif alg == 'Zhang':
            turb = ifunc.spm_zhang2014(RedEdge1=rrs_dict['RedEdge1'].values)
            if limits:
                lims = [0.1, 1000]
                out = np.where((turb < lims[0]) | (turb > lims[1]))
                turb[out] = np.nan

        elif alg == 'Jiang_Green':
            turb = ifunc.spm_jiang2021_green(Aerosol=rrs_dict['Aerosol'].values,
                                             Blue=rrs_dict['Blue'].values,
                                             Green=rrs_dict['Green'].values,
                                             Red=rrs_dict['Red'].values)
            if limits:
                lims = [0.1, 100]
                out = np.where((turb < lims[0]) | (turb > lims[1]))
                turb[out] = np.nan

        elif alg == 'Jiang_Red':
            turb = ifunc.spm_jiang2021_red(Aerosol=rrs_dict['Aerosol'].values,
                                           Blue=rrs_dict['Blue'].values,
                                           Green=rrs_dict['Green'].values,
                                           Red=rrs_dict['Red'].values)
            if limits:
                lims = [0.1, 500]
                out = np.where((turb < lims[0]) | (turb > lims[1]))
                turb[out] = np.nan

        elif alg == 'Dogliotti':
            turb = ifunc.spm_dogliotti_S2(Red=rrs_dict['Red'].values,
                                          Nir2=rrs_dict['Nir2'].values)
            if limits:
                lims = [0.1, 1000]
                out = np.where((turb < lims[0]) | (turb > lims[1]))
                turb[out] = np.nan

        elif alg == 'Conde':
            turb = ifunc.spm_conde(Red=rrs_dict['Red'].values)
            if limits:
                lims = [0.1, 100]
                out = np.where((turb < lims[0]) | (turb > lims[1]))
                turb[out] = np.nan

        elif alg == 'Madeira':
            turb = ifunc.spm_madeira(Red=rrs_dict['Red'].values, Nir2=rrs_dict['Nir2'].values)
            if limits:
                lims = [0.1, 2000]
                out = np.where((turb < lims[0]) | (turb > lims[1]))
                turb[out] = np.nan

        elif alg == 'Hybrid':
            turb = ifunc.spm_s3(Red=rrs_dict['Red'].values, Nir2=rrs_dict['Nir2'].values)
            if limits:
                lims = [0.1, 1000]
                out = np.where((turb < lims[0]) | (turb > lims[1]))
                turb[out] = np.nan

        elif alg == 'Jiang':
            if mode_Jiang == 'pixel':
                turb = ifunc.spm_jiang2021(Aerosol=rrs_dict['Aerosol'].values,
                                           Blue=rrs_dict['Blue'].values,
                                           Green=rrs_dict['Green'].values,
                                           Red=rrs_dict['Red'].values,
                                           RedEdge2=rrs_dict['RedEdge2'].values,
                                           Nir2=rrs_dict['Nir2'].values, mode=mode_Jiang)
            elif mode_Jiang == 'polygon':
                for i, shape in enumerate(shapefile):
                    values, slices, mask = self.extract_px(rasterio_rast=rasterio_rast, shapefile=shape,
                                                           rrs_dict=rrs_dict, bands=['Aerosol', 'Blue', 'Green',
                                                                                     'Red', 'RedEdge2', 'Nir2'])
                    # Verifying if there are more pixels than the minimum
                    valid_pixels = np.isnan(values[0]) == False
                    if np.count_nonzero(valid_pixels) >= min_px:
                        out = ifunc.spm_jiang2021(Aerosol=values[0].reshape(mask.shape),
                                                  Blue=values[1].reshape(mask.shape),
                                                  Green=values[2].reshape(mask.shape),
                                                  Red=values[3].reshape(mask.shape),
                                                  RedEdge2=values[4].reshape(mask.shape),
                                                  Nir2=values[5].reshape(mask.shape), mode=mode_Jiang).flatten()
                        # classifying only the valid pixels inside the polygon
                        values = np.where(valid_pixels, out, 0)
                        # adding to avoid replacing values of cropping by other polygons
                        turb[slices[0], slices[1]] += values.reshape(mask.shape)
            if limits:
                lims = [0.1, 1000]
                out = np.where((turb < lims[0]) | (turb > lims[1]))
                turb[out] = np.nan

        # removing espurious values and zeros
        out = np.where((turb == 0) | np.isinf(turb))
        turb[out] = np.nan

        return turb

    # def secchi_dd(self, rrs_dict, class_owt_spt, upper_lim=50, lower_lim=0):
    #     """
    #     Function to calculate the Secchi disk depth (SDD) based on the optical water type (OWT)
    #
    #     Parameters
    #     ----------
    #     rrs_dict: rrs_dict: a xarray Dataset containing the Rrs bands
    #     class_owt_spt: an array, with the same size as the input bands, with the OWT pixels
    #
    #     Returns
    #     -------
    #     secchi: an array, with the same size as the input bands, with the modeled values
    #     """
    #     import getpak.inversion_functions as ifunc
    #     secchi = np.zeros(rrs_dict['Rrs_B4'].shape, dtype='float32')
    #
    #     # spm functions for each OWT
    #     classes = [1, 4, 5, 6, 2, 7, 8, 11, 12, 3, 9, 10, 13]
    #     index = np.where(np.isin(class_owt_spt, classes))
    #     if len(index[0] > 0):
    #         secchi[index] = ifunc.functions['SDD_Lee']['function'](Red=rrs_dict['Rrs_B4'].values[index])
    #
    #     # removing espurious values and zeros
    #     if isinstance(upper_lim, (int, float)) and isinstance(lower_lim, (int, float)):
    #         out = np.where((secchi < lower_lim) | (secchi > upper_lim))
    #         secchi[out] = np.nan
    #
    #     out = np.where((secchi == 0) | np.isinf(secchi))
    #     secchi[out] = np.nan
    #
    #     return secchi

    @staticmethod
    def water_colour(rrs_dict, sensor='S2MSI', bands=['Blue', 'Green', 'Red', 'RedEdge1']):
        """
        Function to calculate the water colour of each pixel based on the Forel-Ule scale, using the bands of Sentinel-2
        MSI, with coefficients derived by linear correlation by van der Woerd and Wernand (2018). The different
        combinations of S2 bands (10, 20 or 60 m resolution) in the visible spectrum can be used, with the default
        being at 20 m.

        Parameters
        ----------
        @rrs_dict: rrs_dict: a xarray Dataset containing the Rrs bands
        @sensor: a string of the satellite mission, one of:
            S2MSI for Sentinel-2 MSI A and B
            S3OLCI for Sentinel-3 OLCI A and B
        @bands: an array containing the bands of the rrs_dict to be extracted, in order from blue to red edge

        Returns
        -------
        @colour: an array, with the same size as the input bands, with the classified pixels
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
        if sensor == 'S2MSI':
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
            classes[i] = int(Methods._Forel_Ule_scale(alpha_corrected[i], sensor))
        # adding the classes to the matrix
        colour[nzero] = classes

        return colour

    @staticmethod
    def _Forel_Ule_scale(angle, sensor='S2MSI'):
        """
        Calculates the Forel-Ule water colour scale, depending on the hue angle calculated,
        based on the classification by Novoa et al. (2013)
        Parameters
        ----------
        angle: the hue angle, in degrees

        Returns
        -------
        The water colour class (1-21)
        """
        if sensor == 'S2MSI':
            mapping = [(21.0471, 21), (24.4487, 20), (28.2408, 19), (32.6477, 18), (37.1698, 17), (42.3707, 16),
                       (47.8847, 15), (53.4431, 14), (59.4234, 13), (64.9378, 12), (70.9617, 11), (78.1648, 10),
                       (88.5017, 9), (99.5371, 8), (118.5208, 7), (147.4148, 6), (178.7020, 5), (202.8305, 4),
                       (217.1473, 3), (224.8037, 2)]
        score = lambda s: next((L for x, L in mapping if s < x), 1)

        return score(angle)

    # Water mask intersecting methods
    def sch_date_matchups(self, fst_dates, snd_dates, fst_tile_list, snd_tile_list):
        """
        Function to search for the match-up dates of two sets of images given two sets of dates.
        This function also writes the directories of the matchups for each date

        Parameters
        ----------
        @fst_dates: list of dates of the first set of images
        @snd_dates: list of dates of the second set of images
        @fst_tile_list: list of the path to the first set of images
        @snd_tile_list: list of the path to the first set of images

        Returns
        -------
        @matches, str_matches: Two dicts which the keys are the matchups dates, and the values are the paths to the
        images of the matchups
        @dates: list of the matchup dates
        """
        matches = {}
        str_matches = {}  # STR dict to avoid -> TypeError: Object of type PosixPath is not JSON serializable
        dates = []
        for n, date in enumerate(fst_dates):
            arr_index = np.where(np.array(snd_dates) == date)[0]
            if len(arr_index) > 0:
                # first check if the date is repeating
                if date not in matches:
                    matches[date] = {'IMG': fst_tile_list[n], 'WM': snd_tile_list[arr_index[0]]}
                    str_matches[date] = {'IMG': str(fst_tile_list[n]),
                                         'WM': str(snd_tile_list[arr_index[0]])}  # redundant backup
                    dates.append(date)
                else:
                    date2 = date + '_2'
                    matches[date2] = {'IMG': fst_tile_list[n], 'WM': snd_tile_list[arr_index[0]]}
                    str_matches[date2] = {'IMG': str(fst_tile_list[n]),
                                         'WM': str(snd_tile_list[arr_index[0]])}  # redundant backup
                    dates.append(date2)

        print(f'Found {len(dates)} match-ups\n')
        # self.log.info(f'Found {len(dates)} match-ups\n')

        return matches, str_matches, dates

    @staticmethod
    def get_waterdetect_masks(input_folder, output_folder=None):
        """
        This function finds all WaterDetect water masks in a folder, getting their dates and paths. It also gives the
        option to rename the water masks from their original folder (outputs from the WaterDetect processing) to a new
        folder, if there is a valid path for output folder

        Parameters
        ----------
        @input_folder: folder where the waterdetect masks are
        @output_folder: folder to copy (only) the water masks to

        Returns
        -------
        wd_dates: list of dates of the water masks
        wd_masks_list: list of the path to water masks
        """
        from pathlib import Path
        import shutil
        wd_dates, wd_masks_list = [], []
        if isinstance(output_folder, str):
            # Assure path to file exists, if not, create it.
            Path(os.path.dirname(output_folder)).mkdir(parents=True, exist_ok=True)
            for root, dirs, files in os.walk(input_folder, topdown=False):
                for name in files:
                    if name.endswith('.tif') and '_water_mask' in name:
                        f = Path(os.path.join(root, name))
                        newname = f.parent.parent.name + '_water_mask.tif'
                        dest_plus_name = os.path.join(output_folder, newname)
                        # copying to new folder
                        shutil.copyfile(f, dest_plus_name)
                        # print(f'COPYING: {f} TO: {dest_plus_name}\n')
                        # appending the date and path
                        nome = f.parent.parent.name.split('_')  
                        # check because for MAJA the dates are in position 2,
                        #  while for other products it is 3
                        date = nome[1][0:8] if nome[1][0] == '2' else nome[2][0:8]
                        wd_dates.append(date)
                        wd_masks_list.append(Path(dest_plus_name))

            print(f'Copied {len(wd_masks_list)} water masks to: {output_folder}\n')
            # self.log.info(f'Copied {len(wd_masks_list)} water masks to: {output_folder}\n')
        else:
            for file in os.listdir(input_folder):
                if file.endswith('.tif') and '_water_mask' in file:
                    f = Path(os.path.join(input_folder, file))
                    # appending the date and path
                    nome = file.split(
                        '_')  # check because for MAJA the dates are in position 2, while for other products it is 3
                    date = nome[1][0:8] if nome[1][0] == '2' else nome[2][0:8]
                    wd_dates.append(date)
                    wd_masks_list.append(f)
            print(f'Found {len(wd_masks_list)} water masks in {input_folder}\n')
            # self.log.info(f'Found {len(wd_masks_list)} water masks {input_folder}\n')

        return wd_dates, wd_masks_list

    @staticmethod
    def copy_waterdetect_invalidmasks(input_folder, output_folder):
        """
        Function to find all invalid masks from waterdetect in a folder, get their dates, and copy them to a
        # new folder with a new name. This function also writes the path of the water masks for each date

        Parameters
        ----------
        @input_folder: folder where the waterdetect masks are
        @output_folder: folder to copy (only) the invalid masks to
        """
        from pathlib import Path
        import shutil
        wd_masks_list = []
        for root, dirs, files in os.walk(input_folder, topdown=False):
            for name in files:
                if name.endswith('.tif') and '_invalid_mask' in name:
                    f = Path(os.path.join(root, name))
                    dest_plus_name = os.path.join(output_folder, name)
                    # copying to new folder
                    shutil.copyfile(f, dest_plus_name)
                    wd_masks_list.append(Path(dest_plus_name))

        print(f'Copied {len(wd_masks_list)} invalid masks to: {output_folder}\n')

        return None

    # @staticmethod
    # def intersect_watermask(rrs_dict, water_mask_dir):
    #     """
    #     Find all invalid masks from WaterDetect in a folder, get their dates,
    #     and copy them to a new folder with a new name. Also writes the path 
    #     of the water masks for each date

    #     Parameters
    #     ----------
    #     @input_folder: folder where the waterdetect masks are
    #     @output_folder: folder to copy (only) the invalid masks to
    #     """
    #     # Loading WD mask
    #     ref_data = rasterio.open(str(water_mask_dir))
    #     wd_mask = ref_data.read(1)
    #     wd_trans = ref_data.transform
    #     wd_proj = ref_data.crs
    #     ref_data = None

    #     # intersecting the GRS data with the waterdetect mask
    #     if wd_trans == rrs_dict.attrs['trans'] and wd_proj == rrs_dict.attrs['proj']:
    #         img = rrs_dict.where(wd_mask == 1).persist()
    #         print(f'Done intersection with water mask.')
    #     else:
    #         print(f'The water mask in not on the same tile as input image!')
    #         print(f'W.Mask-trans:{wd_trans} / Rrs-trans:{rrs_dict.attrs["trans"]}')
    #         print(f'W.Mask-proj:{wd_proj} / Rrs-proj:{rrs_dict.attrs["proj"]}')
    #         img = None
    #         sys.exit(1)

    #     return img
    
    @staticmethod
    def intersect_watermask(rrs_dict, water_mask_dir):
        """
        Test overlap before reprojecting/intersecting water mask to Rrs data using rioxarray.

        Returns
        -------
        xarray.DataArray or None
            Masked Rrs data if overlap exists, else None
        """
        # Load WaterDetect mask
        wd_mask = rxr.open_rasterio(water_mask_dir, masked=True).squeeze()

        # Ensure Rrs has CRS and spatial info
        rrs_dict = rrs_dict.rio.write_crs(rrs_dict.attrs['proj'], inplace=False)

        # Check bounding box overlap before reprojecting
        wd_bounds = box(*wd_mask.rio.bounds())
        rrs_bounds = box(*rrs_dict.rio.bounds())

        if not wd_bounds.intersects(rrs_bounds):
            print("No spatial overlap between Rrs image and water mask.")
            return None

        # Reproject water mask to match Rrs
        wd_mask_matched = wd_mask.rio.reproject_match(rrs_dict)

        # Now mask Rrs using the reprojected water mask
        img = rrs_dict.where(wd_mask_matched == 1).persist()

        # Optional: Check if mask actually selected any pixels
        # if img['Red'].notnull().sum().compute().item() == 0:
        if img['Red'].notnull().any().compute() == False:
            print("Water mask does not cover any valid Rrs pixels.")
            return None
        
        print(" Done intersection with water mask.")
        return img
