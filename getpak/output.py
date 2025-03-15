import importlib_resources


class Raster:
    """
    Generic class containing methods for spatial manipulations

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

    def __init__(self, parent_log=None):
        # if parent_log:
        #     self.log = parent_log
        # else:
        #     INSTANCE_TIME_TAG = datetime.now().strftime('%Y%m%dT%H%M%S')
        #     logfile = os.path.join(os.getcwd(), 'getpak_raster_' + INSTANCE_TIME_TAG + '.log')
        #     self.log = u.create_log_handler(logfile)

        # Import CRS projection information from /data/s2_proj_ref.json
        s2projdata = importlib_resources.files(__name__).joinpath('data/s2_proj_ref.json')
        with s2projdata.open('rb') as fp:
            byte_content = fp.read()
        self.s2projgrid = json.loads(byte_content)

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
        img_parent_name = os.path.basename(Path(img_path_str).parents[1])
        sliced_ipn = img_parent_name.split('_')
        tile_id = sliced_ipn[5][1:]
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
