import os
import rasterio
import numpy as np

from getpak import gdal
from pathlib import Path
from dask.distributed import Client as dkClient, LocalCluster
from rasterio.warp import calculate_default_transform, reproject, Resampling

# GET-Pak imports
from getpak import Utils

u = Utils()

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
        pass

    @staticmethod
    def s2_to_tiff(ndarray_data, output_img, no_data=0, gdal_driver_name="GTiff",
                   tile_id=None, img_ref=None):
    ##TODO: improve resolution handling.
    # Current dictionary only handles 20m (5490x5490),
    # future versions should also handle 10m (10980x10980).
        """
        Given an input ndarray and destination to save the output,
        generate a raster.tif using GDT_Float32.
        """ 

        if tile_id:
            tile_metadata = u.get_tile_s2_projection(tile_id)
            trans = tile_metadata['trans']
            proj = tile_metadata['proj']
        elif img_ref:
            # Gather information from the template file
            ref_data = gdal.Open(img_ref)
            trans = ref_data.GetGeoTransform()
            proj = ref_data.GetProjection()
        else:
            raise ValueError('Either tile_id or img_ref must be provided')

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
        # self.log.info('')
        pass

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
                           options=[compression]) as file:
            file.write(ndarray_data, 1)
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
