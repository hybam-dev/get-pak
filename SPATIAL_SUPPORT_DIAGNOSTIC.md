# May sparse ROI support diagnostic

Date: 2026-09-11. This is a read-only investigation of existing validation inputs and outputs. No QC threshold, retrieval algorithm, mask, or raster was changed.

## Inputs and rules

- ROI: /media/dvd/T7/GPK_50RKU_GLORIA_test-data/shp/liangzi_lake_epsg32650.shp, EPSG:32650, one polygon, bounds 251385.82,3335350.90 to 272150.18,3361624.96 m.
- May source: /home/dvd/getpak_consolidated_inputs_20260910_may/inputs/50RKU/S2B_MSIL2Agrs_20210509T025539_N0500_R032_T50RKU_20230309T130856.nc.
- September source: /home/dvd/getpak_consolidated_inputs_20260910_sep/inputs/50RKU/S2A_MSIL2Agrs_20210911T025551_N0500_R032_T50RKU_20230116T101431.nc.
- Masks: matching Sentinel-2B May and Sentinel-2A September WaterDetect rasters. Both are EPSG:32650, 5490 x 5490, 20 m, and contain only classes 0 and 1.
- The pipeline uses nearest-neighbour mask reprojection, accepts mask value 1, filters Red values below 0, and applies the existing low-Rrs threshold of 0.002 over Aerosol through RedEdge2. ROI statistics use rasterstats with all_touched=True.

## Retention evidence

| Stage | May | September | Denominator / interpretation |
| --- | ---: | ---: | --- |
| ROI pixels on the NetCDF coordinate grid | 413,201 | 413,201 | all_touched ROI pixels on the 20 m coordinate grid |
| ROI pixels inside source footprint | 413,201 | 413,201 | source FOOTPRINT polygon; no footprint clipping in either scene |
| finite source Red pixels in ROI | 413,201 | 413,201 | before GET-Pak masking or QC; no source fill/NaN loss observed |
| finite source Green pixels in ROI | 413,201 | 413,201 | before GET-Pak masking or QC |
| finite source pixels in all 8 required Rrs bands | 413,201 | 413,201 | before GET-Pak masking or QC |
| mask value 1 after nearest-neighbour alignment | 407,599 | 412,965 | over the coordinate-grid ROI; classes were only 0 and 1 |
| mask value 0 | 5,602 | 236 | over the coordinate-grid ROI |
| Red non-positive after water mask | 14 | 7 | over the coordinate-grid ROI; includes values removed by the existing Red rule |
| low-Rrs rejection after Red non-negative admission | 0 | 0 | over the coordinate-grid ROI; existing threshold/rules |
| Red support after these coordinate-grid rules | 407,585 | 412,958 | coordinate-grid diagnostic, not the mis-georeferenced May output grid |
| report-grid ROI candidate pixels | 12,531 | 413,201 | all_touched ROI pixels on the exported raster grid |
| exported Red / Green ROI counts | 63 / 63 | 412,965 / 412,965 | exact pipeline zonal-statistics counts from existing rasters |
| exported Chl-a / Turbidity / HySPM counts | 54 / 54 / 54 | 412,954 / 412,946 / 412,899 | exact pipeline zonal-statistics counts from existing rasters |

The observed report values therefore agree with the existing output rasters: 63 Red/Green pixels and 54 pixels for each reported water-quality product in May; 412,965 Red/Green, 412,954 Chl-a, 412,946 Turbidity, and 412,899 HySPM pixels in September.

## Spatial/grid finding

September is internally aligned: the source and output transform is 20 m, origin 199980,3400020, and bounds 199980,3290220 to 309780,3400020 m. The ROI is wholly inside that grid and retained output is distributed across the ROI bounds.

May has a material grid inconsistency in the existing source/read/write path:

- The 5490 x 5490 NetCDF Rrs coordinates have 20 m spacing and coordinate-grid bounds 199980,3290220 to 309780,3400020 m. The ROI is wholly inside this coordinate grid.
- GRS.get_grs_dict obtains a 10 m GDAL transform for the same 5490 x 5490 array and stores it as the raster transform. The rioxarray transform read from the result is also 10 m, with bounds approximately 199985,3290225 to 309775,3400015 m, while the x/y coordinate spacing remains 20 m.
- Exported May rasters use the 10 m transform, origin 199980,3400020, and bounds 199980,3345120 to 254880,3400020 m. The ROI extends to x=272150 m, so the exported grid intersects only x=253976.98 to 254880 m: about 0.9 km of the ROI width, versus about 20.8 km in the ROI geometry.
- Direct no-data scans of the existing May outputs place the retained Rrs patch at approximately x=253995 to 254865 m and y=3345125 to 3346075 m, rather than across the full ROI. September retained support spans approximately x=251390 to 272130 m and y=3335370 to 3361610 m.

This is a localized output-grid/ROI alignment and coverage problem. It explains the large drop from hundreds of thousands of finite, water-admitted coordinate-grid pixels to 63 exported Red/Green pixels. The additional reduction from 63 Rrs pixels to 54 water-quality pixels is a later product-validity difference already visible in the exported rasters; it is not evidence of source coverage loss.

The sidecar Water_pixels values, 4,139,096 for May and 3,880,521 for September, are not ROI denominators. In getpak.methods.filter_pixels, Water_pixels is assigned from the full masked Red array before ROI extraction, so these are whole output-array water-support counts. The ROI denominators above are stated separately.

## Cause and limitations

- Dominant observed loss: May output grid/georeferencing alignment and footprint clipping relative to the ROI.
- Not supported as the dominant cause: source radiometry/fill. All eight required source Rrs bands were finite at every one of the 413,201 coordinate-grid ROI pixels in this inspection.
- Not supported as the dominant cause: water-mask admission. May retained 407,599 of 413,201 coordinate-grid ROI pixels after nearest-neighbour mask alignment; the mask had only 0/1 values.
- Later QC and product validity contribute small losses in the coordinate-grid diagnostic and the exported L2B counts, but cannot explain the May-scale drop by themselves.
- The source attributes report CLOUD_COVERAGE_ASSESSMENT of 0.396% for May and 9.397% for September. No source cloud/shadow flag was used here to attribute the sparse May patch physically; cloud or shadow causation remains unproven from the inspected evidence.
- This diagnostic does not tune thresholds, add a minimum-pixel rule, select SPM residuals, or claim lake-wide spatial representativeness. It explains the existing difficult-input execution alongside the finishing patch.
