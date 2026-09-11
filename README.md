# GET-Pak

GET-Pak turns atmospherically corrected Sentinel-2 MSI imagery into maps and summaries of inland-water quality. It produces suspended particulate matter, turbidity, chlorophyll-a, reflectance, and optical water type (OWT) products, and can summarize them over regions of interest (ROIs).

It supports automated batch processing from GRS NetCDF products and from ACOLITE L2R NetCDF products. A SeaDAS reader is available for interactive workflows; SeaDAS batch processing is not part of the settings-driven command. GET-Pak does not run GRS or ACOLITE itself.

![GET-Pak workflow](img/get-pak-workflow.png)

## Install

GET-Pak depends on GDAL and other geospatial libraries. A Conda environment is the simplest setup on Linux:

```bash
conda env create -f environment.yml
conda activate getpak-review
```

Copy or adapt `settings.ini`. At minimum, set the input, output, and water-mask folders, the tile, the processor, and any ROI shapefiles. For GRS, also set the GRS version (for example, `v20`).

Run the settings-driven workflow with:

```bash
getpak -c settings.ini run
# or, from a source checkout:
python main.py -c settings.ini run
```

The command uses the values in the configuration file. Set `compute_l2b = True` to create Level-2B products and `make_report = True` to create ROI reports. Set `report_rrs = True` when reflectance-band rasters and diagnostics are wanted.

## Input folders

For a GRS batch, the input root must contain a directory for the tile, with one or more NetCDF scenes inside it:

```text
inputs/
└── 50RKU/
    └── S2A_MSIL1C_20210911T025551_N0500_R032_T50RKU_20230116T101431.nc
```

The configured tile is `50RKU`; the required GRS input layout is <inputs>/<tile>/*.nc, for example <inputs>/50RKU/*.nc. GET-Pak reads the acquisition time from the source metadata and uses it for filenames and reports.

For ACOLITE, point `inputs` to a directory containing external `*_L2R.nc` products and set `ac_processor = ACOLITE`. ACOLITE products must contain usable sensor, tile, acquisition-time, projection, and reflectance metadata. Keep GRS and ACOLITE outputs in separate directories when comparing processors.

A minimal ACOLITE processing section is:

```ini
[processing]
ac_processor = ACOLITE
compute_l2b = True
make_report = True
report_rrs = True
parallel = False
s2_tile = 20LLQ
s2_resolution = 20
```

## Water masks and ROIs

By default, GET-Pak finds one WaterDetect mask for each scene date and tile. An exact acquisition timestamp is preferred when masks contain one. Ambiguous candidates are reported instead of being silently guessed.

To use one deliberate reference mask for all scenes, enable static-mask mode and give its path:

```ini
mask_mode = static
static_mask_path = /absolute/path/to/reference_water_mask.tif
```

A static mask must overlap every scene, cover it as required, and contain compatible binary classes. Grid differences are aligned with nearest-neighbour resampling. No overlap, partial coverage, invalid classes, and all-zero masks are reported per scene.

Set `roi_vectors` to one or more shapefile paths when ROI statistics or an Excel report is needed.
## GRS grid validation

For GRS products, GET-Pak validates finite one-dimensional projected x/y coordinates, dimensions, metre units, monotonic direction, and uniform spacing before mask alignment or export. Coordinates are treated as pixel centres, so the selected GeoTIFF transform uses the signed centre spacing and half-pixel edge offset. A stale source GeoTransform is replaced only when this regular projected-grid contract is unambiguous; ambiguous or irregular grids fail with a scene-level diagnostic.

The scene ledger and output metadata record the original and selected transforms and the validation reason. Outputs created before a detected grid/georeferencing repair may have incorrect spatial support and should be regenerated through the full L2B pipeline with the corrected reader; do not repair them by changing headers or reports alone.


## Outputs and filenames

Products are written below `output/<tile>/` in their product directories (`OWT`, `OWTSPM`, `Chla`, `Turb`, `HySPM`, and optional Rrs-band directories). New raster names follow this pattern:

```text
<Product>_<YYYYMMDDTHHMMSS>_T<tile>_<record_id>.tif
```

For example:

```text
Chla_20210911T025551_T50RKU_82e157365cf2261f6ada.tif
```

The timestamp is the source acquisition time in UTC, `record_id` is the existing short output identifier, and the full `scene_uid` remains in metadata and the scene ledger. Platform, processor, and processor version are kept in metadata rather than added to the filename.

Each run also writes a scene ledger named `<run>_scene_ledger.json`, a timing manifest, and a result summary under `output/<tile>/`. ROI workbooks are written there as `<run>_<roi-name>.xlsx`.

## Raster values and no-data

New rasters use encoding version `GETPAK-ENC-2`. Continuous products are unsigned 16-bit rasters with `65535` as no-data; zero is a valid physical value. Categorical OWT products are unsigned 8-bit rasters with `255` as no-data; class zero is preserved when it is a valid class.

| Product | Stored multiplier | Decode physical value | No-data | Unit |
| --- | ---: | ---: | ---: | --- |
| Rrs bands | 10,000 | stored value × 0.0001 | 65,535 | sr-1 |
| Chlorophyll-a | 100 | stored value × 0.01 | 65,535 | mg m-3 |
| Turbidity | 10 | stored value × 0.1 | 65,535 | NTU |
| HySPM | 10 | stored value × 0.1 | 65,535 | mg L-1 |
| OWT and OWTSPM | 1 | stored class code | 255 | class code |

The GeoTIFF stores its actual multiplier, zero offset, unit, no-data, product, encoding version/profile, valid range, overflow count, source acquisition time, tile, provenance identity, and source product name. A raw array reader must decode exactly once using the embedded metadata. If authoritative multiplier metadata is absent, GET-Pak uses the product multiplier from the same [output_encoding] dictionary and records that settings_fallback was used. Rasterio's generic default scale of 1.0 is not treated as explicit metadata. Invalid or contradictory embedded encoding metadata, including a nonzero offset, is an error. Do not apply continuous scaling to OWT classes. Values that are non-finite, invalid under the existing quality rules, or outside the representable physical range become no-data and are counted in the metadata.

The storage range protects against integer wraparound; it does not establish scientific validity of a retrieval at the highest concentration that can be stored.

## Encoding settings

The optional [output_encoding] section is the only encoding configuration. When it is absent, the standard values below are used. Supplied multipliers must be finite and positive; continuous storage remains uint16 with 65535 no-data, and categorical storage remains uint8 with 255 no-data.

```ini
[output_encoding]
encoding_version = GETPAK-ENC-2
rrs_multiplier = 10000
chla_multiplier = 100
turbidity_multiplier = 10
hyspm_multiplier = 10
continuous_dtype = uint16
continuous_nodata = 65535
categorical_dtype = uint8
categorical_nodata = 255
```

The effective profile, physical resolution (1 / multiplier), and maximum (65534 / multiplier) are shown when settings are read. Standard multipliers use profile standard; changed multipliers use profile custom under the same GETPAK-ENC-2 schema. For example, multiplier 5 supports 7000 mg L-1 at 0.2 mg L-1 resolution for HySPM, and multiplier 50 supports 1000 mg m-3 at 0.02 mg m-3 resolution for Chl-a. These are supported storage examples, not claims about retrieval validity.

## Reports

Each ROI report is one XLSX workbook with exactly two worksheets: Water quality (first and active) and Processing details. Both sheets begin with record_id, scene_uid, acquisition_datetime_utc, acquisition_date, and acquisition_time_utc. Water-quality statistics use physical units and retain valid-pixel counts and quality indicators; requested Rrs statistics follow the primary products. Processing details retain source/mask paths, processor and platform metadata, encoding factors and fallback sources, per-band diagnostics, and outcomes.

Dates come from raster acquisition metadata first and the standardized output filename second. They are actual Excel date/time values, never run time or file modification time. Rows are sorted by acquisition time and identity, with frozen leading columns and filters. Missing measurements are blank, while readable status fields distinguish missing data from valid zeros. See docs/report_columns.md for the final columns and meanings.


## Troubleshooting

**No inputs found:** confirm that the processor is correct, the configured tile matches the folder name, and GRS scenes are under `inputs/<tile>/`. ACOLITE inputs must be `*_L2R.nc` and contain matching tile metadata.

**Mask mismatch or ambiguity:** check the mask date, tile, and acquisition timestamp. Remove duplicate candidates or enable static-mask mode with a compatible reference mask.

**No valid pixels:** inspect the mask coverage and ROI, confirm that the input is atmospherically corrected, and check whether quality filtering removed all water pixels. The scene ledger records the per-scene reason.

## Citation and research

Auxiliary data for reproducibility is available at [Zenodo](https://doi.org/10.5281/zenodo.20933323), including the Liangzi Lake ROI, GRS images, and pre-computed water masks.

The current release uses methods described in:

- Harmel, T., Chami, M., Tormos, T., Reynaud, N., Danis, P.-A., 2018. Sunglint correction of MSI-Sentinel-2 imagery over inland and sea waters. *Remote Sensing of Environment* 204, 308–321. [doi:10.1016/j.rse.2017.10.022](https://doi.org/10.1016/j.rse.2017.10.022)
- Tavares, M.H., Guimarães, D., Roussillon, J., Baute, V., Cucherousset, J., Boulêtreau, S., Martinez, J.-M., 2025. A framework to retrieve water-quality parameters in small, optically diverse freshwater ecosystems using Sentinel-2 MSI imagery. *Remote Sensing* 17, 2729. [doi:10.3390/rs17152729](https://doi.org/10.3390/rs17152729)
- Cordeiro, M.C.R., Martinez, J.-M., Peña-Luque, S., 2021. Automatic water detection from multidimensional hierarchical clustering for Sentinel-2 images. *Remote Sensing of Environment* 253, 112209.
