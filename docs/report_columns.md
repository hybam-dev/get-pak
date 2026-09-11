# Excel report columns

GET-Pak writes one workbook per requested ROI with exactly two worksheets. Both sheets use the same chronological row order and link through record_id.

| Worksheet | Column or pattern | Units | Meaning | Missing values |
| --- | --- | --- | --- | --- |
| Water quality | record_id | text | Stable short identity for the reported scene | Never blank for a discovered product |
| Water quality | scene_uid | text | Full processor/scene identity from output metadata or ledger | Blank only when unavailable |
| Water quality | acquisition_datetime_utc | Excel datetime | Source acquisition time in UTC | A successful consolidated output cannot be undated |
| Water quality | acquisition_date | Excel date | Date portion of source acquisition time | Blank only with an acquisition error |
| Water quality | acquisition_time_utc | Excel time | UTC time portion of source acquisition time | Blank only with an acquisition error |
| Water quality | Chla_*, Turb_*, HySPM_* | mg m-3, NTU, mg L-1 | ROI min, max, mean, count, standard deviation, median, status, and ROI feature counts | Numeric measurements are blank; status identifies missing or empty ROI data |
| Water quality | Water_pixels, Neg_Rrs_B4, Low_Rrs, OWT_1 | count | Valid-pixel and quality-filter indicators from the pixel sidecar | Blank when the sidecar is unavailable |
| Water quality | Rrs-band *_* columns | sr-1 | Requested Rrs ROI statistics after primary water-quality products | Blank when report_rrs is false or a product is unavailable |
| Processing details | source_path, mask_path | path text | Source scene and water-mask provenance | Blank when unavailable |
| Processing details | processor, processor_version, platform, tile, source_product_name | text | Processing and source identity | Blank when metadata is unavailable |
| Processing details | *_encoding_version, *_encoding_profile | text | Actual raster encoding schema/profile for each product | Blank for missing products |
| Processing details | *_applied_multiplier, *_decode_multiplier, *_decode_source, *_nodata, *_physical_unit | numeric/text | Actual factor used for decoding, its reciprocal, source (embedded_metadata or settings_fallback), no-data, and unit | Blank for missing products |
| Processing details | rrs_diagnostics_* and scaling_* | mixed | Per-band diagnostics, pre-encoding extrema, invalid/overflow counts, and raster-write details | Blank where a diagnostic does not apply |
| Processing details | status, reason, error, acquisition_status | text | Scene and report outcomes | Blank when no message applies |

For continuous products, a stored value is decoded as stored / multiplier. Standard units are Rrs sr-1, Chl-a mg m-3, turbidity NTU, and HySPM mg L-1. Missing measurements remain blank; zero remains a valid numeric value. The workbook has no unnamed DataFrame index column, no merged data cells, active sheet Water quality, frozen leading columns/header, and filters on both sheets.

Example leading row:

| record_id | scene_uid | acquisition_datetime_utc | acquisition_date | acquisition_time_utc | Chla_mean (mg m-3) |
