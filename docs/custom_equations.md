# Custom equations

GET-Pak can run a small, explicit set of user-authored Python equations alongside the standard OWT products. It is an extension point, not an equation language or plugin framework.

## Enable and configure

Set the flag in [processing]:

    run_custom_equations = True

Select registered functions by identifier:

    [custom_equations]
    enabled = ['example_equation_01', 'example_equation_02']
    # Coefficients are finite scalar overrides using registered_id.coefficient.
    example_equation_02.a = 0
    example_equation_02.b = 1
    example_equation_02.c = 1

The shipped examples use the canonical physical Rrs arrays in sr-1 after scene QC:

- example_equation_01: RedEdge1 / Red.
- example_equation_02: a + b * ratio + c * ratio**2.

Their demonstration defaults produce ratio [2, 3] and polynomial [6, 12] for Red [0.01, 0.02] and RedEdge1 [0.02, 0.06]. They are dimensionless indices, not calibrated chlorophyll-a, turbidity, or SPM retrievals.

Each registry entry declares its identifier, callable, required canonical bands, coefficients, units, equation version, and encoding. A new function is added by editing getpak/custom_equations.py, registering the metadata, and selecting it in settings; automation.py and the report writer do not need edits. Duplicate identifiers, standard-product collisions, unknown coefficients, invalid values, and unavailable canonical bands fail before costly scene processing. GRS and ACOLITE expose the same canonical Sentinel-2 band names used by the existing readers; the full three-scene validation in this pass covers GRS.

## Outputs and validity

Each enabled equation writes a separate <identifier>/ GeoTIFF using the established acquisition/tile/record naming convention. Values are signed Float32, units 1, scale 1, offset 0, and NaN no-data under the GETPAK-ENC-2 custom-float32 profile. Zero and valid negative results are retained. Zero denominators, missing/invalid inputs, and nonrepresentable Float32 results become no-data and are counted in metadata and the scene ledger. Unexpected equation failures receive a product-specific failed status while standard products and other custom equations continue.

These examples do not reuse concentration multipliers or claim scientific calibration. Changing coefficients, equation version, or implementation fingerprint cannot silently reuse an existing custom raster.

## Reports

The existing single ROI workbook still has exactly Water quality and Processing details sheets. Custom min/max/mean/median/std/count fields are appended after standard water-quality groups. Processing details include equation identifier/version, resolved coefficients, required bands, encoding profile, implementation fingerprint, and failure/diagnostic information. The report can discover already-generated registered custom rasters without executing equations, even when run_custom_equations = False; setting the flag only controls new scene computation.

No arbitrary formula strings, eval/exec, arbitrary imports, new OWT selection, or second workbook is supported. The examples demonstrate software extensibility only; they have no water-quality calibration, full scientific validation, or reviewer acceptance.
