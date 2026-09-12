"""Small, explicit extension point for user-defined demonstration equations.

Functions are ordinary Python callables registered below.  Settings select
registered identifiers and scalar coefficients; settings never contain code.
"""
from dataclasses import dataclass
import ast
import hashlib
import inspect
import json
import marshal
import math
from pathlib import Path

import numpy as np


CANONICAL_BANDS = frozenset({
    "Aerosol", "Blue", "Green", "Red", "RedEdge1", "RedEdge2",
    "RedEdge3", "Nir1", "Nir2", "Swir1", "Swir2",
})


@dataclass(frozen=True)
class CustomEquationSpec:
    identifier: str
    function: object
    required_bands: tuple
    coefficients: dict
    units: str
    description: str
    equation_version: str
    encoding: dict

    @property
    def implementation_fingerprint(self):
        """Return a deterministic identity for the supported equation boundary."""
        module = inspect.getmodule(self.function)
        module_source = ""
        module_path = getattr(module, "__file__", None)
        if module_path:
            try:
                module_source = Path(module_path).read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                module_source = ""
        def callable_payload(function):
            try:
                source = inspect.getsource(function)
            except (OSError, TypeError):
                source = ""
            code = getattr(function, "__code__", None)
            code_bytes = marshal.dumps(code) if code is not None else repr(function).encode("utf-8")
            return source.encode("utf-8") + b"\0" + code_bytes

        function_payload = callable_payload(self.function)
        helper_payloads = []
        globals_dict = getattr(self.function, "__globals__", {})
        code = getattr(self.function, "__code__", None)
        for name in sorted(getattr(code, "co_names", ())):
            helper = globals_dict.get(name)
            if not callable(helper) or helper is self.function:
                continue
            helper_payloads.append(
                name.encode("utf-8") + b"@" +
                str(getattr(helper, "__module__", "")).encode("utf-8") + b"=" +
                callable_payload(helper)
            )
        contract = {
            "identifier": self.identifier,
            "required_bands": list(self.required_bands),
            "units": self.units,
            "equation_version": self.equation_version,
            "encoding": self.encoding,
        }
        payload = json.dumps(contract, sort_keys=True, default=repr,
                             separators=(",", ":"))
        digest = hashlib.sha256()
        digest.update(module_source.encode("utf-8"))
        digest.update(b"\0")
        digest.update(function_payload)
        for helper_payload in helper_payloads:
            digest.update(b"\0helper:")
            digest.update(helper_payload)
        digest.update(b"\0")
        digest.update(payload.encode("utf-8"))
        return digest.hexdigest()[:16]


@dataclass(frozen=True)
class EquationEvaluation:
    """Optional equation return contract for domain-specific diagnostics."""

    values: object
    diagnostics: dict


def _safe_ratio(red_edge, red):
    red_edge, red = np.broadcast_arrays(
        np.asarray(red_edge, dtype=float), np.asarray(red, dtype=float)
    )
    result = np.full(red.shape, np.nan, dtype=float)
    valid = np.isfinite(red_edge) & np.isfinite(red) & (red != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(red_edge, red, out=result, where=valid)
    return result


def example_equation_01(RedEdge1, Red):
    """Dimensionless demonstration ratio: RedEdge1 / Red."""
    red = np.asarray(Red, dtype=float)
    return EquationEvaluation(
        _safe_ratio(RedEdge1, red),
        {"zero_denominator": int(np.count_nonzero(np.isfinite(red) & (red == 0)))},
    )


def example_equation_02(RedEdge1, Red, a=0.0, b=1.0, c=1.0):
    """Dimensionless demonstration polynomial: a + b*ratio + c*ratio**2."""
    ratio = _safe_ratio(RedEdge1, Red)
    with np.errstate(over="ignore", invalid="ignore"):
        values = a + b * ratio + c * ratio ** 2
    red = np.asarray(Red, dtype=float)
    return EquationEvaluation(
        values,
        {"zero_denominator": int(np.count_nonzero(np.isfinite(red) & (red == 0)))},
    )


CUSTOM_EQUATION_REGISTRY = {
    "example_equation_01": CustomEquationSpec(
        identifier="example_equation_01",
        function=example_equation_01,
        required_bands=("RedEdge1", "Red"),
        coefficients={},
        units="1",
        description="Dimensionless demonstration index, ratio RedEdge1 / Red; not calibrated.",
        equation_version="1.0",
        encoding={"dtype": "float32", "scale": 1.0, "offset": 0.0, "nodata": np.nan,
                  "profile": "custom-float32"},
    ),
    "example_equation_02": CustomEquationSpec(
        identifier="example_equation_02",
        function=example_equation_02,
        required_bands=("RedEdge1", "Red"),
        coefficients={"a": 0.0, "b": 1.0, "c": 1.0},
        units="1",
        description="Dimensionless demonstration polynomial; not a calibrated retrieval.",
        equation_version="1.0",
        encoding={"dtype": "float32", "scale": 1.0, "offset": 0.0, "nodata": np.nan,
                  "profile": "custom-float32"},
    ),
}


def _as_enabled(value):
    if value is None or str(value).strip() == "":
        return []
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raw = str(value).strip()
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            parsed = [item.strip() for item in raw.split(",") if item.strip()]
        items = parsed if isinstance(parsed, (list, tuple)) else [parsed]
    result = [str(item).strip().strip(chr(39) + chr(34)) for item in items]
    if any(not item for item in result):
        raise ValueError("[custom_equations] enabled contains an empty identifier.")
    return result


def resolve_configuration(config, supported_bands=None):
    """Validate and resolve the [custom_equations] section before processing."""
    section = dict(config.get("custom_equations", {}))
    enabled = _as_enabled(section.pop("enabled", []))
    if len(enabled) != len(set(enabled)):
        raise ValueError("[custom_equations] enabled contains duplicate identifiers.")
    standard = {"OWT", "OWTSPM", "Chla", "Turb", "HySPM", "Aerosol", "Blue",
                "Green", "Red", "RedEdge1", "RedEdge2", "RedEdge3", "Nir2"}
    unknown = [name for name in enabled if name not in CUSTOM_EQUATION_REGISTRY]
    if unknown:
        raise ValueError("Unknown custom equation(s): " + ", ".join(unknown))
    collisions = sorted(set(enabled) & standard)
    if collisions:
        raise ValueError("Custom equation identifier collides with standard product: " + ", ".join(collisions))
    supported = CANONICAL_BANDS if supported_bands is None else set(supported_bands)
    for name in enabled:
        missing = sorted(set(CUSTOM_EQUATION_REGISTRY[name].required_bands) - supported)
        if missing:
            raise ValueError(f"Custom equation {name!r} requires unsupported band(s): {', '.join(missing)}.")

    overrides = {}
    raw_coefficients = section.pop("coefficients", None)
    if raw_coefficients not in (None, ""):
        try:
            parsed = ast.literal_eval(str(raw_coefficients))
        except (SyntaxError, ValueError) as exc:
            raise ValueError("[custom_equations] coefficients must be a literal mapping.") from exc
        if not isinstance(parsed, dict):
            raise ValueError("[custom_equations] coefficients must be a mapping.")
        for product, values in parsed.items():
            if product not in CUSTOM_EQUATION_REGISTRY or not isinstance(values, dict):
                raise ValueError(f"Invalid coefficient mapping for custom equation {product!r}.")
            overrides.setdefault(product, {}).update(values)
    for key, value in section.items():
        if "." not in key:
            raise ValueError(f"Unknown [custom_equations] option {key!r}.")
        product, coefficient = key.split(".", 1)
        if product not in CUSTOM_EQUATION_REGISTRY:
            raise ValueError(f"Unknown custom equation in coefficient option {key!r}.")
        overrides.setdefault(product, {})[coefficient] = value

    resolved = []
    for name in enabled:
        spec = CUSTOM_EQUATION_REGISTRY[name]
        coefficients = dict(spec.coefficients)
        for coefficient, value in overrides.get(name, {}).items():
            if coefficient not in coefficients:
                raise ValueError(f"Unknown coefficient {coefficient!r} for custom equation {name!r}.")
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Coefficient {name}.{coefficient} must be finite.") from exc
            if not math.isfinite(value):
                raise ValueError(f"Coefficient {name}.{coefficient} must be finite.")
            coefficients[coefficient] = value
        resolved.append((spec, coefficients))
    return tuple(resolved)


def evaluate(spec, bands, coefficients):
    """Evaluate one product, returning Float32 values and diagnostics."""
    missing = [name for name in spec.required_bands if name not in bands]
    diagnostics = {
        "status": "success", "message": None, "required_bands": list(spec.required_bands),
        "missing_bands": missing, "zero_denominator": 0, "invalid_count": 0,
        "undefined_count": 0, "float32_overflow_count": 0,
        "finite_output_count": 0,
    }
    try:
        # Only required inputs participate in conversion and shape calculation.
        raw_arrays = {
            name: np.asarray(bands[name], dtype=float)
            for name in spec.required_bands if name in bands
        }
        shape = np.broadcast_shapes(*(value.shape for value in raw_arrays.values())) \
            if raw_arrays else ()
    except Exception as exc:
        diagnostics.update(
            status="failed",
            message=f"{spec.identifier} required-band input is malformed: {exc}",
        )
        return np.full((), np.nan, dtype=np.float32), diagnostics

    if missing:
        diagnostics.update(status="missing_required_band",
                           message="Missing required band(s): " + ", ".join(missing))
        return np.full(shape, np.nan, dtype=np.float32), diagnostics
    try:
        arrays = {
            name: np.broadcast_to(raw_arrays[name], shape)
            for name in spec.required_bands
        }
        finite = np.ones(shape, dtype=bool)
        for value in arrays.values():
            finite &= np.isfinite(value)
        diagnostics["invalid_count"] = int(np.count_nonzero(~finite))
        evaluated = spec.function(**arrays, **coefficients)
        explicit = {}
        if isinstance(evaluated, EquationEvaluation):
            evaluated, explicit = evaluated.values, dict(evaluated.diagnostics or {})
        elif isinstance(evaluated, tuple) and len(evaluated) == 2 and isinstance(evaluated[1], dict):
            evaluated, explicit = evaluated
        for key, value in explicit.items():
            if key in diagnostics:
                diagnostics[key] = int(value) if key.endswith("_count") or key == "zero_denominator" else value
        result = np.asarray(evaluated, dtype=float)
        if result.shape != shape:
            result = np.broadcast_to(result, shape)
        with np.errstate(over="ignore", invalid="ignore"):
            too_large = finite & np.isfinite(result) & (np.abs(result) > np.finfo(np.float32).max)
            cast = result.astype(np.float32)
        cast_invalid = finite & np.isfinite(result) & ~np.isfinite(cast)
        diagnostics["float32_overflow_count"] = int(np.count_nonzero(too_large | cast_invalid))
        diagnostics["undefined_count"] = int(np.count_nonzero(finite & ~np.isfinite(result)))
        valid_output = finite & np.isfinite(result) & ~too_large & ~cast_invalid
        cast[~valid_output] = np.nan
        diagnostics["finite_output_count"] = int(np.count_nonzero(np.isfinite(cast)))
        return cast, diagnostics
    except Exception as exc:
        diagnostics.update(status="failed", message=f"{spec.identifier} evaluation failed: {exc}")
        return np.full(shape, np.nan, dtype=np.float32), diagnostics


def evaluate_enabled(resolved, bands):
    return {spec.identifier: (spec, *evaluate(spec, bands, coefficients), coefficients)
            for spec, coefficients in resolved}
