"""GeoShift evaluation harness.

Implements the tasks, nulls and controls of methodology v7
(SHA-256 DB331CA8AE9FAEC6C022ED26ECBBCC9026EE003339BD68B874D46149A7FC9FEC).

Nothing here chooses a fold. Fold definitions are read from the pre-registered CSVs
in 05-prereg, which were built and hashed before any model was fitted.
"""
__version__ = "7.0.0"
METHODOLOGY_SHA256 = (
    "DB331CA8AE9FAEC6C022ED26ECBBCC9026EE003339BD68B874D46149A7FC9FEC")
