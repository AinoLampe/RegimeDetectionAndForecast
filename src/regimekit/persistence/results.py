"""Result writers/readers (initial: the prepared dataset of `regimekit fetch`).

`prepared.parquet` layout: one row per observation; columns `time` (timestamp[ns]), `obs_<m>`,
`mask_<m>`, `dt`, `cov_<entry point>_<k>` and `label_<n>`; everything else (`clock`, `nominal_dt`,
covariate names per entry point, `snapshot_ids`, `prep_stats`, the Part II hooks) is JSON in the
schema metadata under `regimekit`. Arrays round-trip bit for bit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from regimekit.data.contracts import BoundaryConstraints, PreparedDataset
from regimekit.data.prepare import prep_stats_from_json, prep_stats_to_json
from regimekit.utils.errors import PersistenceError

_META_KEY = b"regimekit"
_FORMAT = 1
_ENTRY_POINTS = ("transition", "duration", "emission")


def write_prepared(dataset: PreparedDataset, path: Path | str) -> Path:
    """Write `dataset` to a parquet file (parent directories are created).

    Returns:
        The written path (so `read_prepared(write_prepared(ds, p)) == ds`).
    """
    path = Path(path)
    observations = np.asarray(dataset.observations)
    mask = np.asarray(dataset.mask)
    columns: dict[str, Any] = {"time": pa.array(np.asarray(dataset.times, dtype="datetime64[ns]"))}
    for m in range(observations.shape[1]):
        columns[f"obs_{m}"] = pa.array(observations[:, m])
        columns[f"mask_{m}"] = pa.array(mask[:, m])
    columns["dt"] = pa.array(np.asarray(dataset.dt))
    blocks: dict[str, int | None] = {}
    for ep in _ENTRY_POINTS:
        block = getattr(dataset, f"cov_{ep}")
        blocks[ep] = None if block is None else int(block.shape[1])
        if block is not None:
            values = np.asarray(block)
            for k in range(values.shape[1]):
                columns[f"cov_{ep}_{k}"] = pa.array(values[:, k])
    n_labels = None
    if dataset.labels is not None:
        labels = np.asarray(dataset.labels)
        n_labels = int(labels.shape[1])
        for n in range(n_labels):
            columns[f"label_{n}"] = pa.array(labels[:, n])
    boundaries = dataset.boundaries
    meta = {
        "format": _FORMAT,
        "n_series": int(observations.shape[1]),
        "clock": dataset.clock,
        "nominal_dt": dataset.nominal_dt,
        "blocks": blocks,
        "cov_names": {k: list(v) for k, v in dataset.cov_names.items()},
        "snapshot_ids": list(dataset.snapshot_ids),
        "prep_stats": prep_stats_to_json(dataset.prep_stats),
        "support": dataset.support,
        "n_categories": dataset.n_categories,
        "n_labels": n_labels,
        "boundaries": None
        if boundaries is None
        else {
            "must_switch_at": list(boundaries.must_switch_at),
            "must_not_switch_at": list(boundaries.must_not_switch_at),
        },
        "group": dataset.group,
    }
    table = pa.table(columns).replace_schema_metadata({_META_KEY: json.dumps(meta).encode("utf-8")})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def read_prepared(path: Path | str) -> PreparedDataset:
    """Read a dataset written by `write_prepared`.

    Raises:
        PersistenceError: missing file, not a regimekit dataset, or an unsupported format version.
    """
    path = Path(path)
    try:
        table = pq.read_table(path)
    except (OSError, pa.ArrowInvalid) as exc:
        raise PersistenceError(f"cannot read prepared dataset {path}: {exc}", path=str(path)) from exc
    raw = (table.schema.metadata or {}).get(_META_KEY)
    if raw is None:
        raise PersistenceError(f"{path} is not a regimekit prepared dataset", path=str(path))
    meta = json.loads(raw)
    if meta.get("format") != _FORMAT:
        raise PersistenceError(f"unsupported prepared-dataset format {meta.get('format')!r}", path=str(path))

    def col(name: str) -> np.ndarray:
        return np.asarray(table.column(name).to_numpy())

    def stack(names: list[str], dtype: type) -> np.ndarray:
        if not names:  # a (T, 0) block (CLAUDE.md §3.7) has no columns
            return np.empty((table.num_rows, 0), dtype=dtype)
        return np.stack([col(n) for n in names], axis=1)

    n_series = meta["n_series"]
    observations = np.stack([col(f"obs_{m}") for m in range(n_series)], axis=1)
    mask = np.stack([col(f"mask_{m}") for m in range(n_series)], axis=1)
    blocks: dict[str, Any] = {}
    for ep in _ENTRY_POINTS:
        k = meta["blocks"][ep]
        blocks[ep] = (
            None if k is None else jnp.asarray(stack([f"cov_{ep}_{i}" for i in range(k)], np.float64))
        )
    labels = None
    if meta["n_labels"] is not None:
        labels = jnp.asarray(stack([f"label_{n}" for n in range(meta["n_labels"])], np.bool_))
    b = meta["boundaries"]
    return PreparedDataset(
        times=col("time").astype("datetime64[ns]"),
        observations=jnp.asarray(observations),
        dt=jnp.asarray(col("dt")),
        mask=jnp.asarray(mask),
        cov_transition=blocks["transition"],
        cov_duration=blocks["duration"],
        cov_emission=blocks["emission"],
        cov_names={k: tuple(v) for k, v in meta["cov_names"].items()},
        clock=meta["clock"],
        nominal_dt=meta["nominal_dt"],
        prep_stats=prep_stats_from_json(meta["prep_stats"]),
        snapshot_ids=tuple(meta["snapshot_ids"]),
        support=meta["support"],
        n_categories=meta["n_categories"],
        labels=labels,
        boundaries=None
        if b is None
        else BoundaryConstraints(tuple(b["must_switch_at"]), tuple(b["must_not_switch_at"])),
        group=meta["group"],
    )
