"""Phase 1: Generate atmosphere→spectrum training data using petitRADTRANS.

Usage
-----
    python scripts/generate_data.py
    python scripts/generate_data.py data.n_samples=10000 data.n_workers=8
"""

from __future__ import annotations

from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

from plato_fno.physics.petitrt_driver import generate_dataset_hdf5


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    output_path = Path(cfg.data.output_path)

    logger.info(f"Generating {cfg.data.n_samples} atmosphere–spectrum pairs")
    logger.info(f"Wavelength range: {cfg.data.wavelength_range} µm, {cfg.data.n_wavelengths} points")
    logger.info(f"Species: {list(cfg.data.species)}")
    logger.info(f"Workers: {cfg.data.n_workers}, Chunks: {cfg.data.chunk_size}")
    logger.info(f"Output: {output_path}")

    generate_dataset_hdf5(
        output_path=output_path,
        n_samples=cfg.data.n_samples,
        wavelength_range=tuple(cfg.data.wavelength_range),
        n_wavelengths=cfg.data.n_wavelengths,
        n_layers=cfg.data.n_layers,
        species=list(cfg.data.species),
        seed=cfg.seed,
        n_workers=cfg.data.n_workers,
        chunk_size=cfg.data.chunk_size,
    )

    logger.success(f"Dataset written to {output_path}")


if __name__ == "__main__":
    main()
