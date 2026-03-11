# Changelog

All notable changes to this project are documented in this file.

## [2.1.0] - 2025-01-20

### Changed

- aligned the public documentation with the current package interface
- standardized setup instructions on Poetry
- updated CLI examples to use separate building and ground-motion preprocessors
- documented the 100-response training interface and the smaller default optimization set

### Fixed

- cross-validation documentation now matches the fold-local preprocessing behavior
- optimization documentation now reflects the `--building-prep`, `--gm-prep`, and `--gm-strategy` interface

### Added

- smoke tests for config consistency and dataset loading

## [2.0.0] - 2025-01-20

### Added

- modular package structure under `src/`
- neural-network surrogate model and training pipeline
- NSGA-II optimization module
- Poetry-based packaging and development tooling

## [1.0.0] - 2024-06-26

Initial research dataset release.