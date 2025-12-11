"""
Sampling module for raster data.

Provides different sampling strategies for extracting pixel samples from rasters.
Handles random, stratified, and systematic sampling with reproducibility.
"""

from typing import Optional, Tuple, Literal
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class SamplingStrategy(str, Enum):
    """Valid sampling strategies for raster data."""

    random = "random"
    stratified = "stratified"
    systematic = "systematic"


class Sampling(BaseModel):
    """Sampling strategies for raster data.

    Provides different sampling strategies for extracting pixel samples from rasters.
    Handles random, stratified, and systematic sampling with reproducibility.

    Attributes
    ----------
    strategy : SamplingStrategy or str
        Sampling strategy: 'random', 'stratified', or 'systematic'
    n_samples : int, optional
        Number of samples to draw. If None, uses all pixels.
    seed : int, optional
        Random seed for reproducibility
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy: SamplingStrategy = Field(
        default=SamplingStrategy.random, description="Sampling strategy"
    )
    n_samples: Optional[int] = Field(
        default=10000, description="Number of samples to draw"
    )
    seed: Optional[int] = Field(default=None, description="Random seed")

    def __init__(
        self,
        strategy: str = "random",
        n_samples: Optional[int] = 10000,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """Initialize sampling configuration.

        Parameters
        ----------
        strategy : str or SamplingStrategy, optional
            Sampling strategy: 'random', 'stratified', or 'systematic' (default: 'random')
        n_samples : int, optional
            Number of samples to draw (default: 10000). If None, uses all pixels.
        seed : int, optional
            Random seed for reproducibility
        """
        # Convert string to enum if needed
        if isinstance(strategy, str):
            try:
                strategy = SamplingStrategy(strategy)
            except ValueError:
                valid_strategies = [s.value for s in SamplingStrategy]
                raise ValueError(
                    f"Invalid sampling strategy '{strategy}'. "
                    f"Must be one of: {', '.join(valid_strategies)}"
                )

        super().__init__(strategy=strategy, n_samples=n_samples, seed=seed, **kwargs)

    def sample_indices(
        self,
        valid_indices: Tuple[np.ndarray, np.ndarray],
        target_values: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample pixel indices according to the configured strategy.

        Parameters
        ----------
        valid_indices : Tuple[np.ndarray, np.ndarray]
            Tuple of (row_indices, col_indices) for valid pixels
        target_values : np.ndarray, optional
            Target variable values for stratified sampling

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (sampled_row_indices, sampled_col_indices)
        """
        # Set random seed if provided
        if self.seed is not None:
            np.random.seed(self.seed)

        n_valid = len(valid_indices[0])

        # If n_samples is None or greater than valid pixels, use all
        if self.n_samples is None or self.n_samples >= n_valid:
            print(f"  Using all {n_valid:,} valid pixels")
            return valid_indices

        # Sample according to strategy
        if self.strategy == SamplingStrategy.random:
            return self._sample_random(valid_indices, n_valid)

        elif self.strategy == SamplingStrategy.stratified:
            if target_values is None:
                raise ValueError("Stratified sampling requires target_values parameter")
            return self._sample_stratified(valid_indices, target_values)

        elif self.strategy == SamplingStrategy.systematic:
            return self._sample_systematic(valid_indices, n_valid)

        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")

    def _sample_random(
        self, valid_indices: Tuple[np.ndarray, np.ndarray], n_valid: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random sampling strategy."""
        sample_idx = np.random.choice(n_valid, size=self.n_samples, replace=False)
        sample_indices = (
            valid_indices[0][sample_idx],
            valid_indices[1][sample_idx],
        )
        print(f"  Sampled {self.n_samples:,} random pixels")
        return sample_indices

    def _sample_stratified(
        self, valid_indices: Tuple[np.ndarray, np.ndarray], target_values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stratified sampling strategy based on target variable classes."""
        # Get unique classes and their counts
        unique_classes, class_counts = np.unique(target_values, return_counts=True)
        n_per_class = self.n_samples // len(unique_classes)

        sampled_row_idx = []
        sampled_col_idx = []

        for cls in unique_classes:
            cls_indices = np.where(target_values == cls)[0]
            n_cls_samples = min(n_per_class, len(cls_indices))
            cls_sample_idx = np.random.choice(
                cls_indices, size=n_cls_samples, replace=False
            )

            sampled_row_idx.extend(valid_indices[0][cls_sample_idx])
            sampled_col_idx.extend(valid_indices[1][cls_sample_idx])

        sample_indices = (np.array(sampled_row_idx), np.array(sampled_col_idx))
        print(f"  Sampled {len(sampled_row_idx):,} stratified pixels")
        return sample_indices

    def _sample_systematic(
        self, valid_indices: Tuple[np.ndarray, np.ndarray], n_valid: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Systematic grid sampling strategy."""
        step = int(np.sqrt(n_valid / self.n_samples))
        sample_idx = np.arange(0, n_valid, step)
        sample_indices = (
            valid_indices[0][sample_idx],
            valid_indices[1][sample_idx],
        )
        print(f"  Sampled {len(sample_idx):,} systematic pixels")
        return sample_indices
