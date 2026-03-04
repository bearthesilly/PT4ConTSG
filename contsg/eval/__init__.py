"""
Evaluation package for ConTSG.

This package provides comprehensive evaluation infrastructure for
conditional time series generation models.

Components:
- Evaluator: Full-featured evaluator with metric orchestration
- MetricRunner: Metric lifecycle management
- CLIPEmbedder: CLIP embedding extraction wrapper
- StatsCache: Reference statistics caching for Frechet metrics
- CaseStudyVisualizer: Case study visualization (GT vs Generated)
- metrics: Comprehensive metric collection (28 metrics)

Example:
    from contsg.eval import Evaluator

    # Load from experiment
    evaluator = Evaluator.from_experiment("experiments/exp1/")

    # Initialize CLIP for CLIP-dependent metrics
    evaluator.init_clip()

    # Run evaluation
    results = evaluator.evaluate(
        metrics=["dtw", "fid", "cttp"],
        n_samples=10,
        sampler="ddpm"
    )
"""

from __future__ import annotations

# Core evaluation components
from contsg.eval.evaluator import (
    Evaluator,
    MetricRunner,
    BatchEvaluationData,
    EvaluationContext,
)

# CLIP integration
from contsg.eval.embedder import CLIPEmbedder
from contsg.eval.stats_cache import StatsCache

# Visualization
from contsg.eval.visualizer import CaseStudyVisualizer, VisualizationCase

# Import metrics package to trigger registration
from contsg.eval import metrics


__all__ = [
    # Core evaluator
    "Evaluator",
    "MetricRunner",
    "BatchEvaluationData",
    "EvaluationContext",
    # CLIP components
    "CLIPEmbedder",
    "StatsCache",
    # Visualization
    "CaseStudyVisualizer",
    "VisualizationCase",
    # Metrics (re-exported from subpackage)
    "metrics",
]
