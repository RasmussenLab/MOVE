Functions
=========

Analysis
----------

.. autofunction:: move.analysis.metrics.calculate_accuracy

.. autofunction:: move.analysis.metrics.calculate_cosine_similarity

.. autofunction:: move.analysis.metrics.norm

.. autoclass:: move.analysis.metrics.ComputeAccuracyMetrics
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: move.analysis.feature_importance.FeatureImportance
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: move.analysis.fdr.argnearest

.. autofunction:: move.analysis.hdi.hdi_bounds

Visualization
---------------

.. autofunction:: move.visualization.figure.create_figure

.. autofunction:: move.visualization.style.style_settings

.. autofunction:: move.visualization.style.color_cycle

.. autofunction:: move.visualization.grid.generate_grid

.. autofunction:: move.visualization.grid.facet_grid

.. autofunction:: move.visualization.grid.find_grid_dimensions

.. autofunction:: move.visualization.loss_curves.plot_loss_curves

.. autofunction:: move.visualization.metrics.plot_metrics_boxplot

.. autofunction:: move.visualization.latent_space.plot_latent_space_with_cat

.. autofunction:: move.visualization.latent_space.plot_latent_space_with_con

.. autofunction:: move.visualization.feature_importance.plot_categorical_feature_importance

.. autofunction:: move.visualization.feature_importance.plot_continuous_feature_importance

.. autofunction:: move.visualization.scale.axis_scale

.. autofunction:: move.visualization.contrast.get_luminance

.. autofunction:: move.visualization.contrast.get_contrast_ratio

Core utilities
----------------

.. autofunction:: move.core.logging.get_logger

.. autofunction:: move.core.qualname.get_fully_qualname

.. autofunction:: move.core.seed.set_global_seed

.. autoclass:: move.core.exceptions.CudaIsNotAvailable
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: move.core.exceptions.ShapeAndWeightMismatch
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: move.core.exceptions.UnsetProperty
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: move.core.typing.EncodedData
    :members:
    :undoc-members:
