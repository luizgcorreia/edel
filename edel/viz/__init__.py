"""Visualization modules."""

from edel.viz.data import (
    plot_abstract_length_dist,
    plot_publication_year_dist,
    plot_citation_dist,
    plot_segmentation_stats,
    plot_filtering_stats,
    plot_language_dist,
    set_viz_style,
)
from edel.viz.projection import (
    plot_projection_2d,
    plot_transition_signatures,
    plot_movement_magnitudes,
    plot_epistemic_transition_space,
    plot_paper_style_pca,
    plot_diffusion_eigenvalues,
    plot_unified_discourse_space,
)
from edel.viz.vector_field import (
    plot_vector_field,
    plot_field_magnitude,
    plot_field_density,
)
from edel.viz.clustering import (
    plot_clusters_on_landscape,
    plot_field_clusters,
    plot_cluster_trajectories,
)
from edel.viz.labeling import (
    print_cluster_summaries,
    plot_epistemic_map,
)
from edel.viz.landscape import (
    plot_landscape_3d,
    plot_landscape_contour,
)
