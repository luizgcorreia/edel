"""Projection-related computation functions."""


def get_projection_xy(df, method: str = "diffusion", scale: float = 8.0):
    """Extract projected X and Y coordinates from the DataFrame."""
    x_col = f"proj_p_{method}_x"
    y_col = f"proj_p_{method}_y"

    x = df[x_col].values * scale
    y = df[y_col].values * scale
    return x, y
