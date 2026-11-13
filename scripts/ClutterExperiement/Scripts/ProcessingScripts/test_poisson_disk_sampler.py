import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import scipy as sc
    import numpy as np
    import seaborn as sns

    return np, sns


@app.cell
def _(np):

    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from scipy.stats import qmc

    rng = np.random.default_rng()
    radius = 0.7
    engine = qmc.PoissonDisk(d=2, radius=radius, rng=rng, u_bounds=[9,5])
    sampled_points = engine.fill_space()
    return (sampled_points,)


@app.cell
def _(sampled_points, sns):
    sns.scatterplot(x=sampled_points[:,0], y=sampled_points[:,1])
    return


if __name__ == "__main__":
    app.run()
