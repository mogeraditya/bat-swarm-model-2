import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    return np, pd, plt, sns


@app.cell
def _(pd):
    scores_dataframe = pd.read_csv("./scripts/GroupExperiment/Data/AnalysisData/summary_merged_17_11.csv")
    scores_dataframe["seed"] = scores_dataframe["seed"].astype("category")
    # scores_dataframe = scores_dataframe[scores_dataframe["MEMORY_WINDOW_FOR_CONSISTENCY"] == 5]
    return (scores_dataframe,)


@app.cell
def _(np):
    varying_params = np.array([
        "NUM_BATS",
        "seed",
        "CONSISTENCY_TIME_DELAY_CAP",
        "MEMORY_WINDOW_FOR_CONSISTENCY",
        "LOUDNESS_TIME_DELAY_CAP",
        "control_type",
        ])
    return


@app.cell
def _():
    # input_param = "seed"
    # score_cols = ["wall collision total", "interbat collision total","space_occupied", "duration"]
    # df=scores_dataframe.copy()
    # scores_dataframe[score_cols] = scores_dataframe.groupby([c for c in varying_params if c != input_param], dropna=False)[score_cols].transform("mean")
    return


@app.cell
def _():
    return


@app.cell
def _(scores_dataframe, sns):
    sns.pairplot(scores_dataframe[['wall collision total', 'space_occupied', 'NUM_BATS']])
    return


@app.cell
def _(scores_dataframe, sns):
    sns.lineplot(data=scores_dataframe, x="NUM_BATS", y="wall collision total", hue="control_type")
    return


@app.cell
def _(scores_dataframe, sns):
    sns.lineplot(data=scores_dataframe, x="NUM_BATS", y="interbat collision total", hue="control_type")
    return


@app.cell
def _(plt, scores_dataframe, sns):
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    sns.lineplot(data=scores_dataframe, x="NUM_BATS", y="wall collision total", hue="control_type")
    plt.subplot(1,2,2)
    sns.lineplot(data=scores_dataframe, x="NUM_BATS", y="interbat collision total", hue="control_type")
    return


@app.cell
def _(plt, scores_dataframe, sns):
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    sns.lineplot(data=scores_dataframe[scores_dataframe["NUM_BATS"]==100], x="control_type", y="wall collision total", hue="MEMORY_WINDOW_FOR_CONSISTENCY")
    plt.subplot(1,2,2)
    sns.lineplot(data=scores_dataframe, x="control_type", y="interbat collision total", hue="MEMORY_WINDOW_FOR_CONSISTENCY")
    return


@app.cell
def _(plt, scores_dataframe, sns):
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    sns.lineplot(data=scores_dataframe[scores_dataframe["NUM_BATS"]==100], x="control_type", y="distance travelled", hue="MEMORY_WINDOW_FOR_CONSISTENCY")
    plt.subplot(1,2,2)
    sns.lineplot(data=scores_dataframe, x="control_type", y="interbat collision total", hue="MEMORY_WINDOW_FOR_CONSISTENCY")
    return


@app.cell
def _(plt, scores_dataframe, sns):
    plt.figure(figsize=(12,6))
    normalized_df = scores_dataframe.copy()
    normalized_df["wall collision per bat"] = scores_dataframe["wall collision total"]/scores_dataframe["NUM_BATS"]
    normalized_df["interbat collision per bat"] = scores_dataframe["interbat collision total"]/scores_dataframe["NUM_BATS"]
    plt.subplot(1,2,1)
    sns.lineplot(data=normalized_df, x="NUM_BATS", y="wall collision per bat", hue="control_type")
    plt.subplot(1,2,2)
    sns.lineplot(data=normalized_df, x="NUM_BATS", y="interbat collision per bat", hue="control_type")
    return (normalized_df,)


@app.cell
def _(plt, scores_dataframe, sns):
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    sns.lineplot(data=scores_dataframe, x="NUM_BATS", y="distance travelled", hue="control_type")
    plt.subplot(1,2,2)
    sns.lineplot(data=scores_dataframe, x="NUM_BATS", y="time_spent_rotating", hue="control_type")
    return


@app.cell
def _(plt, scores_dataframe):
    plt.hist(scores_dataframe["control_type"])
    return


@app.cell
def _(normalized_df, sns):
    sns.boxplot(data=normalized_df, x="NUM_BATS", y="wall collision per bat", hue="control_type")
    return


@app.cell
def _(normalized_df, sns):
    sns.boxplot(data=normalized_df, x="control_type", y="wall collision per bat", hue="MEMORY_WINDOW_FOR_CONSISTENCY")
    return


if __name__ == "__main__":
    app.run()
