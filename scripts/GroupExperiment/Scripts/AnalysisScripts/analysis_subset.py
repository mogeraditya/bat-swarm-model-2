import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    return mo, np, pd, plt, sns


@app.cell
def _(pd):
    scores_dataframe = pd.read_csv("./scripts/GroupExperiment/Data/AnalysisData/summary_merged_19_11.csv")
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
        # "LOUDNESS_TIME_DELAY_CAP",
        "control_type",
        "arena_size_scaling",
        # "BAT_FRONTAL_RANGE",
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
def _():
    # plt.figure(figsize=(12,6))
    # plt.subplot(1,2,1)
    # sns.lineplot(data=scores_dataframe, x="control_type", y="distance travelled", hue="MEMORY_WINDOW_FOR_CONSISTENCY")
    # plt.subplot(1,2,2)
    # sns.lineplot(data=scores_dataframe, x="control_type", y="interbat collision total", hue="MEMORY_WINDOW_FOR_CONSISTENCY")
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
    sns.lineplot(data=scores_dataframe, x="NUM_BATS", y="distance_covered", hue="control_type")
    plt.subplot(1,2,2)
    sns.lineplot(data=scores_dataframe, x="NUM_BATS", y="time_spent_rotating", hue="control_type")
    return


@app.cell
def _(scores_dataframe, sns):
    sns.countplot(data=scores_dataframe,x="NUM_BATS", hue="arena_size_scaling")
    return


@app.cell
def _(normalized_df, sns):
    sns.lineplot(data=normalized_df[normalized_df["arena_size_scaling"]=="perimeter"], x="NUM_BATS", y="wall collision total", hue="control_type")
    return


@app.cell
def _(normalized_df, sns):
    sns.lineplot(data=normalized_df[normalized_df["arena_size_scaling"]=="perimeter"], x="NUM_BATS", y="interbat collision total", hue="control_type")
    return


@app.cell
def _(normalized_df, sns):
    sns.violinplot(data=normalized_df, x="control_type", y="wall collision per bat", hue="arena_size_scaling")
    return


@app.cell
def _(mo):
    arena_scaling = ["density", "perimeter", "fixed"]
    slider_param = mo.ui.slider(start=0, stop=len(arena_scaling)-1, step=1, label="arean scaling")
    slider_consistency = mo.ui.slider(start=1, stop=7, step=2, label="memory window for consistency", value=3)
    mo.vstack([slider_param, slider_consistency])
    return arena_scaling, slider_consistency, slider_param


@app.cell
def _(arena_scaling, scores_dataframe, slider_consistency, slider_param):
    df_slider = scores_dataframe[(scores_dataframe["arena_size_scaling"]==arena_scaling[slider_param.value])&(scores_dataframe["MEMORY_WINDOW_FOR_CONSISTENCY"]==slider_consistency.value)]
    # df_slider
    return (df_slider,)


@app.cell
def _(df_slider, plt, sns):
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    g1 = sns.lineplot(data=df_slider, x="NUM_BATS", y="wall collision total", hue="control_type", errorbar="pi")
    g1.set(yscale="log")
    plt.subplot(1,2,2)
    sns.lineplot(data=df_slider, x="NUM_BATS", y="interbat collision total", hue="control_type", errorbar="pi")
    return


@app.cell
def _(df_slider, plt, sns):
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    sns.lineplot(data=df_slider, x="NUM_BATS", y="duration", hue="control_type",  errorbar="pi")
    plt.subplot(1,2,2)
    sns.lineplot(data=df_slider, x="NUM_BATS", y="space_occupied", hue="control_type", errorbar="pi")
    return


@app.cell
def _(arena_scaling, mo, slider_consistency, slider_param):
    controls = ["null_control", "negative_control", "positive_control", "treatment"]
    slider_control = mo.ui.slider(start=0, stop=len(arena_scaling), step=1, label="control type")
    mo.vstack([slider_param, slider_consistency, slider_control])
    return controls, slider_control


@app.cell
def _(arena_scaling, controls, scores_dataframe, slider_control, slider_param):
    df_slider2 = scores_dataframe[(scores_dataframe["arena_size_scaling"]==arena_scaling[slider_param.value])&(scores_dataframe['control_type']==controls[slider_control.value])]
    return (df_slider2,)


@app.cell
def _(df_slider2, plt, sns):
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    g3 = sns.lineplot(data=df_slider2, x="NUM_BATS", y="wall collision total", hue="MEMORY_WINDOW_FOR_CONSISTENCY", errorbar="pi")
    # g2.set(yscale="log")
    plt.subplot(1,2,2)
    sns.lineplot(data=df_slider2, x="NUM_BATS", y="duration", hue="MEMORY_WINDOW_FOR_CONSISTENCY", errorbar="pi")
    return


@app.cell
def _(df_slider, plt, sns):
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    g2 = sns.lineplot(data=df_slider, x="NUM_BATS", y="wall collision total", hue="MEMORY_WINDOW_FOR_CONSISTENCY", errorbar="pi")
    # g2.set(yscale="log")
    plt.subplot(1,2,2)
    sns.lineplot(data=df_slider, x="NUM_BATS", y="interbat collision total", hue="control_type", errorbar="pi")
    return


if __name__ == "__main__":
    app.run()
