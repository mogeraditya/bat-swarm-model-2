import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", auto_download=["html", "ipynb"])


@app.cell
def _():
    import numpy as np
    import marimo as mo
    import seaborn as sns
    import scipy as scp
    import pandas as pd
    import matplotlib.pyplot as plt
    import altair as alt
    from mpl_toolkits.mplot3d import Axes3D
    from tqdm import tqdm 
    import plotly.graph_objects as go
    import plotly.express as px
    from scipy.spatial import ConvexHull
    import matplotlib as mpl
    # mpl.rcParams['font.family'] = 'cursive'
    mpl.rcParams['font.size'] = 10
    return mo, np, pd, plt, px, sns, tqdm


@app.cell
def _(pd):
    scores_dataframe = pd.read_csv("./Data/AnalysisData/summary_merged_26_10_2026.csv")
    scores_dataframe = scores_dataframe[scores_dataframe["distance travelled"]>70]
    scores_dataframe["seed"] = scores_dataframe["seed"].astype("category")


    return (scores_dataframe,)


@app.cell
def _():
    # input_param = "seed"
    # score_cols = ["wall collision total", "space_occupied", "duration"]
    # # df=scores_dataframe.copy()
    # scores_dataframe[score_cols] = scores_dataframe.groupby([c for c in varying_params if c != input_param], dropna=False)[score_cols].transform("mean")
    return


@app.cell
def _(np):
    varying_params = np.array([
        "CALL_RATE_FAST",
        "CALL_DIRECTIONALITY",
        "EMITTED_SPL",
        "HEARING_THRESHOLD",
        "BAT_RADIAL_RESOLUTION",
        "BAT_ANGULAR_RESOLUTION",
        "BAT_FRONTAL_RANGE",
        "CONSISTENCY_TIME_DELAY_CAP",
        "MEMORY_WINDOW_FOR_CONSISTENCY",
    ])
    labels = np.array(['fast call rate (Hz)', 'call directionality', 'emitted spl', 'hearing threshold (dB)', 'radial resolution', 'angular resolution', 'frontal range (degrees)', 'consistency time delay cap', 'memory window'])
    number_of_params = len(varying_params)
    return labels, number_of_params, varying_params


@app.cell
def _(np, tqdm):
    def find_pareto_front(df, obj_cols):
        """
        Find Pareto optimal points in a DataFrame.

        Parameters:
        - df: DataFrame with columns in obj_cols
        - obj_cols: list of column names: ['wall collision total', 'space_occupied', 'duration']

        Returns:
        - Boolean mask: True for Pareto optimal points
        """
        # Extract objective values
        data = df[obj_cols].values  # shape: (n, 3)
        n = len(data)

        # Initialize mask: assume all are non-dominated initially
        is_pareto = [True] * n

        # Compare each point against every other point
        for i in tqdm(range(n), desc="running pareto computation:"):
            for j in range(n):
                if i == j:
                    continue
                # Check if point j dominates point i
                # We want: j has better or equal in all, and strictly better in at least one
                # But note: we want to minimize collision and duration, maximize space
                # So we reverse the logic for space_occupied (maximize)
                if (data[j, 0] <= data[i, 0] and  # j has ≤ collisions
                    data[j, 1] >= data[i, 1] and  # j has ≥ space
                    data[j, 2] <= data[i, 2] and  # j has ≤ duration
                    (data[j, 0] < data[i, 0] or
                     data[j, 1] > data[i, 1] or
                     data[j, 2] < data[i, 2])):  # at least one strict improvement
                    is_pareto[i] = False
                    break  # i is dominated, no need to check further
        return np.array(is_pareto)

    return (find_pareto_front,)


@app.cell
def _(find_pareto_front, scores_dataframe):
    obj_list = ['wall collision total', 'space_occupied', 'duration']
    pareto_solutions_ids = find_pareto_front(scores_dataframe, obj_list)
    pareto_solutions_df = scores_dataframe[pareto_solutions_ids]
    pareto_solutions_df
    return pareto_solutions_df, pareto_solutions_ids


@app.cell
def _(plt, scores_dataframe, sns):
    sns.jointplot(scores_dataframe, x="wall collision total", y="space_occupied", kind="kde", color="red")
    sns.scatterplot(scores_dataframe, x="wall collision total", y="space_occupied", alpha =0.05, color="yellow")
    plt.ylabel("space occupied")
    plt.xlabel("collision counts")
    # plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./kde_scatterplot.png", transparent= True)
    plt.show()
    return


@app.cell
def _(plt, scores_dataframe, sns):
    # plt.figure(figsize=(10,10))
    sns.jointplot(scores_dataframe, x="wall collision total", y="duration", kind="kde", color="red")
    sns.scatterplot(scores_dataframe, x="wall collision total", y="duration", alpha =0.05, color="yellow")
    plt.ylabel("duration")
    plt.xlabel("collision counts")
    # plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./kde_scatterplot_2.png", transparent= True)
    plt.show()
    return


@app.cell
def _(pareto_solutions_df, plt, sns):
    sns.jointplot(pareto_solutions_df, x="wall collision total", y="space_occupied", kind="kde", color="red")
    sns.scatterplot(pareto_solutions_df, x="wall collision total", y="space_occupied", alpha =0.05, color="yellow")
    plt.ylabel("space occupied")
    plt.xlabel("collision counts")
    # plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./kde_scatterplot.png", transparent= True)
    plt.show()
    return


@app.cell
def _(pareto_solutions_df, plt, sns):
    # plt.figure(figsize=(10,10))
    sns.jointplot(pareto_solutions_df, x="wall collision total", y="duration", kind="kde", color="red")
    sns.scatterplot(pareto_solutions_df, x="wall collision total", y="duration", alpha =0.05, color="yellow")
    plt.ylabel("duration")
    plt.xlabel("collision counts")
    # plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./kde_scatterplot_2.png", transparent= True)
    plt.show()
    return


@app.cell
def _(px):
    def plot_interactive_pareto(summary_df, is_pareto, output_file, optimized_params, plot_non_pareto=True):
        """
        Interactive 3D Pareto plot with highlighted Pareto optimal points and surface.

        Objectives:
        - Minimize wall collision total
        - Maximize space_occupied
        - Minimize duration
        """
        # Define objective columns
        obj_cols = ["wall collision total", "space_occupied", "duration"]

        # Ensure all columns exist
        if not all(col in summary_df.columns for col in obj_cols):
            raise ValueError(f"Missing required columns: {obj_cols}")

        # Step 1: Find Pareto optimal points
        # is_pareto = pareto_df
        pareto_df = summary_df[is_pareto].copy()
        non_pareto_df = summary_df[~is_pareto].copy()

        # Step 2: Prepare hover data
        hover_data = {"job id": True}
        hover_data.update({param: True for param in optimized_params})

        fig = px.scatter_3d(
            pareto_df,
            x="wall collision total",
            y="space_occupied",
            z="duration",
            hover_data=hover_data,
            opacity=1,
            size_max = 0.1,
            color_discrete_sequence=["red"],
        )

        # Step 3: Create base scatter plot (non-Pareto points)
        if plot_non_pareto:
            non_pareto_scatter = px.scatter_3d(
                non_pareto_df,
                x="wall collision total",
                y="space_occupied",
                z="duration",
                hover_data=hover_data,
                opacity=0.5,
                size_max = 0.1,
                color_discrete_sequence=["lightgray"],
            )
            for trace in non_pareto_scatter.data:
                fig.add_trace(trace)
        # Step 4: Add Pareto points (highlighted)


        # Add Pareto points to figure

        fig.update_traces(marker=dict(size=1))
        # Step 5: Fit a surface (Convex Hull approximation of Pareto front)
        # Only if we have enough points
        # if len(pareto_df) >= 4:
        #     points = pareto_df[obj_cols].values  # (n, 3)
        #     try:
        #         hull = ConvexHull(points)
        #         # Create a mesh for the convex hull surface
        #         for simplex in hull.simplices:
        #             # Each simplex is a triangle (3 points)
        #             p1, p2, p3 = points[simplex]
        #             # Add triangle to the figure
        #             fig.add_trace(go.Mesh3d(
        #                 x=[p1[0], p2[0], p3[0]],
        #                 y=[p1[1], p2[1], p3[1]],
        #                 z=[p1[2], p2[2], p3[2]],
        #                 color='rgba(255, 165, 0, 0.2)',
        #                 opacity=0.5,
        #                 showlegend=True
        #             ))
        #     except Exception as e:
        #         print("Could not compute convex hull:", e)

        # Step 6: Update layout
        fig.update_layout(
            width=900,
            height=800,
            title="Pareto Front: Pareto Optimal Solutions (Red) with Surface Approximation",
            scene=dict(
                xaxis_title="Wall Collisions (counts)",
                yaxis_title="Space Occupied (counts)",
                zaxis_title="Duration (s)",
            ),
            legend_title_text="Legend",
            margin=dict(l=0, r=0, b=0, t=50)
        )

        # Save and show
        fig.write_html(output_file)
        fig.show()


    return (plot_interactive_pareto,)


@app.cell
def _(
    pareto_solutions_ids,
    plot_interactive_pareto,
    scores_dataframe,
    varying_params,
):
    plot_interactive_pareto(scores_dataframe, pareto_solutions_ids,"./scatter_plot.html", varying_params, plot_non_pareto=False)
    return


@app.cell
def _(pareto_solutions_df, sns):
    g = sns.JointGrid(data=pareto_solutions_df, x="wall collision total", hue="space_occupied", y="duration")
    g.plot(sns.scatterplot, sns.histplot)
    # g.plot_joint(sns.kdeplot,
    #              fill=True, clip=((0, 15), (60, 170)),
    #              thresh=0, levels=100, cmap="rocket")
    # g.plot_marginals(sns.histplot,kde=True, color="#03051A", alpha=1, bins=25)
    return


@app.cell
def _(pareto_solutions_df, sns):
    g2 = sns.JointGrid(data=pareto_solutions_df, x="wall collision total", y="duration", space=0)
    g2.plot_joint(sns.kdeplot,
                 fill=True, clip=((0, 15), (10000,70000)),
                 thresh=0, levels=100, cmap="rocket")
    g2.plot_marginals(sns.histplot, kde=True, color="#03051A", alpha=1, bins=25)
    return


@app.cell
def _(pareto_solutions_df, sns):
    sns.pairplot(pareto_solutions_df[['wall collision total', 'space_occupied', 'duration']])
    return


@app.cell
def _(mo, varying_params):
    slider_param = mo.ui.slider(start=0, stop=len(varying_params)-1, step=1, label="param")
    slider_collision = mo.ui.slider(start=0, stop=15, step=1, label="wall collision cap", value=3)
    mo.vstack([slider_param, slider_collision])
    return slider_collision, slider_param


@app.cell
def _(pareto_solutions_df, slider_collision):
    pareto_solutions_df[pareto_solutions_df["wall collision total"]<=slider_collision.value][pareto_solutions_df["duration"]<30000][pareto_solutions_df["space_occupied"]>0]
    return


@app.cell
def _(pairplot_pareto_df, pareto_solutions_df, slider_collision):
    print(list(pareto_solutions_df[pareto_solutions_df["wall collision total"]<=slider_collision.value][pairplot_pareto_df["duration"]<30000][pairplot_pareto_df["space_occupied"]>110]["job id"]))
    return


@app.cell
def _(focus_parameter, pairplot_pareto_df, slider_collision, sns):
    print(focus_parameter)
    sns.pairplot(pairplot_pareto_df[pairplot_pareto_df["wall collision total"]<=slider_collision.value], hue=focus_parameter)
    return


@app.cell
def _():
    return


@app.cell
def _(pareto_solutions_df, slider_param, sns, varying_params):
    focus_parameter = varying_params[slider_param.value]
    print(focus_parameter)
    pairplot_pareto_df = pareto_solutions_df[['wall collision total', 'space_occupied', 'duration', focus_parameter]]
    sns.pairplot(pairplot_pareto_df, hue=focus_parameter)
    return focus_parameter, pairplot_pareto_df


@app.cell
def _(pareto_solutions_df, sns):
    sns.histplot(data=pareto_solutions_df, x="wall collision total")
    return


@app.cell
def _(
    labels,
    np,
    number_of_params,
    pareto_solutions_df,
    plt,
    sns,
    varying_params,
):
    plt.figure(figsize=(25,20))

    number_of_subfigs = int(np.ceil(number_of_params/3))
    bat_speeds = [1,3,5]
    for i_d, param_d in enumerate(varying_params):
   
        plt.subplot(3,number_of_subfigs,i_d+1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        sns.countplot(pareto_solutions_df, x=param_d, stat="percent")
        plt.ylim(0,100)
        plt.xlabel(labels[list(varying_params).index(param_d)])
        # plt.legend(title="balls")
        # plt.legend(title= labels[list(varying_params).index(param_d)])

    plt.savefig("param_counts_10.svg", transparent= True)
    plt.show()
    plt.clf()
    return (number_of_subfigs,)


@app.cell
def _(slider_collision):
    slider_collision
    return


@app.cell
def _(
    labels,
    number_of_subfigs,
    pareto_solutions_df,
    plt,
    slider_collision,
    sns,
    varying_params,
):
    plt.figure(figsize=(15,10))
    for i_p, param_p in enumerate(varying_params):
   
        plt.subplot(3,number_of_subfigs,i_p+1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        sns.countplot(pareto_solutions_df[pareto_solutions_df["wall collision total"]<slider_collision.value], x=param_p, stat="count")
        # plt.ylim(0,100)
        plt.xlabel(labels[list(varying_params).index(param_p)])
        # plt.legend(title="balls")
        # plt.legend(title= labels[list(varying_params).index(param_d)])

    plt.savefig("param_counts_10.svg", transparent= True)
    plt.show()
    plt.clf()
    return


@app.cell
def _(
    pareto_solutions_df,
    plot_interactive_pareto,
    slider_collision,
    varying_params,
):
    plot_interactive_pareto(pareto_solutions_df, pareto_solutions_df["wall collision total"]<slider_collision.value,"./scatter_plot_pareto_subset.html", varying_params, plot_non_pareto=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
