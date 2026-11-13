"""Orchestrator: builds the figure, wires up whichever modules were chosen,
and drives the FuncAnimation loop exactly like the original `visualize()`.

Saving the animation is the slow part (every frame is a full matplotlib render
of several polar axes). When possible the frames are rendered in parallel
worker processes and piped, in order, into a single ffmpeg process. Each worker
builds the very same figure and renders each frame with the same `savefig`
call `FFMpegWriter.grab_frame` uses, so the encoded video is unchanged.
"""

import io
import multiprocessing
import os
import pickle
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cbook
from matplotlib.gridspec import GridSpec
from plotting.arena_module import ArenaModule
from plotting.common import stitch_together_history_lists
from plotting.info_text_module import InfoTextModule
from plotting.intensity import IntensityRadialModule
from plotting.ipi_radial_module import IPIRadialModule
from plotting.polarization_milling_module import PolarizationMillingModule
from plotting.sensory_volume_module import SensoryVolumeModule
from plotting.sum_radial_module import SumRadialModule
from tqdm.rich import tqdm

# Original behaviour = every panel included.
DEFAULT_MODULES = [
    ArenaModule,
    IPIRadialModule,
    SumRadialModule,
    SensoryVolumeModule,
    InfoTextModule,
    IntensityRadialModule,
]

# Handy string aliases, e.g. modules=["arena", "sensory_volume"]

AVAILABLE_MODULES = {
    "arena": ArenaModule,
    "ipi_radial": IPIRadialModule,
    "sum_radial": SumRadialModule,
    "sensory_volume": SensoryVolumeModule,
    "info_text": InfoTextModule,
    "polarization_milling": PolarizationMillingModule,
    "intensity": IntensityRadialModule,
}

# frames rendered per worker task; a raw 1600x900 RGBA frame is ~5.8 MB
_FRAMES_PER_TASK = 4
_MAX_AUTO_RENDER_WORKERS = 8


def _resolve_module(m):
    if isinstance(m, str):
        return AVAILABLE_MODULES[m]()
    if isinstance(m, type):
        return m()
    return m  # already an instance, e.g. ArenaModule(grid_pos=(0, 0))


def _build_figure(
    modules,
    history,
    parameters_df,
    bats,
    obstacles,
    jammers,
    focal_bat,
    show_sounds,
    resolution,
    rows,
    columns,
):
    """Create the figure, its axes and set up every module (shared by the main
    process and the render workers, so all of them draw the same figure)."""
    module_instances = [_resolve_module(m) for m in modules]

    fig = plt.figure(figsize=(16, 9))

    gridspec = GridSpec(rows, columns, figure=fig)

    for module in module_instances:
        if module.GRID_POS is not None:
            module.add_axis(fig, gridspec)
        else:
            module.add_axis(fig, gridspec)  # InfoTextModule overrides -> no-op

    context = dict(
        parameters_df=parameters_df,
        bats=bats,
        obstacles=obstacles,
        jammers=jammers,
        history=history,
        focal_bat=focal_bat,
        show_sounds=show_sounds,
        resolution=resolution,
    )

    for module in module_instances:
        module.setup(**context)

    def init():
        artists = []
        for module in module_instances:
            artists += module.init_frame()
        return artists

    def animate(i):
        frame = history[i]
        artists = []
        for module in module_instances:
            artists += module.update_frame(i, frame, context)
        return artists

    return fig, module_instances, init, animate


def _add_legend(module_instances):
    arena_module = next(
        (m for m in module_instances if isinstance(m, ArenaModule)), None
    )
    if arena_module is not None:
        handles, labels = arena_module.get_legend_handles()
        print(labels)
        arena_module.ax.legend(
            loc="center right", bbox_to_anchor=(-0.1, 0.5), handles=handles
        )


def _default_render_workers():
    """Parallel rendering only from a main process (e.g. not inside hydra's
    joblib launcher workers, which already run several jobs in parallel)."""
    process = multiprocessing.current_process()
    if process.name != "MainProcess" or process.daemon:
        return 1
    return max(1, min(_MAX_AUTO_RENDER_WORKERS, (os.cpu_count() or 1) // 2))


# ----------------------------------------------------------------------------
# parallel frame rendering
# ----------------------------------------------------------------------------
_worker = {}


def _render_worker_init(payload):
    mpl.use("Agg")
    sys.stdout = open(os.devnull, "w")  # module setup prints; keep the log clean
    state = pickle.loads(payload)
    _worker.clear()
    _worker.update(state)
    _worker["figure"] = None


def _worker_figure_at(frame_index):
    """Worker figure with module state as it is right before `frame_index`
    is drawn (frames before it are replayed without rendering)."""
    figure_state = _worker["figure"]
    if figure_state is None or figure_state["next_frame"] > frame_index:
        if figure_state is not None:
            plt.close(figure_state["fig"])
        fig, module_instances, init, animate = _build_figure(
            pickle.loads(_worker["modules"]), **_worker["build_kwargs"]
        )
        _add_legend(module_instances)
        init()  # Animation.save -> _init_draw()
        figure_state = {"fig": fig, "animate": animate, "next_frame": 0}
        _worker["figure"] = figure_state
    while figure_state["next_frame"] < frame_index:
        figure_state["animate"](figure_state["next_frame"])
        figure_state["next_frame"] += 1
    return figure_state


def _render_frames(start, stop):
    figure_state = _worker_figure_at(start)
    fig = figure_state["fig"]
    width, height = _worker["frame_size_inches"]
    frames = []
    with (
        mpl.rc_context({"savefig.bbox": None}),
        cbook._setattr_cm(fig.canvas, _is_saving=True, manager=None),
    ):
        for i in range(start, stop):
            figure_state["animate"](i)
            figure_state["next_frame"] = i + 1
            # identical to FFMpegWriter.grab_frame
            fig.set_size_inches(width, height)
            buffer = io.BytesIO()
            fig.savefig(
                buffer,
                format="rgba",
                dpi=_worker["dpi"],
                **_worker["savefig_kwargs"],
            )
            frames.append(buffer.getvalue())
    return frames


class _PrerenderedFFMpegWriter(animation.FFMpegWriter):
    """FFMpegWriter that also accepts frames rendered elsewhere."""

    def write_raw_frame(self, rgba_bytes):
        self._proc.stdin.write(rgba_bytes)


def _save_animation_in_parallel(fig, filename, fps, n_frames, n_workers, payload):
    # same dpi / facecolor resolution as matplotlib's Animation.save
    dpi = mpl.rcParams["savefig.dpi"]
    if dpi == "figure":
        dpi = fig.dpi
    writer = _PrerenderedFFMpegWriter(fps=fps)

    with writer.saving(fig, filename, dpi):
        savefig_kwargs = {}
        if not writer._supports_transparency():
            facecolor = mpl.rcParams["savefig.facecolor"]
            if facecolor == "auto":
                facecolor = fig.get_facecolor()
            r, g, b, a = mcolors.to_rgba(facecolor)
            savefig_kwargs["facecolor"] = a * np.array([r, g, b]) + 1 - a
            savefig_kwargs["transparent"] = False
        payload = dict(
            payload,
            dpi=dpi,
            frame_size_inches=(writer._w, writer._h),
            savefig_kwargs=savefig_kwargs,
        )

        chunks = [
            (s, min(s + _FRAMES_PER_TASK, n_frames))
            for s in range(0, n_frames, _FRAMES_PER_TASK)
        ]
        progress = tqdm(total=n_frames, file=sys.stdout, desc="Animating")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_render_worker_init,
            initargs=(pickle.dumps(payload),),
        ) as executor:
            pending = deque()
            next_chunk = 0
            max_in_flight = 2 * n_workers
            while next_chunk < len(chunks) or pending:
                while next_chunk < len(chunks) and len(pending) < max_in_flight:
                    pending.append(executor.submit(_render_frames, *chunks[next_chunk]))
                    next_chunk += 1
                for frame in pending.popleft().result():
                    writer.write_raw_frame(frame)
                    progress.update(1)
        progress.close()


def visualize(
    output_dir,
    save_animation,
    unique_id,
    resolution,
    show_sounds,
    focal_bat,
    modules=None,
    rows=2,
    columns=3,
    render_workers=None,
):
    """Builds (and optionally saves) an animation made of the requested modules.

    Args:
        modules (list, optional): list of module classes / instances / string
            aliases (see AVAILABLE_MODULES) to include. Defaults to every
            panel the original script hard-coded.
        render_workers (int, optional): processes used to render the saved
            video. None picks automatically (1 inside hydra/joblib workers).
    """

    if modules is None:
        modules = DEFAULT_MODULES

    if render_workers is None:
        render_workers = _default_render_workers()
    try:  # snapshot the (not yet set up) modules for the render workers
        pickled_modules = pickle.dumps(list(modules)) if render_workers > 1 else None
    except Exception:  # e.g. an unpicklable custom module instance
        pickled_modules = None

    history, parameters_df, bats, obstacles, jammers = stitch_together_history_lists(
        output_dir
    )
    history = history[::resolution]

    build_kwargs = dict(
        history=history,
        parameters_df=parameters_df,
        bats=bats,
        obstacles=obstacles,
        jammers=jammers,
        focal_bat=focal_bat,
        show_sounds=show_sounds,
        resolution=resolution,
        rows=rows,
        columns=columns,
    )
    fig, module_instances, init, animate = _build_figure(modules, **build_kwargs)

    render_in_parallel = (
        save_animation
        and pickled_modules is not None
        and len(history) > _FRAMES_PER_TASK
    )
    if render_in_parallel:
        frames = range(len(history))  # progress is reported by the parallel saver
    else:
        frames = tqdm(range(len(history)), file=sys.stdout, desc="Animating")

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=frames,
        init_func=init,
        blit=False,
        interval=parameters_df["FRAME_RATE"] * 0.00001,
    )

    print(len(history))
    _add_legend(module_instances)

    #     arena_module = next(
    #         (m for m in module_instances if isinstance(m, ArenaModule)), None
    #     )
    # if arena_module is not None:
    #     handles, labels = arena_module.get_legend_handles()
    #     print(labels)
    #     plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), handles=handles)

    if save_animation:
        filename = save_animation + f"/animation_without_sound_id_{unique_id}.mp4"
        if render_in_parallel:
            payload = dict(modules=pickled_modules, build_kwargs=build_kwargs)
            _save_animation_in_parallel(
                fig,
                filename,
                parameters_df["FRAME_RATE"],
                len(history),
                render_workers,
                payload,
            )
            # `ani` is only kept for plt.show(); the frames were saved above, so
            # silence matplotlib's "Animation was deleted without rendering".
            ani._draw_was_started = True
        else:
            ffwriter = animation.FFMpegWriter(fps=parameters_df["FRAME_RATE"])
            ani.save(filename, writer=ffwriter)
    plt.show()
