"""Base class every pluggable animation panel inherits from.

A module knows how to:
  1. create its own axis (`add_axis`)
  2. pre-compute whatever data it needs + build its artists (`setup`)
  3. reset those artists for FuncAnimation's init_func (`init_frame`)
  4. update those artists for a given frame index (`update_frame`)

The `Visualizer` never has to know anything about what a given panel draws -
it just loops over whichever modules were selected for the run.
"""


class AnimationModule:
    #: default (row, col) inside the 2x3 GridSpec used originally.
    #: Set to None for modules that don't need a dedicated axis (e.g. text).
    GRID_POS = (0, 0)
    #: projection to hand to `fig.add_subplot`, e.g. "polar" or None
    PROJECTION = None

    def __init__(self, **kwargs):
        # `grid_pos` / `projection` kwargs let you reposition a panel without
        # touching its class definition - e.g. ArenaModule(grid_pos=(0, 0))
        self.kwargs = kwargs
        self.fig = None
        self.ax = None

    def add_axis(self, fig, gridspec):
        row, col = self.kwargs.get("grid_pos", self.GRID_POS)
        projection = self.kwargs.get("projection", self.PROJECTION)
        self.fig = fig
        self.ax = fig.add_subplot(gridspec[row, col], projection=projection)
        return self.ax

    def setup(self, **context):
        raise NotImplementedError

    def init_frame(self):
        raise NotImplementedError

    def update_frame(self, i, frame, context):
        raise NotImplementedError
