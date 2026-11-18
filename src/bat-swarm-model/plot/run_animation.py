# from plotting.arena_module import ArenaModule
# from plotting.info_text_module import InfoTextModule
# from plotting.polarization_milling_module import PolarizationMillingModule
# from plotting.sensory_volume_module import SensoryVolumeModule
from plotting.visualizer import visualize

if __name__ == "__main__":
    # Same as the original script (all 5 panels):
    # visualize(output_dir=..., save_animation=..., unique_id=..., resolution=1,
    #           show_sounds=True, focal_bat=0)

    # Only want the arena + sensory volume + text counters this time:
    OUTPUT_DIR = (
        r"/home/adityamoger/Documents/GitHub/claude-bs/outputs/2026-11-18/19-49-26/"
    )
    visualize(
        output_dir=OUTPUT_DIR,
        save_animation=OUTPUT_DIR,
        unique_id="plots",
        resolution=30,
        show_sounds=False,
        focal_bat=0,
        modules=[
            "arena",
            # "polarization_milling",
            # "info_text",
            # "ipi_radial",
            # "sum_radial",
            # "intensity",
            # "sensory_volume",
        ],
        rows=1,
        columns=1,
        # or equivalently: modules=["arena", "sensory_volume", "info_text"]
    )
