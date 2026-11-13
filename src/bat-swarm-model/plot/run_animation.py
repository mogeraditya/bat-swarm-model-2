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
    OUTPUT_DIR = r"./outputs/2026-11-13/22-15-24/"
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
            "info_text",
            "ipi_radial",
            "sum_radial",
            "intensity",
            # "sensory_volume",
        ],
        rows=2,
        columns=3,
        # or equivalently: modules=["arena", "sensory_volume", "info_text"]
    )
