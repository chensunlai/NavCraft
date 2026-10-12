"""Explicit bake settings survive the native boundary without altering targets."""

from types import SimpleNamespace

import pytest
import yaml


def test_objectnav_configurations_explicitly_align_hm3d_v2_navmesh():
    from pathlib import Path

    for name in ("objectnav_visual", "objectnav_visual_fine"):
        config = yaml.safe_load(Path(f"config/environment/{name}.yaml").read_text())[
            "environment"
        ]
        assert config["navmesh_cell_height"] == 0.05
        assert config["agent_radius"] == 0.18
        assert config["agent_height"] == 0.88
    baseline = yaml.safe_load(Path("config/environment/objectnav.yaml").read_text())[
        "environment"
    ]
    assert "navmesh_cell_height" not in baseline


def test_native_constructor_rebakes_on_voxel_mismatch_without_gpu(
    tmp_path, monkeypatch
):
    habitat_sim = pytest.importorskip("habitat_sim")
    from src.environment.simulator import HabitatSimulator

    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    settings.agent_height, settings.agent_radius = 0.88, 0.18

    class Simulator:
        def __init__(self, config):
            self.pathfinder = SimpleNamespace(
                is_loaded=True, nav_mesh_settings=settings
            )
            self.received_settings = None
            self.agent = SimpleNamespace(set_state=lambda state: None)

        def seed(self, seed):
            pass

        def recompute_navmesh(self, pathfinder, value):
            self.received_settings = value
            return True

        def get_agent(self, index):
            return self.agent

        def close(self):
            pass

    monkeypatch.setattr(habitat_sim, "Simulator", Simulator)
    (tmp_path / "scene.glb").touch()
    episode = SimpleNamespace(
        scene_id="scene.glb", start_position=[0, 0, 0], start_rotation=[0, 0, 0, 1]
    )
    simulator = HabitatSimulator(
        episode,
        {
            "scenes": str(tmp_path),
            "depth": False,
            "agent_height": 0.88,
            "agent_radius": 0.18,
            "navmesh_cell_height": 0.05,
        },
    )
    assert simulator.sim.received_settings.cell_height == pytest.approx(0.05)
    assert simulator.sim.received_settings.agent_height == pytest.approx(0.88)
    assert simulator.sim.received_settings.agent_radius == pytest.approx(0.18)
    assert settings.cell_height == pytest.approx(0.2)


@pytest.mark.parametrize("height", [0, -0.1, float("nan"), float("inf")])
def test_invalid_voxel_height_is_rejected(height):
    pytest.importorskip("habitat_sim")
    from src.environment.simulator import SensorConfig

    with pytest.raises(ValueError):
        SensorConfig.from_dict({"navmesh_cell_height": height})
