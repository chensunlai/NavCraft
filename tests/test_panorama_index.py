import pytest
from PIL import Image

from src.harness.observations import Observations


@pytest.fixture
def observations(tmp_path):
    store = Observations(tmp_path, {"origin": [0, 0], "size_m": 10})
    Image.new("RGB", (640, 480), (80, 100, 120)).save(tmp_path / "source.png")
    return store


def add_panorama(observations, group="p_one", pos=None, step=0, heading=0, **kwargs):
    return [observations.add(observations.workspace / "source.png", group=group, yaw=yaw,
                             pose={"pos": pos or [.2, .2], "heading": (heading + yaw) % 360},
                             step=step, **kwargs)
            for yaw in range(0, 360, 60)]


def test_panorama_reference_reuses_existing_grid_and_survives_reopen(observations):
    frames = add_panorama(observations)
    grid = observations.grid([row["frame"] for row in frames])
    group = observations.panorama("p_one")
    assert group["pose"] == frames[0]["pose"]
    assert [row["frame"] for row in group["frames"]] == [row["frame"] for row in frames]
    image = observations.image("p_one")
    assert image["image"] == "p_one"
    assert image["path"] == grid["path"]
    assert len(observations.views) == 1
    assert observations.resolve_pixel("p_one", [650, 556]) == (frames[4]["frame"], [10, 24])
    restored = Observations(observations.workspace, observations.transform)
    assert restored.image("p_one")["path"] == grid["path"]
    assert len(restored.views) == 1


def test_panorama_image_is_created_once_and_crops_resolve_to_original_frames(observations):
    frames = add_panorama(observations)
    first = observations.image("p_one")
    assert observations.image("p_one")["path"] == first["path"]
    assert len(observations.views) == 1
    crop = observations.crop("p_one", [0, 0, .5, .5])
    assert observations.resolve_pixel(crop["image"], [200, 200]) == (frames[0]["frame"], [200, 174])
    with pytest.raises(ValueError, match="label_or_padding"):
        observations.resolve_pixel("p_one", [650, 510])
    with pytest.raises(ValueError, match="pixel_out_of_bounds"):
        observations.resolve_pixel("p_one", [1920, 100])


def test_history_groups_panorama_and_filters_non_scene_images(observations):
    source = observations.workspace / "source.png"
    observations.add(source, kind="target")
    single = observations.add(source, step=1, pose={"pos": [.1, .1], "heading": 0})
    frames = add_panorama(observations, step=2)
    observations.add(source, kind="question")
    history = observations.history(limit=1)
    assert history["total"] == 2
    assert history["next_offset"] == 1
    assert history["events"][0]["image"] == single["frame"]
    second = observations.history(limit=1, offset=1)
    assert second["events"][0]["image"] == "p_one"
    assert second["events"][0]["frames"] == [row["frame"] for row in frames]
    assert second["next_offset"] is None


def test_current_panorama_requires_same_step_position_and_heading(observations):
    frames = add_panorama(observations, step=2)
    current = frames[0]
    assert observations.current_panorama(current) == "p_one"
    assert observations.current_panorama({**current, "step": 3}) is None
    assert observations.current_panorama({**current, "pose": {"pos": [.21, .2], "heading": 0}}) is None
    assert observations.current_panorama({**current, "pose": {"pos": [.2, .2], "heading": 30}}) is None
    add_panorama(observations, "p_two", step=2)
    assert observations.current_panorama(current) == "p_two"


@pytest.mark.parametrize("invalid", ["incomplete", "target", "position", "step", "yaw", "pose", "first_pose"])
def test_incomplete_or_inconsistent_panorama_cannot_be_bound(observations, invalid):
    frames = add_panorama(observations)
    if invalid == "incomplete":
        del observations.items[frames[-1]["frame"]]
    elif invalid == "target":
        frames[-1]["kind"] = "target"
    elif invalid == "position":
        frames[-1]["pose"]["pos"] = [.3, .2]
    elif invalid == "step":
        frames[-1]["step"] = 1
    elif invalid == "yaw":
        frames[-1]["yaw"] = None
    elif invalid == "pose":
        frames[-1]["pose"] = None
    elif invalid == "first_pose":
        frames[0]["pose"] = None
    with pytest.raises(ValueError, match="Panorama"):
        observations.panorama("p_one")
    assert observations.current_panorama(frames[0]) is None


@pytest.mark.parametrize("current", [
    None, [], {}, {"step": 0, "pose": None},
    {"step": 0, "pose": {"pos": [], "heading": 0}},
    {"step": False, "pose": {"pos": [.2, .2], "heading": 0}},
    {"step": 0, "pose": {"pos": [.2, .2], "heading": False}},
    {"step": 0, "pose": {"pos": [.2, .2], "heading": 0}, "kind": "target"},
])
def test_current_panorama_rejects_invalid_current_observation(observations, current):
    add_panorama(observations)
    assert observations.current_panorama(current) is None


def test_history_keeps_individual_frames_from_interrupted_panorama(observations):
    row = observations.add(observations.workspace / "source.png", group="p_interrupted", yaw=0,
                           step=0, pose={"pos": [.2, .2], "heading": 0})
    assert observations.history()["events"][0]["image"] == row["frame"]


def test_panorama_metadata_is_not_mutable_through_reads(observations):
    add_panorama(observations)
    panorama = observations.panorama("p_one")
    panorama["pose"]["pos"][0] = .9
    panorama["frames"][0]["kind"] = "target"
    assert observations.panorama("p_one")["pose"]["pos"] == [.2, .2]
