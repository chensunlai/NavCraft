"""Visual-map navigation contracts through real session/tools/runtime code."""

from dataclasses import replace
import hashlib
import json

from PIL import Image
import pytest

from src.dataset.episodes.base import EpisodeSpec
from src.environment.fake import FakeEnvironment, FakeSimulator
from src.environment.tasks.base import GOATTask, ObjectNavTask
from src.legacy.harness.agent.base import AgentLimits, ToolCall
from src.legacy.harness.agent.context import AgentContext
from src.legacy.harness.agent.loop import AgentRuntime
from src.legacy.harness.agent.model import ScriptedModel
from src.legacy.harness.assets import SceneAssets
from src.legacy.harness.session import NavigationSession
from src.legacy.harness.trace import TraceReader, TraceWriter

TOOLS = [
    "nav_observe",
    "nav_step",
    "nav_finish",
    "read_text",
    "view_images",
    "mark_pose",
]
PRIVATE_KEYS = {
    "pose",
    "camera",
    "depth",
    "positions",
    "rotation_xyzw",
    "camera_position",
    "world_from_optical",
    "reference_path",
    "gt_path",
    "image_goal",
    "distance",
    "private_coordinates",
    "success",
    "target_position",
}


def assert_public(value):
    if isinstance(value, dict):
        assert not PRIVATE_KEYS.intersection(value)
        for child in value.values():
            assert_public(child)
    elif isinstance(value, list):
        for child in value:
            assert_public(child)
    elif isinstance(value, str):
        assert "731.123" not in value
        assert "raw/scene.json" not in value


@pytest.fixture
def visual_factory(tmp_path):
    root = tmp_path / "scene_maps"
    scene = root / "hm3d" / "val" / "00001-fixture"
    (scene / "raw").mkdir(parents=True)
    (scene / "evidence").mkdir()
    Image.new("RGB", (200, 400), "white").save(scene / "floor_plan.png")
    Image.new("RGB", (30, 20), "blue").save(scene / "evidence_rooms.jpg")
    Image.new("RGB", (30, 20), "green").save(scene / "evidence" / "chair.jpg")
    (scene / "evidence" / "rgbd.npz").write_bytes(b"private data")
    (scene / "scene_description.md").write_text(
        "Chair room. [Evidence](evidence/chair.jpg).\nDoor across the chair.\n"
    )
    (scene / "raw" / "scene.json").write_text(
        json.dumps(
            {
                "levels": [{"id": "L1", "height_m": 0, "bounds": [0, 10, 0, 20]}],
            }
        )
    )
    (scene / "raw" / "regions.json").write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "id": "R1",
                        "level": "L1",
                        "polygon": [[0, 0], [10, 0], [10, 20], [0, 20]],
                    }
                ],
            }
        )
    )
    (scene / "raw" / "floor_plan.svg").write_text(
        """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 200"><g id="axes_1"><g id="region-R1"><path d="M 10 180 L 90 180 L 90 20 L 10 20 z" clip-path="url(#clip)" style="fill: blue"/></g></g><defs><clipPath id="clip"><rect x="10" y="20" width="80" height="160"/></clipPath></defs></svg>"""
    )
    opened = []

    def factory(*, goat=False, simulator=FakeSimulator):
        run = tmp_path / f"run_{len(opened)}"
        goals = [
            {
                "modality": "object",
                "instruction": "Find chair",
                "positions": [[5, 0, -10.25]],
            }
        ]
        if goat:
            goals.append(
                {
                    "modality": "image",
                    "instruction": "Find image target",
                    "positions": [[5, 0, -10]],
                    "image_goal": {
                        "position": [731.123, 882, 993],
                        "rotation": [0, 0, 0, 1],
                        "image_dimensions": [12, 16],
                    },
                }
            )
        episode = EpisodeSpec(
            id="fixture:visual",
            source_id="visual",
            benchmark="goat" if goat else "objnav",
            scene_id="hm3d/val/00001-fixture/fixture.basis.glb",
            start_position=[5, 0, -10],
            start_rotation=[0, 0, 0, 1],
            goals=goals,
            source_file="fixture",
            reference_path=[[731.123, 992, 993]],
        )
        environment = FakeEnvironment(
            {"width": 16, "height": 12, "depth": True},
            GOATTask({}) if goat else ObjectNavTask({}),
            run,
            simulator_factory=simulator,
        )
        memory = SceneAssets(episode.scene_id, root, run)
        trace = TraceWriter(run, episode.id)
        session = NavigationSession(environment, episode, trace, TOOLS, assets=memory)
        initial = session.start()
        opened.append((session, trace))
        return session, initial, scene

    yield factory
    for session, trace in opened:
        session.close()
        trace.close()


def public_initial(initial):
    return json.loads(initial[1]["content"][0]["text"])


def function(call_id, name, arguments):
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
    }


def test_initial_hint_and_all_runtime_outputs_hide_true_spatial_state(visual_factory):
    session, initial, _ = visual_factory()
    data = public_initial(initial)
    assert_public(data)
    map_ref = data["memory"]["initial_map"]["map_ref"]
    assert session.assets.path_for_image(map_ref).exists()
    with Image.open(session.assets.path_for_image(map_ref)) as image:
        assert image.getpixel((100, 200)) == (224, 32, 32)
        assert image.getpixel((100, 182)) == (224, 32, 32)
    assert {spec.name for spec in session.toolset.specs()} == set(TOOLS)
    calls = [
        ToolCall("read", "read_text", {"ref": "scene_description.md"}),
        ToolCall("move", "nav_step", {"actions": ["move_forward", "turn_right"]}),
        ToolCall("pano", "nav_observe", {"mode": "panorama"}),
        ToolCall("observe", "nav_observe", {}),
    ]
    context = AgentContext(initial)
    for call in calls:
        result = session.execute(call)
        assert result.ok
        assert_public(result.data)
        context.append_output([function(call.call_id, call.name, call.arguments)])
        context.append_tool_result(result)
    snapshot = context.snapshot()
    assert "image_role" not in json.dumps(snapshot)
    for item in snapshot:
        if item.get("type") == "function_call_output":
            assert_public(json.loads(item["output"]))


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("read_text", {"ref": "raw/scene.json"}),
        ("view_images", {"refs": ["raw/floor_plan.svg"]}),
        ("view_images", {"refs": ["evidence/rgbd.npz"]}),
        ("view_images", {"refs": ["../private/state.json"]}),
        ("nav_depth", {"frame_id": "frame_000000", "points": [[0, 0]]}),
    ],
)
def test_actor_tools_reject_privileged_files_and_depth(visual_factory, name, arguments):
    session, _, _ = visual_factory()
    result = session.execute(ToolCall("denied", name, arguments))
    assert not result.ok and result.images == []
    assert session.environment.task.steps == 0


def test_every_atomic_step_is_reloadable_and_replays_do_not_move(visual_factory):
    session, _, _ = visual_factory()
    call = ToolCall(
        "move", "nav_step", {"actions": ["move_forward", "turn_left", "move_forward"]}
    )
    result = session.execute(call)
    refs = [row["frame_id"] for row in result.data["executed_actions"]]
    assert len(refs) == len(set(refs)) == 3
    assert result.images == [session.assets.path_for_image(refs[-1])]
    read = session.execute(ToolCall("history", "view_images", {"refs": refs}))
    assert read.ok and len(read.images) == 3
    assert all(path.is_file() for path in read.images)
    point = session.environment.sim.position()
    assert (
        session.execute(ToolCall("move", "nav_step", json.dumps(call.arguments)))
        is result
    )
    assert session.environment.sim.position() == point
    assert session.environment.task.steps == 3
    different = session.execute(
        ToolCall("move", "nav_step", {"actions": ["turn_right"]})
    )
    assert not different.ok and session.environment.task.steps == 3


def test_collision_interrupts_remaining_actions(visual_factory):
    class CollidingSimulator(FakeSimulator):
        def step(self, action):
            return True

    session, _, _ = visual_factory(simulator=CollidingSimulator)
    result = session.execute(
        ToolCall(
            "move",
            "nav_step",
            {"actions": ["move_forward", "turn_left", "move_forward"]},
        )
    )
    assert len(result.data["executed_actions"]) == 1
    assert result.data["executed_actions"][0]["collision"]
    assert session.environment.task.steps == 1
    assert session.environment.sim.position() == [5, 0, -10]


def test_panorama_preserves_state_cached_image_and_unique_capture_refs(visual_factory):
    session, _, _ = visual_factory()
    session.execute(ToolCall("look", "nav_step", {"actions": ["look_up", "turn_left"]}))
    simulator = session.environment.sim
    original = (
        simulator.position(),
        simulator.yaw,
        simulator.pitch,
        session.environment.task.steps,
    )
    current = session.transition.observation
    first = session.execute(ToolCall("pano", "nav_observe", {"mode": "panorama"}))
    assert original == (
        simulator.position(),
        simulator.yaw,
        simulator.pitch,
        session.environment.task.steps,
    )
    assert session.transition.observation is current
    assert first.data["observation"]["frame_id"] == current.frame_id
    assert [view["right_degrees"] for view in first.data["views"]] == list(
        range(0, 360, 60)
    )
    refs = [view["image_ref"] for view in first.data["views"]]
    loaded = session.execute(ToolCall("views", "view_images", {"refs": refs}))
    assert loaded.ok and len(loaded.images) == 6
    for path in loaded.images:
        with Image.open(path) as image:
            assert image.getpixel((0, 0))[2] != int(
                current.frame_id.removeprefix("frame_")
            )
    assert (
        session.execute(ToolCall("pano", "nav_observe", {"mode": "panorama"})) is first
    )
    again = session.execute(ToolCall("pano2", "nav_observe", {"mode": "panorama"}))
    assert not set(refs).intersection(view["image_ref"] for view in again.data["views"])
    single = session.execute(ToolCall("single", "nav_observe", {"mode": "single"}))
    assert single.images == [current.rgb]


def test_model_marker_does_not_snap_to_truth_or_mutate_archive(visual_factory):
    session, _, scene = visual_factory()
    original_hash = hashlib.sha256((scene / "floor_plan.png").read_bytes()).hexdigest()
    point = session.environment.sim.position()
    result = session.execute(
        ToolCall(
            "estimate",
            "mark_pose",
            {
                "map_ref": "floor_plan.png",
                "position": [0.2, 0.8],
                "heading": [0.4, 0.8],
                "note": "Hypothesis based on chair identity",
            },
        )
    )
    assert result.ok and result.data["marker_kind"] == "Model estimate"
    assert result.image_roles[str(result.images[0])] == "map"
    with Image.open(result.images[0]) as image:
        assert image.getpixel((40, 320)) == (224, 32, 32)
        assert image.getpixel((100, 200)) == (255, 255, 255)
    assert session.environment.sim.position() == point
    assert (
        original_hash
        == hashlib.sha256((scene / "floor_plan.png").read_bytes()).hexdigest()
    )


def test_goal_switch_has_current_goal_image_without_repeat_change_on_panorama(
    visual_factory,
):
    session, initial, _ = visual_factory(goat=True)
    context = AgentContext(initial)
    changed = session.execute(ToolCall("finish", "nav_finish", {}))
    assert changed.goal_changed and not changed.terminated
    assert_public(changed.data)
    assert changed.data["task"]["goal_id"] == 1
    goal_ref = changed.data["task"]["image_ref"]
    goal_path = session.assets.path_for_image(goal_ref)
    assert changed.image_roles[str(goal_path)] == "goal"
    context.append_output([function("finish", "nav_finish", {})])
    context.append_tool_result(changed)
    assert context._pinned["goal"]["image_path"] == str(goal_path)
    panorama = session.execute(ToolCall("pano", "nav_observe", {"mode": "panorama"}))
    assert not panorama.goal_changed
    assert_public(panorama.data)
    context.append_output([function("pano", "nav_observe", {"mode": "panorama"})])
    context.append_tool_result(panorama)
    assert context._pinned["goal"]["image_path"] == str(goal_path)
    single = session.execute(ToolCall("observe", "nav_observe", {}))
    assert not single.goal_changed
    context.append_output([function("observe", "nav_observe", {})])
    context.append_tool_result(single)
    assert context._pinned["goal"]["image_path"] == str(goal_path)


def test_runtime_skips_old_target_actions_and_trace_requests_are_public(visual_factory):
    session, initial, _ = visual_factory(goat=True)
    model = ScriptedModel(
        {
            "turns": [
                {
                    "output": [
                        function("finish-first", "nav_finish", {}),
                        function(
                            "stale-move", "nav_step", {"actions": ["move_forward"]}
                        ),
                    ]
                },
                {"output": [function("pano", "nav_observe", {"mode": "panorama"})]},
                {"output": [function("finish-second", "nav_finish", {})]},
            ]
        }
    )
    result = AgentRuntime(model, {}).run(
        initial, session.toolset, session.trace, AgentLimits()
    )
    assert result.reason == "terminated"
    assert session.environment.sim.position() == [5, 0, -10]
    assert session.environment.task.steps == 2
    events = list(TraceReader(session.trace.path).events())
    skipped = [event for event in events if event["event"] == "tool_skipped"]
    assert len(skipped) == 1 and skipped[0]["reason"] == "goal_changed"
    for event in events:
        if event["event"] == "model_request":
            for item in event["items"]:
                assert "image_role" not in json.dumps(item)
                if item.get("type") == "function_call_output":
                    assert_public(json.loads(item["output"]))


def test_step_feedback_whitelist_applies_to_each_intermediate_action(
    visual_factory, monkeypatch
):
    session, _, _ = visual_factory()
    step = session.environment.step

    def annotated_step(action):
        transition = step(action)
        return replace(
            transition,
            feedback={
                **transition.feedback,
                "private_coordinates": [731.123, 882, 993],
                "distance": 0.25,
            },
        )

    monkeypatch.setattr(session.environment, "step", annotated_step)
    result = session.execute(
        ToolCall("move", "nav_step", {"actions": ["move_forward", "turn_left"]})
    )
    assert_public(result.data)
