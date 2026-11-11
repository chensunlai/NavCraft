"""Stable IDs for the first task subset."""

JUDGMENT_KINDS = {
    key: {"title": title, "requirements": requirements}
    for key, title, requirements in [
        ("object_relative_position", "Relative object position", "Two marked floor patches with public camera calibration"),
        ("change_reference_frame", "Change reference frame", "Calibrated floor points and an explicit heading"),
        ("distance_estimation", "Estimate metric distance", "Visible floor patches, public calibration and private depth verification"),
        ("compare_distance_size", "Compare distances or sizes", "Calibrated floor points with a separated distance ranking"),
        ("egocentric_layout", "Egocentric layout", "Six-view panorama and observed landmarks"),
        ("visible_topdown_layout", "Visible top-down layout", "Panorama landmarks with public calibrated multiview correspondence"),
        ("cross_view_identity", "Cross-view identity", "Two overlapping views and verified identities"),
        ("relative_camera_pose", "Relative camera pose", "Pure rotation or two calibrated floor correspondences over nonzero translation"),
        ("panorama_view_relation", "Panorama view relationship", "Six simultaneous views and a sighting"),
        ("camera_motion", "Camera motion", "Continuous nonzero motion, two floor correspondences and public calibration"),
        ("traversable_space", "Traversable space", "Replayed forward trials with visible route"),
        ("opening_passability", "Opening passability", "Visible opening and replayed crossing attempt"),
        ("opening_type", "Opening type", "Verified opening class and RGB"),
        ("current_room", "Current room", "Verified room and identifiable RGB evidence"),
        ("room_exit_directions", "Room exit directions", "Visible openings in the current room"),
        ("room_exit_destination", "Room exit destination", "Visible opening and prefix crossing evidence"),
        ("room_adjacency", "Room adjacency", "Observed opening crossings"),
        ("room_route", "Room route", "Observed room graph"),
        ("rooms_with_untraversed_exits", "Rooms with untraversed exits", "Prefix sightings and crossings"),
        ("nearest_node", "Nearest node", "Complete prefix and public positions"),
        ("visited_place", "Visited place", "Public prefix trajectory and a distance criterion"),
        ("image_to_node_map", "Image to node map", "A shared point, public camera poses, calibration and map"),
        ("memory_topdown_layout", "Memory top-down layout", "Node panoramas and public multiview landmark geometry"),
        ("memory_object_relation", "Memory object relation", "Landmarks from different nodes with public multiview geometry"),
        ("direct_node_passability", "Direct node passability", "Replayed direct trial and visually supported segment"),
        ("nearest_landmark", "Nearest landmark", "Target and comparison landmarks with public multiview geometry"),
        ("motion_to_point", "Motion to a point", "Replayed atomic motion and visible waypoint"),
        ("action_collision", "Action collision", "Replayed branch and visible maneuver"),
        ("observed_action_outcome", "Observed action outcome", "Continuous before and after observations"),
        ("blocked_receipt", "Blocked receipt", "Executed blocked movement and public receipt"),
        ("target_identity", "Target identity", "Distinguishable reference and verified instance"),
        ("target_seen_nodes", "Target sightings by node", "Legal reference, complete node views and scoped native instance visibility"),
        ("destination_exit", "Destination exit", "Observed room graph and current opening"),
        ("node_opening_route", "Node and opening route", "Observed node/opening graph"),
        ("exploration_frontier", "Exploration frontier", "Public prefix sightings and crossing history"),
        ("target_arrival", "Target arrival", "Verified object approach region, public distance criterion and answerability review"),
        ("memory_contradiction", "Memory contradiction", "Current direct counterevidence for a specific claim"),
    ]
}

NAVIGATION_KINDS = {
    f"nav_{index:02d}": {"title": title, "requirements": requirements}
    for index, title, requirements in [
        (1, "Reach a visible point", "Visible point and replayed approach"),
        (2, "Cross a visible opening", "Verified opening and replayed crossing"),
        (3, "Recover from being blocked", "Genuinely blocked state and replayed return to a known safe waypoint"),
        (4, "Observe and remember", "Novel place, panorama and room evidence"),
        (5, "Approach a nearby target", "Visible target identity and replayed approach"),
        (7, "Return to a target seen once", "Past target sighting and current occlusion"),
        (9, "Reach a destination across rooms", "Observed multi-room route"),
        (10, "Explore for an unseen target", "Novel target and replayed discovery"),
        (12, "Repair memory en route", "Evidence-checkable corrupted record"),
        (14, "Navigate without relying on memory", "Visible target or supplied route"),
        (16, "Follow a short instruction", "Validated alignment or short complete official instruction"),
        (18, "Complete a short official task", "Original training task and criterion"),
    ]
}
