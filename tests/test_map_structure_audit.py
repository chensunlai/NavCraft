from src.data.map_structure_audit import near_endpoint_clusters


def test_near_endpoint_clusters_are_reported_without_merging_geometry():
    primitives = [
        {"id": "a", "type": "line", "start": [0, 0], "end": [1, 0]},
        {"id": "b", "type": "line", "start": [.02, .01], "end": [0, 1]},
    ]
    clusters = near_endpoint_clusters(primitives)
    assert len(clusters) == 1
    assert len(clusters[0]["endpoints"]) == 2
    assert primitives[0]["start"] == [0, 0]
