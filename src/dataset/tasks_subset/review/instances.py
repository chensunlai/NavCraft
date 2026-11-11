"""Visibility of exactly the privately bound instance scope, never inferred categories."""
from hashlib import sha256


def frame_visibility(bundle, key, identifier, scope=None, root=None):
    frame = bundle.get("frames", {}).get(key, {})
    evidence = bundle.get("instance_evidence", {}).get(key, {})
    record = evidence.get("objects", {}).get(identifier, {})
    binding = record.get("completeness_binding") or {}
    if (evidence.get("valid") is not True or record.get("mapping_valid") is not True
            or record.get("association_verified") is not True or record.get("mapping_complete") is not True
            or not binding.get("scope") or scope is not None and scope != binding["scope"]
            or evidence.get("pose") != frame.get("pose") or evidence.get("camera") != frame.get("camera")):
        return "unknown"
    if root is not None:
        from pathlib import Path
        root = Path(root)
        try:
            if (sha256((root/frame["rgb"]).read_bytes()).hexdigest() != evidence["source_rgb_sha256"] or
                    sha256((root/evidence["mask"]).read_bytes()).hexdigest() != evidence["mask_sha256"]):
                return "unknown"
        except (OSError, KeyError):
            return "unknown"
    pixels = record.get("pixels")
    threshold = bundle.get("instance_capability", {}).get("min_pixels")
    if type(threshold) is not int or threshold <= 0:
        return "unknown"
    if type(pixels) is not int:
        return "unknown"
    return "visible" if pixels >= threshold else "absent" if pixels == 0 else "unknown"


def node_visibility(bundle, keys, identifier, scope=None, root=None):
    if len(keys) != 6 or len(set(keys)) != 6:
        return "unknown"
    values = [frame_visibility(bundle,k,identifier,scope,root) for k in keys]
    if "visible" in values:
        return "visible"
    return "absent" if set(values)=={"absent"} else "unknown"
