"""Bind offline RGB review to immutable collection evidence."""
from copy import deepcopy
from hashlib import sha256
import json
from collections import defaultdict, Counter
from math import dist
from pathlib import Path

from PIL import Image, ImageDraw

from ..bundle import write_json


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def review_candidates(bundle, root):
    root = Path(root)
    names = {("object", s["id"]): s for s in bundle["scene"].get("objects", [])}
    names.update({("opening", s["id"]): s for s in bundle["scene"].get("openings", [])})
    for s in bundle["sightings"]:
        annotation = names.get((s["kind"], s["id"]))
        if annotation is None or not s.get("pixel") or s.get("method") not in {
            "foreground_in_schematic_bounds", "depth_clear_at_annotated_portal",
        }:
            continue
        source = bundle["frames"][s["frame"]]
        digest = sha256((root / source["rgb"]).read_bytes()).hexdigest()
        evidence = {"frame": s["frame"], "sha256": digest, "pixel": s["pixel"],
                    "geometry_method": s["method"], "world": s["world"],
                    "annotation_hash": _digest(annotation),
                    "scene_hashes": bundle.get("source", {}).get("sha256", {})}
        name = annotation.get("category") or annotation.get("name") or annotation["kind"]
        if name == "passage":
            name = "opening"
        yield {"candidate_id": _digest([s["kind"], s["id"], evidence])[:24],
               "kind": s["kind"], "instance": s["id"], "name": name, "source": evidence}

    from itertools import chain
    from ..judgment.construction import judgment_review_proposals
    from ..judgment.targets import target_reviews
    from .semantic_proposals import opening_type_proposals, surface_region_proposals
    from ..judgment.size import span_review_proposals
    from ..judgment.corrections import current_correction_reviews
    from ..navigation.construction import navigation_review_proposals

    proposals = chain(judgment_review_proposals(bundle, root), navigation_review_proposals(bundle, root), target_reviews(bundle), opening_type_proposals(bundle), surface_region_proposals(bundle), span_review_proposals(bundle,root), current_correction_reviews(bundle,root))
    for raw in proposals:
        evidence = {"frames": [{"frame": key, "sha256": sha256(
            (root / bundle["frames"][key]["rgb"]).read_bytes()).hexdigest()} for key in raw["frames"]],
                    "scene_hashes": bundle.get("source", {}).get("sha256", {}),
                    "fact_hash": _digest(raw)}
        yield {**raw, "candidate_id": _digest(evidence)[:24], "source": evidence}


def _checks(candidate):
    return candidate.get("check_keys", ["visible", "name_type_match"])


def apply_reviews(bundle, root, reviews):
    """Require two independently identified reviewers; uncertain/conflicting stays unknown."""
    bundle = deepcopy(bundle)
    bundle.pop("judgment_evidence", None)
    bundle.pop("navigation_evidence", None)
    bundle.pop("derived_sightings", None)
    bundle["targets"] = [t for t in bundle.get("targets",[]) if not t["id"].startswith("judgment-instance-")]
    for sight in bundle["sightings"]:
        if "geometry_verified" not in sight:
            sight["geometry_verified"] = (sight.get("verified") is True or
                                         sight.get("method")=="foreground_in_schematic_bounds")
        for key in ("semantic_verified","semantic_review","identity_verified","type_verified",
                    "review_candidate","review_source","review_name"):
            sight.pop(key,None)
        sight["verified"] = sight["geometry_verified"] and sight["kind"] != "object"
    reviews = list(reviews)
    by_id = {c["candidate_id"]: c for c in review_candidates(bundle, root)}
    while any(r.get("candidate_id") not in by_id for r in reviews):
        known = [r for r in reviews if r.get("candidate_id") in by_id]
        if not known:
            break
        enriched = apply_reviews(bundle, root, known)
        generated={c["candidate_id"]:c for c in review_candidates(enriched,root)}
        if not generated.keys()-by_id.keys():
            break
        by_id.update(generated)
    records = {}
    for review in reviews:
        candidate = by_id.get(review.get("candidate_id"))
        if (candidate is None or review.get("source") != candidate["source"] or
                review.get("kind") != candidate["kind"]):
            raise ValueError("Review does not bind to the current RGB and annotation")
        reviewer = review.get("reviewer")
        decision = review.get("decision")
        checks = review.get("checks")
        if (not isinstance(reviewer, str) or not reviewer.strip() or
                decision not in {"accept", "reject", "uncertain"} or not isinstance(checks, dict) or
                not isinstance(review.get("reason"), str) or not review["reason"].strip()):
            raise ValueError("Malformed review record")
        group = records.setdefault(candidate["candidate_id"], {})
        if reviewer in group:
            raise ValueError("Duplicate reviewer for one candidate")
        group[reviewer] = review
    decisions, facts = {}, []
    for identifier, reviews_by_person in records.items():
        values = [record for record in reviews_by_person.values()
                  if record.get("execution_status","completed")=="completed"]
        independent=[record.get("model") or record["reviewer"] for record in values]
        if len(set(independent))!=len(independent):
            raise ValueError("Independent reviews require distinct actual models")
        if not values:
            continue
        c = by_id[identifier]
        approved = len(values) >= 2 and all(v["decision"] == "accept" and
            all(v["checks"].get(k) is True for k in _checks(c)) for v in values)
        rejected = any(v["decision"] == "reject" for v in values)
        if "fact" in c:
            fact = c["fact"]
            if c.get("value_keys"):
                agreed={}
                for key,allowed in c["value_keys"].items():
                    answers=[v.get("values",{}).get(key) for v in values]
                    if len(set(answers))!=1 or answers[0] not in allowed:
                        approved=False
                        break
                    agreed[key]=answers[0]
                if approved:
                    fact={**fact,"value":{**fact["value"],**agreed}}
            if c.get("exclusive_checks"):
                outcomes = [tuple(k for k in _checks(c) if v["checks"].get(k) is True) for v in values]
                approved = (len(values) >= 2 and len(set(outcomes)) == 1 and len(outcomes[0]) == 1
                            and all(v["decision"] == "accept" for v in values))
                if approved:
                    fact = {**fact, "value": {**fact["value"], "review_result": outcomes[0][0]}}
            if approved:
                facts.append((fact, identifier))
            continue
        key = c["kind"], c["instance"], c["source"]["frame"], tuple(c["source"]["pixel"])
        decisions[key] = ("accept" if approved else "reject" if rejected else "uncertain", c)
    result = deepcopy(bundle)
    result["review_attempt_counts"]=dict(Counter(r.get("execution_status","completed") for r in reviews))
    for s in result["sightings"]:
        key = s["kind"], s["id"], s["frame"], tuple(s.get("pixel", []))
        decision, candidate = decisions.get(key, ("unreviewed", None))
        s["geometry_verified"] = (s.get("geometry_verified", s.get("verified")) is True or
                                  s["method"] == "foreground_in_schematic_bounds")
        s["semantic_verified"] = s["geometry_verified"] and decision == "accept"
        s["semantic_review"] = decision
        if candidate is not None:
            s.update(review_candidate=candidate["candidate_id"], review_source=candidate["source"],
                     review_name=candidate["name"])
        s["verified"] = s["geometry_verified"] and decision != "reject" and (
            s["kind"] != "object" or s["semantic_verified"])
        if s["kind"] == "opening":
            s["type_verified"] = s["semantic_verified"]
    objects = {o["id"]: o for o in bundle["scene"].get("objects", [])}
    sightings = defaultdict(list)
    for s in result["sightings"]:
        if s["kind"] != "object":
            continue
        s["identity_verified"] = False
        obj = objects.get(s["id"], {})
        center, size = obj.get("center_xyz"), obj.get("size_xyz")
        if not s["semantic_verified"] or not center or not size:
            continue
        others = [o for o in objects.values() if o["id"] != obj["id"] and o.get("level") == obj.get("level")]
        if any(o.get("center_xyz") and o.get("size_xyz") and all(
                abs(x-y) <= b/2 + .04 for x,y,b in zip(s["world"],o["center_xyz"],o["size_xyz"])
                ) for o in others):
            continue
        sightings[s["id"]].append(s)
    for group in sightings.values():
        for s in group:
            s["identity_verified"] = any(other["frame"] != s["frame"] and
                dist(s["world"], other["world"]) <= .25 for other in group)
    for fact, identifier in facts:
        rows = result.setdefault(fact["section"], {}).setdefault(fact["key"], [])
        row = {**fact["value"], "review_candidate": identifier, "review_source": by_id[identifier]["source"]}
        if row not in rows:
            rows.append(row)
    from ..sampling.semantics import link_native_identity,derive_instance_sightings
    result=link_native_identity(result,root)
    result["derived_sightings"]=derive_instance_sightings(result,root)
    from ..judgment.targets import register_instance_targets
    register_instance_targets(result,root)
    return result


def model_review(repo, config, output, root, bundle, candidate, reviewer):
    """One model reviews one marked RGB. No coordinates or oracle geometry enter its prompt."""
    from src.harness.runtime.request import PromptLimits
    from .model import run_fixed_evidence

    output, root = Path(output), Path(root)
    output.mkdir(parents=True, exist_ok=True)
    evidence = candidate["source"]
    images = []
    if "frames" in evidence:
        markers = candidate.get("markers")
        if markers is None:
            markers=[None]*len(evidence["frames"])
        for index, (item, marker) in enumerate(zip(evidence["frames"], markers, strict=True)):
            frame = bundle["frames"][item["frame"]]
            path = root / frame["rgb"]
            endpoints=candidate.get("endpoint_markers",[None]*len(evidence["frames"]))[index]
            polygon=candidate.get("polygons",[None]*len(evidence["frames"]))[index]
            if marker is not None or endpoints or polygon:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                if polygon:
                    ImageDraw.Draw(image).polygon([tuple(p) for p in polygon],outline="cyan",width=3)
                for number,(x,y) in enumerate(endpoints or ([marker] if marker is not None else [])):
                    draw=ImageDraw.Draw(image)
                    draw.ellipse((x-9,y-9,x+9,y+9),outline="#00ffff",width=3)
                    if endpoints:
                        draw.text((x+10,y-10),str(number+1),fill="cyan")
                path = output / f"candidate-{index+1}.png"
                image.save(path)
            images.append({"path": path.resolve(), "caption": candidate.get("captions",["RGB evidence"]*len(evidence["frames"]))[index]})
        prompt = candidate["prompt"]
    else:
        frame = bundle["frames"][evidence["frame"]]
        with Image.open(root / frame["rgb"]) as image:
            image = image.convert("RGB")
        x, y = evidence["pixel"]
        ImageDraw.Draw(image).ellipse((x-9, y-9, x+9, y+9), outline="#00ffff", width=3)
        marked = output / "candidate.png"
        image.save(marked)
        images.append({"path": marked.resolve(), "caption": "RGB candidate"})
        prompt = (f"Inspect the exact cyan-marked region. Proposed type: {candidate['name']!r}. "
                  "Determine if a scene entity is clearly visible at the mark and its type matches.")
    check_keys = _checks(candidate)
    prompt += (" If ambiguous report uncertain. Return JSON with decision (accept/reject/uncertain), "
               f"reason (one short visual observation), and boolean keys: {', '.join(check_keys)}.")
    if candidate.get("value_keys"):
        prompt+=" Also return values object with these enum fields: "+json.dumps(candidate["value_keys"])
    response = run_fixed_evidence(repo, config, output / "attempt", prompt, images,
                                  PromptLimits(max_requests=1, max_turns=1, seconds=180))
    valid_response=True
    try:
        answer = json.loads(response["answer"])
        if (answer["decision"] not in {"accept", "reject", "uncertain"} or
                any(type(answer[k]) is not bool for k in check_keys) or
                any(answer.get("values",{}).get(k) not in choices for k,choices in candidate.get("value_keys",{}).items()) or
                not isinstance(answer["reason"], str) or not answer["reason"].strip()):
            raise ValueError("invalid review answer")
    except (AttributeError, KeyError, TypeError, ValueError):
        valid_response=False
        answer = {"decision": "uncertain", **dict.fromkeys(check_keys, False),
                  "reason": "No valid model review: " + str(response.get("reason", "unknown"))}
    response_reason=response.get("reason")
    if response_reason in {"output_limit","time_budget","turn_limit","request_budget","action_budget","path_budget"}:
        execution_status="budget_exhausted"
    elif response_reason in {"runtime_error","service_error","model_error","core_error","tool_timeout","execution_uncertain"}:
        execution_status="service_error"
    else:
        execution_status="completed" if valid_response else "invalid_response"
    last=response.get("last") or {}
    return {"candidate_id": candidate["candidate_id"], "kind": candidate["kind"],
            "source": candidate["source"], "reviewer": reviewer, "decision": answer["decision"],
            "model":last.get("model") or config.get("model"),
            "reasoning_effort":config.get("reasoning_effort","high"),
            "response_reason":response_reason,"execution_status":execution_status,
            "checks": {k: answer[k] for k in check_keys}, "values":answer.get("values",{}), "reason": answer["reason"]}


def write_reviewed_bundle(bundle, root, reviews):
    """Persist separate review and reviewed bundle artifacts; leave capture unchanged."""
    root = Path(root)
    reviews = list(reviews)
    result = apply_reviews(bundle, root, reviews)
    private = root / "private"
    private.mkdir(exist_ok=True, parents=True)
    (private / "reviews.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reviews))
    write_json(private / "reviewed_bundle.json", result)
    return result
