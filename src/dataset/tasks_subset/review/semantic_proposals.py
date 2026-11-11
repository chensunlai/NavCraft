"""RGB type proposals for semantic surfaces, including non-passable openings."""
import re


def opening_type_proposals(bundle):
    objects={o["id"]:o for o in bundle["scene"].get("objects",[])}
    vocabulary=re.compile(r"door|window|mirror|glass|门|窗|镜|玻璃",re.I)
    for sight in bundle["sightings"]:
        annotation=objects.get(sight["id"])
        if (sight["kind"]!="object" or not annotation or
                sight.get("method")!="foreground_in_schematic_bounds" or
                not vocabulary.search(str(annotation.get("name",""))+" "+str(annotation.get("category","")))):
            continue
        yield {"kind":"judgment_opening_surface_type","name":annotation.get("name") or "surface",
               "frames":[sight["frame"]],"markers":[sight["pixel"]],"annotations":[annotation],
               "prompt":"Inspect the cyan-marked physical entity and its surrounding frame, border and full "
                        "visible surface. Classify it as door, opening (unobstructed passage), window, mirror "
                        "or glass barrier. Reflected/through-glass objects do not identify the surface. "
                        "Set surface_bound only if the mark is on or clearly identifies that entity. "
                        "Set type_supported only if its type is visually supported. "
                        "For a door report open/closed/unknown state; for other types report not_applicable.",
               "check_keys":["surface_bound","type_supported"],
               "value_keys":{"surface_type":["door","opening","window","mirror","glass"],
                             "door_state":["open","closed","unknown","not_applicable"]},
               "fact":{"section":"judgment_evidence","key":"opening_types","value":{
                   "frame":sight["frame"],"object":sight["id"],"pixel":sight["pixel"]}}}

    # Annotated opening segments may describe closed doors or windows; neither visibility
    # nor traversability is inferred from their status. Review must identify the surface.
    from ..sampling.geometry import project
    levels={level["id"]:level["floor_height"] for level in bundle["scene"].get("levels",[])}
    for opening in bundle["scene"].get("openings",[]):
        if opening.get("level") not in levels or not opening.get("segment_xz"):
            continue
        (ax,az),(bx,bz)=opening["segment_xz"]
        center=[(ax+bx)/2,levels[opening["level"]]+.8,(az+bz)/2]
        for key,frame in bundle["frames"].items():
            if (frame.get("stream","").startswith(("branch-","replay-")) or
                    not frame.get("camera") or not frame.get("world_from_optical") or not frame.get("pose")):
                continue
            p=project(center,frame["camera"],frame["world_from_optical"])
            if p is None or not .4<p[2]<6:
                continue
            pixel=[round(p[0]),round(p[1])]
            yield {"kind":"judgment_opening_surface_type","name":opening.get("kind","opening"),
                   "frames":[key],"markers":[pixel],"annotations":[opening],
                   "prompt":"Inspect the cyan-marked candidate and surrounding physical frame. A map only "
                            "proposed this location; it does not establish visibility or type. Classify its "
                            "physical type as door/opening/window/mirror/glass. Distinguish the surface from "
                            "reflections or objects behind glass. Set surface_bound only if the mark clearly "
                            "identifies that physical entity; type_supported only if visible cues establish it. "
                            "For a door report open/closed/unknown, otherwise not_applicable.",
                   "check_keys":["surface_bound","type_supported"],
                   "value_keys":{"surface_type":["door","opening","window","mirror","glass"],
                                 "door_state":["open","closed","unknown","not_applicable"]},
                   "fact":{"section":"judgment_evidence","key":"opening_types","value":{
                       "frame":key,"opening":opening["id"],"pixel":pixel}}}

    # Reflective surfaces can have depth belonging to the reflected scene. Use the
    # annotated surface only to propose an RGB region, never as proof of its type.
    for annotation in objects.values():
        name=str(annotation.get("name",""))+" "+str(annotation.get("category",""))
        if not re.search(r"mirror|glass|window|door|镜|玻璃|窗户|房门",name,re.I):
            continue
        if not annotation.get("center_xyz"):
            continue
        for key,frame in bundle["frames"].items():
            if (frame.get("stream","").startswith(("branch-","replay-")) or
                    not frame.get("camera") or not frame.get("world_from_optical")):
                continue
            p=project(annotation["center_xyz"],frame["camera"],frame["world_from_optical"])
            if p is None or not .4<p[2]<6:
                continue
            pixel=[round(p[0]),round(p[1])]
            yield {"kind":"judgment_opening_surface_type","name":annotation.get("name"),
                   "frames":[key],"markers":[pixel],"annotations":[annotation],
                   "prompt":"The cyan marker is an unverified projected surface candidate. Inspect its full "
                            "frame, border and context in RGB. Does it identify a physical door, opening, window, "
                            "mirror or glass barrier? Do not classify a reflected entity as the surface. "
                            "Set surface_bound only if the marker clearly identifies the physical surface and "
                            "type_supported only if visible cues establish its type. Door state is "
                            "open/closed/unknown, or not_applicable for other types.",
                   "check_keys":["surface_bound","type_supported"],
                   "value_keys":{"surface_type":["door","opening","window","mirror","glass"],
                                 "door_state":["open","closed","unknown","not_applicable"]},
                   "fact":{"section":"judgment_evidence","key":"opening_types","value":{
                       "frame":key,"object":annotation["id"],"pixel":pixel,"proposal_method":"annotated_surface_projection"}}}


def surface_region_proposals(bundle):
    """Project a proposed whole surface outline; RGB reviewers must validate the outline."""
    from ..sampling.geometry import project
    from itertools import product
    for obj in bundle["scene"].get("objects",[]):
        if not re.search(r"mirror|glass|window|door|镜|玻璃|窗户|房门",str(obj.get("name","")),re.I):
            continue
        center,size=obj.get("center_xyz"),obj.get("size_xyz")
        if not center or not size:
            continue
        # A thin authored box only proposes a plane. The outline is not a reviewed boundary.
        axis=0 if size[0]>=size[2] else 2
        corners=[]
        for horizontal,vertical in ((-1,-1),(1,-1),(1,1),(-1,1)):
            point=list(center);point[axis]+=horizontal*size[axis]/2;point[1]+=vertical*size[1]/2
            corners.append(point)
        for key,frame in bundle["frames"].items():
            if (frame.get("stream","").startswith(("branch-","replay-")) or
                    not frame.get("camera") or not frame.get("world_from_optical")):
                continue
            points=[project(p,frame["camera"],frame["world_from_optical"]) for p in corners]
            if any(p is None for p in points):
                continue
            polygon=[[round(p[0]),round(p[1])] for p in points]
            area=abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(polygon,polygon[1:]+polygon[:1])))/2
            if area<1500 or max(p[2] for p in points)>6:
                continue
            yield {"kind":"judgment_opening_surface_region_type","name":obj.get("name"),
                   "frames":[key],"markers":[None],"polygons":[polygon],"annotations":[obj],
                   "prompt":"The cyan outline proposes the entire physical surface under consideration. "
                            "First verify that this outline actually follows one visible framed physical surface, "
                            "not a floor patch, reflected object or unrelated plane. Set surface_bound only then. "
                            "Classify that outlined surface as door/opening/window/mirror/glass, using the frame "
                            "and surrounding context as well as reflections. Set type_supported only if visible "
                            "evidence supports it. Door state: open/closed/unknown, otherwise not_applicable.",
                   "check_keys":["surface_bound","type_supported"],
                   "value_keys":{"surface_type":["door","opening","window","mirror","glass"],
                                 "door_state":["open","closed","unknown","not_applicable"]},
                   "fact":{"section":"judgment_evidence","key":"opening_types","value":{
                       "frame":key,"object":obj["id"],"polygon":polygon,
                       "proposal_method":"annotated_surface_outline"}}}
