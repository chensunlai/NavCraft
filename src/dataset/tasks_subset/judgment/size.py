"""Select and validate visible physical support-span endpoints; never infer unseen extents."""
from hashlib import sha256
import json
from math import dist, isfinite
from pathlib import Path

from ..sampling import geometry


MEASUREMENT = ("complete visible horizontal outside-to-outside span of one explicitly named physical "
               "part of the object; both endpoints must be visible boundaries of that same part at the "
               "same height, with no inferred hidden extent")


def endpoint_proposals(bundle):
    """A reproducible view-pair proposal; the model selects pixels, not metric size."""
    objects={o["id"]:o for o in bundle["scene"].get("objects",[])}
    seen={}
    for sight in bundle["sightings"]:
        if sight["kind"]=="object" and sight.get("method")=="foreground_in_schematic_bounds":
            seen.setdefault(sight["id"],[]).append(sight)
    for identifier,views in sorted(seen.items()):
        for first in views:
            for second in views:
                a,b=(bundle["frames"][s["frame"]] for s in (first,second))
                if (.55 <= dist(a["pose"]["position"],b["pose"]["position"]) <= 5
                        and first["frame"]!=second["frame"]):
                    yield {"object":identifier,"name":objects[identifier].get("name"),
                           "frames":[first["frame"],second["frame"]],
                           "markers":[first["pixel"],second["pixel"]],
                           "measurement":MEASUREMENT}
                    break
            else:
                continue
            break


def validate_endpoint_selection(bundle, root, proposal, selection):
    """Consume saved model/human pixel selection, verify real depth and second-view correspondence."""
    root=Path(root)
    if selection.get("measurement")!=MEASUREMENT:
        raise ValueError("Selection must identify the exact physical span")
    if selection.get("axis") not in {"width","depth","length"} or not selection.get("axis_description"):
        raise ValueError("A supported axis enum and visual axis description are required")
    pixels=selection.get("pixels")
    if not isinstance(pixels,list) or len(pixels)!=2 or any(
            not isinstance(p,list) or len(p)!=2 or any(type(v) not in (int,float) or not isfinite(v) for v in p) for p in pixels):
        raise ValueError("Expected exactly two real image pixels")
    first,second=(bundle["frames"][k] for k in proposal["frames"])
    depths=[geometry.read_depth(root/f["depth"],f["camera"]) for f in (first,second)]
    import numpy as np
    masks=[]
    obj=next((o for o in bundle["scene"].get("objects",[]) if o["id"]==proposal["object"]),None)
    if obj is None or not obj.get("semantic_ids"):
        raise ValueError("Physical endpoint selection requires a bound native object mapping")
    for key,frame in zip(proposal["frames"],(first,second)):
        row=bundle.get("instance_evidence",{}).get(key,{})
        record=row.get("objects",{}).get(proposal["object"],{})
        if (not row.get("valid") or not record.get("mapping_valid") or record.get("mapping_ambiguous") or
                record.get("mapped_ids")!=obj["semantic_ids"] or row.get("pose")!=frame.get("pose")
                or row.get("camera")!=frame.get("camera") or
                sha256((root/frame["rgb"]).read_bytes()).hexdigest()!=row.get("source_rgb_sha256")
                or sha256((root/row["mask"]).read_bytes()).hexdigest()!=row.get("mask_sha256")):
            raise ValueError("Native endpoint identity cache is absent or stale")
        masks.append(np.load(root/row["mask"],allow_pickle=False))
    worlds=[];other_pixels=[]
    for pixel in pixels:
        x,y=map(lambda value:int(round(value)),pixel)
        if not 1<=x<first["camera"]["width"]-1 or not 1<=y<first["camera"]["height"]-1:
            raise ValueError("Endpoint pixel outside the first image")
        if int(masks[0][y,x]) not in obj["semantic_ids"]:
            raise ValueError("Selected endpoint is not on the specified physical native object")
        world=geometry.unproject([x,y],depths[0][y,x],first["camera"],first["world_from_optical"])
        if world is None:
            raise ValueError("Endpoint has no valid depth")
        point=geometry.visible_point(world,depths[1],second["camera"],second["world_from_optical"],tolerance=.045)
        if point is None:
            raise ValueError("Endpoint is not observed consistently in the second actual view")
        if int(masks[1][point[1],point[0]]) not in obj["semantic_ids"]:
            raise ValueError("Second-view endpoint is not the same native object")
        worlds.append(world);other_pixels.append(point)
    if (abs(worlds[0][1]-worlds[1][1])>.06 or dist(worlds[0],worlds[1])<.35 or
            dist(*pixels)<24 or dist(*other_pixels)<24):
        raise ValueError("Span is not a resolved pair on one horizontal plane")
    if proposal.get("axis") and selection["axis"]!=proposal["axis"]:
        raise ValueError("Selected axis differs from the requested physical measurement")
    frames=[]
    for key,pair in zip(proposal["frames"],(pixels,other_pixels)):
        frame=bundle["frames"][key]
        frames.append({"frame":key,"sha256":sha256((root/frame["rgb"]).read_bytes()).hexdigest(),
                       "pixels":pair})
    from .calibration import public_intersection
    state=next((s for s in bundle["states"] if set(proposal["frames"])<=set(s["frame_keys"])),None)
    if state is None:
        raise ValueError("The two views have no common observed prefix")
    reconstructed=[]
    for pixel,other,world in zip(pixels,other_pixels,worlds):
        point=public_intersection(first,second,pixel,other,state["transform"])
        if point is None or dist(point,[world[0],world[2]])>.15:
            raise ValueError("Span is not independently reconstructable from public camera inputs")
        reconstructed.append(point)
    return {**proposal,"source":frames,"public_reconstruction_xz":reconstructed,"world_endpoints":worlds,"length_m":dist([worlds[0][0],worlds[0][2]],[worlds[1][0],worlds[1][2]]),
            "selection":selection,"selection_sha256":sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest()}


def span_review_proposals(bundle,root):
    """Both reviewers must confirm physical membership and complete extent from the real images."""
    for saved in bundle.get("size_endpoint_selections",[]):
        span=validate_endpoint_selection(bundle,root,saved,saved["selection"])
        if span!=saved:
            raise ValueError("Endpoint geometry or source changed")
        yield {"kind":"judgment_physical_span","name":span["name"],"frames":span["frames"],
               "markers":[None,None],"endpoint_markers":[s["pixels"] for s in span["source"]],
               "annotations":[span],
               "prompt":f"Object: {span['name']}. The endpoint circles in both images propose this exact measure: "
                         f"{MEASUREMENT}. Reported axis: {span['selection']['axis']} "
                         f"({span['selection']['axis_description']}). Are BOTH endpoints visible physical boundaries of this named object part, and "
                         "do they show its entire outside-to-outside span along the same identifiable axis? "
                         "Reject marks on shadows, unrelated floor or points behind an occluder. "
                         "Set axis_supported only if the reported axis enum and axis_description match the pictures.",
               "check_keys":["object_membership","complete_extent","axis_supported"],
               "fact":{"section":"judgment_evidence","key":"physical_spans","value":{"selection":span}}}


def select_endpoints(repo, config, output, bundle, root, proposal, reviewer):
    """Save the raw pixel selection for replay/import; no size or private coordinates reach the model."""
    from ..review.model import run_fixed_evidence
    from src.harness.runtime.request import PromptLimits
    from PIL import Image,ImageDraw
    output=Path(output);output.mkdir(parents=True,exist_ok=False);images=[]
    for i,(key,mark) in enumerate(zip(proposal["frames"],proposal["markers"])):
        with Image.open(Path(root)/bundle["frames"][key]["rgb"]) as im:
            im=im.convert("RGB")
        x,y=mark;ImageDraw.Draw(im).ellipse((x-8,y-8,x+8,y+8),outline="cyan",width=2)
        path=output/f"source-{i}.png";im.save(path);images.append({"path":path.resolve(),"caption":f"Object view {i+1}"})
    prompt=f"Select the exact physical object identified by the cyan mark: {proposal['name']}. Required measure: {MEASUREMENT}. "
    prompt+=" Requested physical part: "+proposal.get("part_description","Select one fully visible horizontal physical edge, explicitly name it.")+". "
    prompt+="Return JSON {pixels:[[x1,y1],[x2,y2]],axis:width|depth|length,axis_description:short visible physical axis description,reason:short reason} using FIRST image pixels. If any full span is not visible, return {uncertain:true,reason:...}. Do not estimate hidden boundaries or provide a size."
    result=run_fixed_evidence(repo,config,output/"attempt",prompt,images,PromptLimits(max_requests=1,max_turns=1,seconds=180))
    (output/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    selection=json.loads(result.get("answer","null"))
    if not isinstance(selection,dict) or selection.get("uncertain"):
        return None
    selection.update(measurement=MEASUREMENT,reviewer=reviewer)
    return validate_endpoint_selection(bundle,root,proposal,selection)


def build_size_comparisons(c):
    from .construction import arr,enum,mark,row
    from .calibration import public_camera,shared_pair
    from itertools import combinations
    spans=[r for r in c.bundle.get("judgment_evidence",{}).get("physical_spans",[])
           if r.get("review_candidate") and set(r["selection"]["frames"])<=c.keys]
    for first,second in combinations(spans,2):
        a,b=first["selection"],second["selection"]
        if a["object"]==b["object"] or a["selection"]["axis"]!=b["selection"]["axis"]:
            continue
        if abs(a["length_m"]-b["length_m"])<=.3:
            continue
        rng=__import__("random").Random(f"{c.bundle.get('seed')}:{c.state['id']}:{a['object']}:{b['object']}")
        pair=[a,b];rng.shuffle(pair);pictures=[]
        for name,span in zip("AB",pair):
            reverse=bool(rng.getrandbits(1))
            for source in span["source"]:
                pictures.append(row(source["frame"],f"Object {name}; "+public_camera(c,source["frame"]),
                                    [(f"{name}{i+1}",p) for i,p in enumerate(list(reversed(source["pixels"])) if reverse else source["pixels"])]))
        answer="A" if pair[0]["length_m"]>pair[1]["length_m"] else "B"
        c.put("compare_distance_size","Which physical object has the larger complete horizontal physical span "
              f"at these specified parts? A: {pair[0]['selection']['axis_description']}; "
              f"B: {pair[1]['selection']['axis_description']}. Both use the defined {a['selection']['axis']} axis. "
              "The paired endpoint marks are actual visible boundaries, not image widths.",
              enum("A","B"),[answer],pictures,private={"variant":"physical_horizontal_span_comparison",
                                                     "span_facts":[first,second],"margin_m":.3,
                                                     "choice_objects":{name:span["object"] for name,span in zip("AB",pair)}})


def run_endpoints(bundle,root,source,config,candidates=()):
    """Export bound candidates, optionally select pixels with one model, save an unreviewed size bundle."""
    from copy import deepcopy
    from src.common.config import read_config
    from ..bundle import write_json
    root,source=Path(root),Path(source)
    output=root/"private/size_bundle.json"
    if source.resolve()==output.resolve():
        raise ValueError("Endpoint stage output must differ from its source bundle")
    settings=config.get("endpoints",{})
    proposals=list(endpoint_proposals(bundle))
    if settings.get("pairs"):
        proposals=[]
        names={o["id"]:o for o in bundle["scene"]["objects"]}
        for pair in settings["pairs"]:
            identifier,keys=pair["object"],pair["frames"]
            if len(keys)!=2 or identifier not in names:
                raise ValueError("Each endpoint pair requires a known object and two frames")
            marks=[next((s["pixel"] for s in bundle["sightings"] if s["kind"]=="object"
                         and s["id"]==identifier and s["frame"]==key),None) for key in keys]
            if None in marks:
                raise ValueError("Endpoint source view lacks a grounded candidate mark")
            proposals.append({"object":identifier,"name":names[identifier]["name"],
                              "frames":keys,"markers":marks,"measurement":MEASUREMENT,
                              **{k:pair[k] for k in ("axis","part_description") if k in pair}})
    indexed={}
    for proposal in proposals:
        annotation=next(o for o in bundle["scene"]["objects"] if o["id"]==proposal["object"])
        binding={"frames":[{"frame":key,"sha256":sha256((root/bundle["frames"][key]["rgb"]).read_bytes()).hexdigest()}
                           for key in proposal["frames"]],
                 "annotation_sha256":sha256(json.dumps(annotation,sort_keys=True).encode()).hexdigest(),
                 "proposal":proposal}
        identifier=sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()[:24]
        indexed[identifier]={"candidate_id":identifier,**proposal,"binding":binding}
    write_json(root/"private/size_candidates.json",{"source":str(source),
               "bundle_sha256":sha256(source.read_bytes()).hexdigest(),"candidates":list(indexed.values())})
    if set(candidates)-indexed.keys():
        raise ValueError("Unknown size endpoint candidate")
    result=deepcopy(bundle)
    models=config.get("review",{}).get("models",[])
    if not models:
        write_json(output,result)
        return result
    selected=[indexed[k] for k in dict.fromkeys(candidates)] if candidates else list(indexed.values())[:settings.get("limit",1)]
    model=read_config(models[0])
    for proposal in selected:
        directory=root/"private/size_attempts"/proposal["candidate_id"]
        if directory.exists():
            if (directory/"selection.json").exists():
                saved=json.loads((directory/"selection.json").read_text())
                checked=validate_endpoint_selection(bundle,root,saved,saved["selection"])
                if checked!=saved:
                    raise ValueError("Saved endpoint selection no longer binds to its source")
                span=saved
            else:
                continue
        else:
            try:
                span=select_endpoints(Path.cwd(),model,directory,bundle,root,proposal,
                                      f"{model['model']}/{model.get('reasoning_effort','high')}")
                if span:
                    write_json(directory/"selection.json",span)
                else:
                    write_json(directory/"failure.json",{"reason":"model_uncertain"})
            except Exception as exc:
                directory.mkdir(parents=True,exist_ok=True)
                write_json(directory/"failure.json",{"reason":type(exc).__name__,"details":str(exc)})
                span=None
        if span:
            records=result.setdefault("size_endpoint_selections",[])
            if span not in records:
                records.append(span)
    write_json(output,result)
    return result
