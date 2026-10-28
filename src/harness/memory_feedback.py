"""Visual receipts for model-selected positions, using only public images and memory."""

from PIL import Image, ImageDraw

from .maps import _font
from .memory import location


def memory_feedback(session, operations, before):
    selected = {}
    for op in operations:
        identifier = op.get("id", op.get("keep"))
        fields = op.get("fields", {})
        if any(key in fields for key in ("location", "point", "anchor")):
            selected[identifier] = fields.get("location")
        for removed in op.get("remove", []):
            selected.pop(removed, None)
        if op["op"] == "delete":
            selected.pop(identifier, None)
    entities = {e["id"]: e for e in session.memory.entities}
    previous = {e["id"]: location(e) for e in before}
    # Merge can inherit a location without an explicit point/anchor field.
    for identifier, entity in entities.items():
        if location(entity) != previous.get(identifier):
            selected.setdefault(identifier, None)
    selected = {identifier: ref for identifier, ref in selected.items()
                if identifier in entities and location(entities[identifier]) is not None}
    if not selected:
        return None

    # Group marks on the same original image. Crops/grids resolve through the
    # existing index, so preview pixels can also be selected in later calls.
    grouped, selections = {}, []
    for identifier, ref in selected.items():
        if isinstance(ref, dict) and "image" in ref:
            image, pixel = session.observations.resolve_pixel(ref["image"], ref["pixel"])
            mark = {"id": identifier, "image": image, "pixel": pixel}
            selections.append(mark)
            grouped.setdefault(image, []).append(mark)
    sources = list(grouped)[:2]
    mapped = session.renderer.render(session.current["pose"], [],
                                     view={"around": list(selected)}, highlight=list(selected))
    canvas = Image.new("RGB", (960 if sources else 480, 520), "white")
    draw, transforms = ImageDraw.Draw(canvas), []
    font = _font(14)
    draw.text((8, 5), "Saved positions | model selections, not verified", fill="#333333", font=font)

    def panel(reference, x, y, width, height, title, marks=()):
        row = session.observations.image(reference)
        with Image.open(session.workspace / row["path"]) as original:
            image = original.convert("RGB")
        original_width, original_height = image.size
        image.thumbnail((width - 16, height - 30))
        left, top = x + (width - image.width) // 2, y + 24
        canvas.paste(image, (left, top))
        draw.text((x + 8, y + 3), title, fill="#333333", font=font)
        sx, sy = original_width / image.width, original_height / image.height
        transforms.append({"image": reference, "bounds": [left, top, left + image.width, top + image.height],
                           "pixel_to_source": [[sx, 0, -sx * left], [0, sy, -sy * top]]})
        for mark in marks:
            px, py = left + mark["pixel"][0] / sx, top + mark["pixel"][1] / sy
            draw.ellipse((px - 8, py - 8, px + 8, py + 8), outline="#dc2626", width=2)
            draw.text((max(left, min(px + 10, left + image.width - 48)),
                       max(top, min(py - 20, top + image.height - 20))), mark["id"],
                      fill="#dc2626", stroke_width=1, stroke_fill="white", font=font)

    for i, source in enumerate(sources):
        size = session.observations.image(source)["size"]
        height = 480 // len(sources)
        panel(source, 0, 28 + i * height, 480, height,
              f"Selected: {source} ({size[0]}x{size[1]})", grouped[source])
    panel(mapped["view_id"], 480 if sources else 0, 28, 480, 480,
          f"Map: {mapped['view_id']} | saved IDs circled")
    saved = session.observations._save_view(canvas, transforms)
    result = {"image": saved["image"], "path": saved["path"], "size": saved["size"],
              "map": mapped["view_id"], "ids": list(selected),
              "selected": [mark for mark in selections if mark["image"] in sources]}
    if len(grouped) > len(sources):
        result["omitted_source_images"] = len(grouped) - len(sources)
    return result
