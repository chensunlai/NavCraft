"""Read-only actor resources and image-coordinate map annotations."""

from dataclasses import dataclass
import math
from pathlib import Path, PurePosixPath
import re

from PIL import Image, ImageDraw, ImageFont

from .map import MapCalibration


class AssetError(ValueError):
    """An actor requested an unavailable resource or invalid image operation."""


@dataclass(frozen=True)
class ImageResource:
    path: Path
    kind: str
    # Normalized extent on floor_plan.png; None for non-map imagery.
    map_extent: tuple[float, float, float, float] | None = None


def compose_images(
    entries: list[tuple[str, Path]], destination: Path, columns=3
) -> Path:
    """Aspect-preserving, labeled contact sheet; full-size originals stay available."""
    if not entries or len(entries) > 10 or columns not in (1, 2, 3):
        raise AssetError("Contact sheets accept 1 to 10 images and 1 to 3 columns")
    columns = min(columns, len(entries))
    cell_width, cell_height, label_height = 640, 420, 34
    rows = math.ceil(len(entries) / columns)
    output = Image.new(
        "RGB", (columns * cell_width, rows * (cell_height + label_height)), "#eeeeee"
    )
    draw = ImageDraw.Draw(output)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for index, (label, path) in enumerate(entries):
        left, top = (index % columns) * cell_width, (index // columns) * (
            cell_height + label_height
        )
        with Image.open(path) as source:
            picture = source.convert("RGB")
        picture.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
        output.paste(
            picture,
            (
                left + (cell_width - picture.width) // 2,
                top + label_height + (cell_height - picture.height) // 2,
            ),
        )
        draw.text((left + 8, top + 7), label, fill="black", font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.save(destination)
    return destination


class SceneAssets:
    description_ref = "scene_description.md"
    map_ref = "floor_plan.png"
    rooms_ref = "evidence_rooms.jpg"

    def __init__(self, scene_id: str, root: Path, directory: Path):
        self.root = Path(root).resolve()
        self.directory = Path(directory).resolve()
        matches = []
        scene_parts = PurePosixPath(str(scene_id)).parts
        for candidate in self.root.glob("*/*/*/scene_description.md"):
            name = candidate.parent.name
            short_id = name.split("-", 1)[-1]
            if (
                name in scene_parts
                or short_id in scene_parts
                or short_id == scene_id
                or any(part.startswith(short_id + ".") for part in scene_parts)
            ):
                matches.append(candidate.parent)
        if len(matches) != 1:
            raise AssetError(
                f"Expected one archived scene for {scene_id!r}; found {len(matches)}"
            )
        self.scene_directory = matches[0].resolve()
        if (
            self.directory == self.scene_directory
            or self.scene_directory in self.directory.parents
        ):
            raise AssetError(
                "Runtime assets must be outside the immutable scene archive"
            )
        self.runtime_directory = self.directory / "assets"
        self.runtime_directory.mkdir(parents=True, exist_ok=True)
        self.images: dict[str, ImageResource] = {}
        self.texts: dict[str, Path] = {}
        self.latest_map_ref = self.map_ref
        self.counter = 0
        self.calibration = None

    @staticmethod
    def _clean_ref(ref: str) -> str:
        if not isinstance(ref, str) or not ref or "\\" in ref or "\x00" in ref:
            raise AssetError("Resource reference must be a relative resource ID")
        path = PurePosixPath(ref)
        if (
            path.is_absolute()
            or any(part in ("..", ".") for part in ref.split("/"))
            or ":" in ref
        ):
            raise AssetError(
                "Resource reference must not contain absolute paths or traversal"
            )
        return str(path)

    @staticmethod
    def _is_rgb(path: Path) -> bool:
        return (
            path.suffix.lower() in (".jpg", ".jpeg", ".png")
            and not re.search(r"depth|rgb[-_]?d", path.name.lower())
            and not any(part.lower() in ("private", "depth") for part in path.parts)
        )

    def _archive_path(self, ref: str, image: bool) -> Path:
        allowed = (
            ref in (self.map_ref, self.rooms_ref)
            if image
            else ref == self.description_ref
        )
        if image and ref.startswith("evidence/") and self._is_rgb(Path(ref)):
            allowed = not any(
                part in ("raw", "private", "depth") for part in PurePosixPath(ref).parts
            )
        if not allowed:
            raise AssetError("Resource is not part of the actor's file memory")
        path = (self.scene_directory / ref).resolve()
        if self.scene_directory not in path.parents or not path.is_file():
            raise AssetError("Resource is unavailable or leaves the scene archive")
        resolved_ref = str(path.relative_to(self.scene_directory))
        if "raw" in PurePosixPath(resolved_ref).parts or (
            image and not self._is_rgb(Path(resolved_ref))
        ):
            raise AssetError("Resource resolves to a private or depth asset")
        if image and not self._is_rgb(Path(ref)):
            raise AssetError("Only RGB images are readable")
        return path

    def _resource(self, ref: str) -> tuple[str, ImageResource]:
        ref = self._clean_ref(ref)
        if ref not in self.images:
            path = self._archive_path(ref, image=True)
            self.images[ref] = ImageResource(
                path,
                "map" if ref == self.map_ref else "evidence",
                (0, 0, 1, 1) if ref == self.map_ref else None,
            )
        return ref, self.images[ref]

    def path_for_image(self, ref: str) -> Path:
        return self._resource(ref)[1].path

    def image_info(self, ref: str) -> dict:
        ref, resource = self._resource(ref)
        with Image.open(resource.path) as picture:
            width, height = picture.size
        return {"ref": ref, "width": width, "height": height, "kind": resource.kind}

    def _next_ref(self, kind: str) -> str:
        self.counter += 1
        return f"runtime/{kind}_{self.counter:05d}"

    def register_image(self, path: Path, ref=None, kind="observation") -> str:
        path = Path(path).resolve()
        if not path.is_file() or not self._is_rgb(path):
            raise AssetError("Only existing RGB images may be registered")
        with Image.open(path) as picture:
            picture.verify()
        ref = self._clean_ref(ref if ref is not None else self._next_ref(kind))
        if ref in (
            self.map_ref,
            self.rooms_ref,
            self.description_ref,
        ) or ref.startswith(("raw/", "evidence/")):
            raise AssetError("Runtime IDs must not shadow archived resources")
        if ref in self.texts:
            raise AssetError("Resource ID already refers to text")
        if ref in self.images and self.images[ref].path != path:
            raise AssetError("Image ID already refers to another image")
        self.images[ref] = ImageResource(
            path, kind, (0, 0, 1, 1) if kind == "map" else None
        )
        return ref

    def read_text(self, ref: str, start_line=1, max_lines=200) -> dict:
        ref = self._clean_ref(ref)
        if (
            type(start_line) is not int
            or type(max_lines) is not int
            or start_line < 1
            or not 1 <= max_lines <= 400
        ):
            raise AssetError("Use positive start_line and 1 to 400 max_lines")
        path = self.texts.get(ref) or self._archive_path(ref, image=False)
        lines = path.read_text(encoding="utf-8").splitlines()
        selected = lines[start_line - 1 : start_line - 1 + max_lines]
        return {
            "ref": ref,
            "start_line": start_line,
            "total_lines": len(lines),
            "text": "\n".join(
                f"{number}: {line}" for number, line in enumerate(selected, start_line)
            ),
            "next_line": (
                start_line + len(selected)
                if start_line + len(selected) <= len(lines)
                else None
            ),
        }

    @staticmethod
    def _coordinates(value, length: int) -> list[float]:
        if (
            not isinstance(value, (list, tuple))
            or len(value) != length
            or any(
                type(item) not in (int, float)
                or not math.isfinite(item)
                or not 0 <= item <= 1
                for item in value
            )
        ):
            raise AssetError(
                f"Expected {length} finite normalized coordinates in [0,1]"
            )
        return [float(item) for item in value]

    def view_images(
        self, refs: list[str], layout="separate", crops=None
    ) -> tuple[dict, list[Path]]:
        if (
            not isinstance(refs, list)
            or not 1 <= len(refs) <= 10
            or layout not in ("separate", "grid")
        ):
            raise AssetError("Use 1 to 10 image IDs and separate or grid layout")
        crops = [None] * len(refs) if crops is None or crops == [] else crops
        if not isinstance(crops, list) or len(crops) != len(refs):
            raise AssetError("crops must have one entry per image")
        result_refs, sources = [], []
        for reference, crop in zip(refs, crops):
            ref, resource = self._resource(reference)
            if crop is not None:
                left, top, right, bottom = self._coordinates(crop, 4)
                if left >= right or top >= bottom:
                    raise AssetError("Crop must have positive width and height")
                with Image.open(resource.path) as original:
                    width, height = original.size
                    box = (
                        math.floor(left * width),
                        math.floor(top * height),
                        math.ceil(right * width),
                        math.ceil(bottom * height),
                    )
                    cropped = original.crop(box).convert("RGB")
                crop_ref = self._next_ref("crop")
                destination = self.runtime_directory / f"crop_{self.counter:05d}.png"
                cropped.save(destination)
                extent = None
                if resource.map_extent is not None:
                    a, b, c, d = resource.map_extent
                    extent = (
                        a + box[0] / width * (c - a),
                        b + box[1] / height * (d - b),
                        a + box[2] / width * (c - a),
                        b + box[3] / height * (d - b),
                    )
                self.images[crop_ref] = ImageResource(
                    destination, resource.kind, extent
                )
                sources.append(
                    {
                        "ref": crop_ref,
                        "source_ref": ref,
                        "crop": [
                            box[0] / width,
                            box[1] / height,
                            box[2] / width,
                            box[3] / height,
                        ],
                    }
                )
                ref = crop_ref
            else:
                sources.append({"ref": ref, "source_ref": ref})
            result_refs.append(ref)
        if layout == "grid":
            grid_ref = self._next_ref("grid")
            destination = self.runtime_directory / f"grid_{self.counter:05d}.jpg"
            compose_images(
                [(ref, self.path_for_image(ref)) for ref in result_refs], destination
            )
            self.images[grid_ref] = ImageResource(destination, "grid")
            return {
                "layout": layout,
                "images": [self.image_info(grid_ref)],
                "sources": sources,
            }, [destination]
        return {
            "layout": layout,
            "images": [self.image_info(ref) for ref in result_refs],
            "sources": sources,
        }, [self.path_for_image(ref) for ref in result_refs]

    def _arrow(
        self, origin, heading, label, note="", level_id=None
    ) -> tuple[dict, list[Path]]:
        source = self.path_for_image(self.map_ref)
        with Image.open(source) as original:
            picture = original.convert("RGB")
        width, height = picture.size
        a = (origin[0] * width, origin[1] * height)
        dx, dy = (heading[0] - origin[0]) * width, (heading[1] - origin[1]) * height
        length = math.hypot(dx, dy)
        if length < 1e-8:
            raise AssetError("Heading must differ from position")
        arrow_length = max(35, min(width, height) * 0.04)
        dx, dy = dx / length, dy / length
        tip = (a[0] + dx * arrow_length, a[1] + dy * arrow_length)
        draw = ImageDraw.Draw(picture)
        line_width = max(4, round(min(width, height) / 250))
        draw.line([a, tip], fill="#e02020", width=line_width)
        draw.polygon(
            [
                tip,
                (tip[0] - dx * 15 - dy * 8, tip[1] - dy * 15 + dx * 8),
                (tip[0] - dx * 15 + dy * 8, tip[1] - dy * 15 - dx * 8),
            ],
            fill="#e02020",
        )
        draw.ellipse((a[0] - 5, a[1] - 5, a[0] + 5, a[1] + 5), fill="#e02020")
        draw.rectangle((5, 5, 290, 34), fill="white")
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 19)
        except OSError:
            font = ImageFont.load_default()
        draw.text((10, 9), label, fill="#b00000", font=font)
        ref = self._next_ref("map")
        destination = self.runtime_directory / f"map_{self.counter:05d}.png"
        picture.save(destination)
        self.images[ref] = ImageResource(destination, "map", (0, 0, 1, 1))
        self.latest_map_ref = ref
        result = {
            "map_ref": ref,
            "marker_kind": label,
            "note": note,
            "images": [self.image_info(ref)],
        }
        if level_id is not None:
            result["level_id"] = level_id
        return result, [destination]

    def initial_map(self, pose: dict) -> tuple[dict, list[Path]]:
        if self.calibration is None:
            try:
                self.calibration = MapCalibration(self.scene_directory)
            except (ValueError, OSError, KeyError, IndexError) as error:
                raise AssetError(f"Cannot calibrate archived map: {error}") from error
        with Image.open(self.path_for_image(self.map_ref)) as image:
            _, _, svg_width, svg_height = self.calibration.viewbox
            if not math.isclose(
                image.width / image.height, svg_width / svg_height, rel_tol=0.001
            ):
                raise AssetError(
                    "Archived PNG and calibration SVG have different aspect ratios"
                )
        try:
            level_id, origin, heading = self.calibration.initial_arrow(pose)
        except (ValueError, KeyError, TypeError) as error:
            raise AssetError(f"Cannot mark initial pose: {error}") from error
        return self._arrow(origin, heading, "Initial pose hint", level_id=level_id)

    def mark_pose(
        self, map_ref: str, position, heading, note: str
    ) -> tuple[dict, list[Path]]:
        _, resource = self._resource(map_ref)
        if resource.map_extent is None:
            raise AssetError(
                "Pose markers require a map image, not a contact sheet or RGB frame"
            )
        position, heading = self._coordinates(position, 2), self._coordinates(
            heading, 2
        )
        if not isinstance(note, str) or len(note) > 2000:
            raise AssetError("Pose note must be a short text")
        left, top, right, bottom = resource.map_extent

        def original_point(point):
            return (left + point[0] * (right - left), top + point[1] * (bottom - top))

        return self._arrow(
            original_point(position), original_point(heading), "Model estimate", note
        )

    def write_summary(self, text: str) -> str:
        ref = self._next_ref("summary") + ".md"
        destination = self.runtime_directory / f"summary_{self.counter:05d}.md"
        destination.write_text(text, encoding="utf-8")
        self.texts[ref] = destination
        return ref
