"""Small deterministic label placement; anchors always keep their true positions."""

import math


def overlaps(a, b, gap=0):
    return (a[0] < b[2] + gap and a[2] + gap > b[0]
            and a[1] < b[3] + gap and a[3] + gap > b[1])


def layout_labels(items, bounds, obstacles=()):
    """Place measured label boxes, prioritising rooms/doors over ordinary objects.

    Each item supplies id, point, size and priority. Returned rectangles include
    text padding, stay inside bounds and avoid both markers and other labels.
    """
    placed, missing = [], []
    limit = math.ceil(math.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1]))
    for item in sorted(items, key=lambda row: (row.get("priority", 1), row["id"])):
        x, y = item["point"]
        width, height = item["size"]
        rectangle = None
        for gap in range(9, limit, 12):
            # Side, top/bottom and diagonal candidates expand away from the anchor.
            candidates = [(x + gap, y - height / 2), (x - width - gap, y - height / 2),
                          (x - width / 2, y - height - gap), (x - width / 2, y + gap),
                          (x + gap, y - height - gap), (x + gap, y + gap),
                          (x - width - gap, y - height - gap), (x - width - gap, y + gap)]
            for left, top in candidates:
                rect = [round(left), round(top), round(left) + width, round(top) + height]
                if (rect[0] < bounds[0] or rect[1] < bounds[1]
                        or rect[2] > bounds[2] or rect[3] > bounds[3]):
                    continue
                if any(overlaps(rect, other, 3) for other in obstacles):
                    continue
                if any(overlaps(rect, other["rect"], 3) for other in placed):
                    continue
                rectangle = rect
                break
            if rectangle is not None:
                break
        if rectangle is None:
            missing.append(item["id"])
        else:
            end = [max(rectangle[0], min(x, rectangle[2])),
                   max(rectangle[1], min(y, rectangle[3]))]
            placed.append({**item, "rect": rectangle, "leader": [list(item["point"]), end]})
    return placed, missing
