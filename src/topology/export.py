"""Writing a topology out in the two formats the pipeline needs.

The tool and the engine want different files, and the difference is not cosmetic.

``write_tool_json`` produces what ``SphereManager.load_from_json`` reads: version 2, every
sphere carrying its LBS binding, and connections of 2 or 3 indices only, because that is
all the loader accepts. Quads are split into two triangles here so the file opens.

``write_engine_text`` produces the SMPL20-style text ``scripts/json2txt.py`` emits, with
quads kept whole in the fourth count slot. Engine-side a quad is one prysmoid-like bumper
rather than two, which is why it is worth keeping.

The two therefore disagree about the triangle count on purpose, and the JSON records the
original grouping under ``quad_groups`` so the split can be undone rather than guessed at.

The bindings written into the JSON reproduce ``SphereManager.assign_parent_joint`` exactly
-- closest kintree bone, bound to that bone's parent, secondary joint at the child end,
blend weight from the projection along the bone. Writing them here rather than letting the
tool recompute them on load is what makes a generated template reproducible: the tool only
auto-assigns when ``parent_joint`` is missing, so a file that carries its own bindings
survives a round trip through the GUI unchanged.
"""

from __future__ import annotations

import json
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .anatomy import Body, assign_parent_joint_like_tool

__all__ = [
    "compute_bindings",
    "build_payload",
    "write_tool_json",
    "write_engine_text",
    "split_quads",
    "merge_quads",
]

# Keep in sync with src/sphere_manager.py JSON_VERSION.
JSON_VERSION = 2


def split_quads(connections: Iterable[Sequence[int]]) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """Split every 4-connection across its 0-2 diagonal.

    Returns ``(connections_for_the_tool, original_quads)``. The diagonal matches
    ``Bumper.triangles`` so the split geometry is identical to the quad's.
    """
    tool_connections: list[tuple[int, ...]] = []
    quads: list[tuple[int, ...]] = []
    for connection in connections:
        connection = tuple(int(index) for index in connection)
        if len(connection) == 4:
            first, second, third, fourth = connection
            quads.append(connection)
            tool_connections.append((first, second, third))
            tool_connections.append((first, third, fourth))
        else:
            tool_connections.append(connection)
    return tool_connections, quads


def merge_quads(
    connections: Iterable[Sequence[int]], quad_groups: Iterable[Sequence[int]]
) -> list[tuple[int, ...]]:
    """Inverse of ``split_quads``: fold ``(a,b,c),(a,c,d)`` back into ``(a,b,c,d)``.

    Each quad takes the position of the earlier of its two triangles, so a connection
    list that went through ``split_quads`` comes back in its original order. A quad whose
    two split triangles are not both present is skipped with a printed warning and its
    triangles are kept, so a file the tool has edited still converts. The same rule is
    implemented standalone in ``scripts/json2txt.py``, which cannot import this package.
    """
    rows = [tuple(int(index) for index in connection) for connection in connections]
    consumed = [False] * len(rows)
    quad_at_position: dict[int, tuple[int, ...]] = {}

    def find_unconsumed(triangle: tuple[int, ...]) -> int | None:
        for position, row in enumerate(rows):
            if not consumed[position] and row == triangle:
                return position
        return None

    for quad in quad_groups:
        first, second, third, fourth = (int(index) for index in quad)
        first_position = find_unconsumed((first, second, third))
        second_position = find_unconsumed((first, third, fourth))
        if first_position is None or second_position is None:
            print(
                f"[warn] quad {(first, second, third, fourth)}: split triangles not both "
                "present in the connection list; keeping the triangles"
            )
            continue
        consumed[first_position] = True
        consumed[second_position] = True
        quad_at_position[min(first_position, second_position)] = (first, second, third, fourth)

    merged: list[tuple[int, ...]] = []
    for position, row in enumerate(rows):
        if position in quad_at_position:
            merged.append(quad_at_position[position])
        if not consumed[position]:
            merged.append(row)
    return merged


def compute_bindings(body: Body, centers: np.ndarray, radii: np.ndarray) -> list[dict]:
    """Per-sphere LBS binding and radius scaling, exactly as the tool computes them.

    ``rest_offset`` is the sphere's position in the parent joint's rest frame; in T-pose
    every global joint rotation is the identity, so it reduces to the plain offset, but the
    rotation is applied anyway so a template authored against a posed reference is right
    too.

    ``rest_radius`` is the *ratio* of the sphere's radius to the distance to the nearest
    body vertex, which is what ``SphereManager.update_sphere_radius`` multiplies back when
    the body shape changes. A ratio of 1 means the sphere touches the skin exactly.
    """
    centers = np.atleast_2d(np.asarray(centers, dtype=np.float64))
    radii = np.asarray(radii, dtype=np.float64)
    parent_joint, child_joint, _ = assign_parent_joint_like_tool(
        centers, body.joints, body.parents
    )

    bindings: list[dict] = []
    for index, (center, radius) in enumerate(zip(centers, radii)):
        parent = int(parent_joint[index])
        child = int(child_joint[index])

        parent_rotation = body.rotations[parent]
        rest_offset = parent_rotation.T @ (center - body.joints[parent])

        child_rotation = body.rotations[child]
        secondary_rest_offset = child_rotation.T @ (center - body.joints[child])

        bone = body.joints[child] - body.joints[parent]
        bone_length_squared = float(bone @ bone)
        along = (
            float(np.clip((center - body.joints[parent]) @ bone / bone_length_squared, 0.0, 1.0))
            if bone_length_squared > 1e-12 else 0.0
        )

        distance_to_vertices = np.linalg.norm(body.vertices - center, axis=1)
        nearest_vertex = int(np.argmin(distance_to_vertices))
        touch_radius = float(distance_to_vertices[nearest_vertex])

        bindings.append({
            "parent_joint": parent,
            "rest_offset": [float(value) for value in rest_offset],
            "secondary_joint": child,
            "secondary_rest_offset": [float(value) for value in secondary_rest_offset],
            "lbs_weight": 1.0 - along,
            "rest_radius": float(radius / touch_radius) if touch_radius > 1e-12 else 1.0,
            "nearest_vertex_index": nearest_vertex,
        })
    return bindings


def build_payload(
    body: Body,
    centers: np.ndarray,
    radii: np.ndarray,
    connections: Iterable[Sequence[int]],
    *,
    locked: Sequence[bool] | None = None,
) -> dict:
    """The dict ``SphereManager.load_from_json`` expects, plus quad provenance."""
    centers = np.atleast_2d(np.asarray(centers, dtype=np.float64))
    radii = np.asarray(radii, dtype=np.float64)
    tool_connections, quads = split_quads(connections)
    bindings = compute_bindings(body, centers, radii)

    spheres = []
    for index, (center, radius, binding) in enumerate(zip(centers, radii, bindings)):
        spheres.append({
            "center": [float(value) for value in center],
            "radius": float(radius),
            **binding,
            "locked": bool(locked[index]) if locked is not None else False,
        })

    payload = {
        "version": JSON_VERSION,
        "spheres": spheres,
        "connections": [{"indices": list(connection)} for connection in tool_connections],
    }
    if quads:
        payload["quad_groups"] = [list(quad) for quad in quads]
    return payload


def write_tool_json(path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _truncate(value: float, decimals: int = 4) -> float:
    """Truncate, not round -- matching ``scripts/json2txt.py`` exactly.

    Done on the shortest round-trip decimal representation rather than through
    ``math.trunc(value * 10**decimals)``, whose multiplication can land one ulp below an
    integer (``0.0192 * 10000 == 191.99999999999997``) and truncate a unit low. The engine
    text must survive txt -> json -> txt unchanged, so the digits have to be exact.
    """
    quantum = Decimal(1).scaleb(-decimals)
    truncated = Decimal(repr(float(value))).quantize(quantum, rounding=ROUND_DOWN)
    # ``+ 0.0`` turns a truncated ``-0.0000`` into ``0.0`` so the sign never prints.
    return float(truncated) + 0.0


def write_engine_text(
    path,
    centers: np.ndarray,
    radii: np.ndarray,
    connections: Iterable[Sequence[int]],
    *,
    title: str = "Sphere Mesh 2.0",
) -> Path:
    """Write the SMPL20-style text, quads intact.

    Layout, matching ``scripts/json2txt.py``: title; a counts line of
    ``spheres triples doubles quads``; one ``x y z radius`` line per sphere; then the
    connections grouped triples, doubles, quads in that order.
    """
    centers = np.atleast_2d(np.asarray(centers, dtype=np.float64))
    radii = np.asarray(radii, dtype=np.float64)
    grouped = {2: [], 3: [], 4: []}
    for connection in connections:
        connection = tuple(int(index) for index in connection)
        if len(connection) not in grouped:
            raise ValueError(f"connection {connection} has an unsupported arity")
        if len(set(connection)) != len(connection):
            raise ValueError(f"connection {connection} repeats a sphere")
        if min(connection) < 0 or max(connection) >= len(centers):
            raise ValueError(f"connection {connection} references a sphere that does not exist")
        grouped[len(connection)].append(connection)

    lines = [
        title,
        f"{len(centers)} {len(grouped[3])} {len(grouped[2])} {len(grouped[4])}",
    ]
    for center, radius in zip(centers, radii):
        lines.append(
            f"{_truncate(center[0]):.4f} {_truncate(center[1]):.4f} "
            f"{_truncate(center[2]):.4f} {radius}"
        )
    for arity in (3, 2, 4):
        for connection in grouped[arity]:
            lines.append(" ".join(str(index) for index in connection))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
