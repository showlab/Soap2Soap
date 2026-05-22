"""
FramePool — dependency-aware synchronous execution engine for Camera Tree mode.

Adapted from pai_v1_backend/app/model/utils/frame_pool.py (async → sync).

Key idea:
  1. Build a DAG where each shot knows:
       - depends_on_shots: [first shot in same group] (for shots 2+ in a group)
       - depends_on_anchors: [parent group IDs]    (for first shot of child groups)
  2. Topological sort the DAG
  3. Execute shots in that order, passing completed image paths as references
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any


@dataclass
class FrameUnit:
    shot_idx: int
    group_id: str
    group_desc: str = ""
    depth: int = 0

    # Dependencies
    depends_on_shots: List[int] = field(default_factory=list)
    """Index of the anchor (first) shot in the same group, for shots 2+."""

    depends_on_anchors: List[str] = field(default_factory=list)
    """Parent group IDs — first shot of a child group depends on parent anchors."""

    # Runtime
    status: str = "pending"
    result_path: Optional[str] = None


def build_frame_dependency_graph(
    groups: List[Dict[str, Any]],
) -> tuple[List[FrameUnit], Dict[str, int]]:
    """
    Build FrameUnit list and group_id → first_shot_idx mapping.

    Args:
        groups: list of {group_id, shot_indices, parent_group_ids, description}

    Returns:
        (frames, group_first_shot)
    """
    frames: List[FrameUnit] = []
    group_first_shot: Dict[str, int] = {}

    # Compute depths
    group_map = {g["group_id"]: g for g in groups}
    depth_cache: Dict[str, int] = {}

    def get_depth(gid: str) -> int:
        if gid in depth_cache:
            return depth_cache[gid]
        g = group_map.get(gid)
        if not g or not g.get("parent_group_ids"):
            depth_cache[gid] = 0
            return 0
        d = max(get_depth(pid) for pid in g["parent_group_ids"] if pid in group_map) + 1
        depth_cache[gid] = d
        return d

    for group in groups:
        gid = group["group_id"]
        shot_indices = group.get("shot_indices", [])
        parents = group.get("parent_group_ids", [])
        desc = group.get("description", "")
        depth = get_depth(gid)

        for i, shot_idx in enumerate(shot_indices):
            fu = FrameUnit(
                shot_idx=shot_idx,
                group_id=gid,
                group_desc=desc,
                depth=depth,
            )
            if i > 0:
                # shots 2+ in the group depend on the anchor (first shot)
                fu.depends_on_shots = [shot_indices[0]]
            elif i == 0 and parents:
                # first shot of child group depends on parent anchors
                fu.depends_on_anchors = list(parents)

            if i == 0:
                group_first_shot[gid] = shot_idx

            frames.append(fu)

    return frames, group_first_shot


def topological_sort(
    frames: List[FrameUnit],
    group_first_shot: Dict[str, int],
) -> List[FrameUnit]:
    """
    Return frames in an order where all dependencies come before dependents.
    """
    frame_map = {f.shot_idx: f for f in frames}
    visited: set = set()
    order: List[FrameUnit] = []

    def visit(f: FrameUnit):
        if f.shot_idx in visited:
            return
        visited.add(f.shot_idx)
        # Visit shot dependencies first
        for dep_idx in f.depends_on_shots:
            if dep_idx in frame_map:
                visit(frame_map[dep_idx])
        # Visit anchor dependencies first
        for gid in f.depends_on_anchors:
            anchor_idx = group_first_shot.get(gid)
            if anchor_idx is not None and anchor_idx in frame_map:
                visit(frame_map[anchor_idx])
        order.append(f)

    for f in frames:
        visit(f)

    return order


class FramePool:
    """
    Synchronous dependency-aware keyframe execution engine.

    Usage:
        frames, group_first_shot = build_frame_dependency_graph(groups)
        pool = FramePool(frames, group_first_shot)
        results = pool.run(generator_fn)

    generator_fn signature:
        (frame: FrameUnit, reference_paths: List[str]) -> Optional[str]
        Returns the saved image path on success, None on failure.
    """

    def __init__(self, frames: List[FrameUnit], group_first_shot: Dict[str, int]):
        self.frames = {f.shot_idx: f for f in frames}
        self.group_first_shot = group_first_shot
        self.results: Dict[int, str] = {}

    def run(
        self,
        generator: Callable[[FrameUnit, List[str]], Optional[str]],
    ) -> Dict[int, str]:
        """Execute all frames in dependency order."""
        sorted_frames = topological_sort(list(self.frames.values()), self.group_first_shot)

        for frame in sorted_frames:
            ref_paths = self._collect_references(frame)
            try:
                result = generator(frame, ref_paths)
                if result:
                    frame.result_path = result
                    frame.status = "done"
                    self.results[frame.shot_idx] = result
                else:
                    frame.status = "failed"
            except Exception as e:
                frame.status = "failed"
                print(f"  ❌ FramePool: shot {frame.shot_idx} failed: {e}")

        return self.results

    def _collect_references(self, frame: FrameUnit) -> List[str]:
        """Collect reference image paths for this frame."""
        refs = []
        # Previous shot in same group (anchor)
        for dep_idx in frame.depends_on_shots:
            path = self.results.get(dep_idx)
            if path:
                refs.append(path)
        # Parent group anchor
        for gid in frame.depends_on_anchors:
            anchor_idx = self.group_first_shot.get(gid)
            if anchor_idx is not None:
                path = self.results.get(anchor_idx)
                if path:
                    refs.append(path)
        return refs

    @property
    def stats(self) -> Dict[str, int]:
        counts: Dict[str, int] = {"done": 0, "failed": 0, "pending": 0}
        for f in self.frames.values():
            counts[f.status] = counts.get(f.status, 0) + 1
        return counts
