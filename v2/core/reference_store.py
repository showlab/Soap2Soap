"""
ReferenceStore — entity-to-image mapping for cross-shot consistency.
Ported and simplified from pai_v1_backend/app/model/schema/reference_store.py
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ReferenceEntity:
    entity_id: str        # "@character_01"
    entity_type: str      # "character" | "environment"
    description: str      # full visual description
    image_index: int = 0  # position order in API calls
    image_path: Optional[str] = None  # local path to generated image

    def get_image_source(self) -> Optional[str]:
        return self.image_path


class ReferenceStore:
    """Maps entity tokens to generated reference images."""

    def __init__(self, scene_id: str = "scene"):
        self.scene_id = scene_id
        self.entities: Dict[str, ReferenceEntity] = {}

    def add_entity(self, entity: ReferenceEntity):
        self.entities[entity.entity_id] = entity

    def get_entity(self, entity_id: str) -> Optional[ReferenceEntity]:
        return self.entities.get(entity_id)

    def get_entity_name(self, entity_id: str) -> str:
        """Extract short display name from entity description."""
        entity = self.entities.get(entity_id)
        if not entity:
            return "unknown"
        if entity.entity_type == "character":
            match = re.search(r"Name:\s*([^\n,\.]+)", entity.description)
            if match:
                return match.group(1).strip()
        return "character"

    def get_ordered_images(self) -> List[str]:
        """Return image paths in image_index order, skipping missing ones."""
        sorted_e = sorted(self.entities.values(), key=lambda e: e.image_index)
        return [e.image_path for e in sorted_e if e.image_path]

    def has_image(self, entity_id: str) -> bool:
        e = self.entities.get(entity_id)
        return bool(e and e.image_path)

    def __len__(self) -> int:
        return len(self.entities)

    def __repr__(self) -> str:
        return f"ReferenceStore(entities={list(self.entities.keys())})"
