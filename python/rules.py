from pathlib import Path

import yaml

UNKNOWN = "unknown"

_RULES_FILE = Path(__file__).with_name("disposal_rules.yaml")


class Rules:
    def __init__(self, path=_RULES_FILE):
        doc = yaml.safe_load(Path(path).read_text())
        self.min_confidence = float(doc["min_confidence"])
        self.categories = tuple(doc["categories"])
        self._by_label = {
            label: category
            for category, spec in doc["categories"].items()
            for label in (spec.get("labels") or [])
        }

    def resolve(self, label, confidence):
        if confidence < self.min_confidence:
            return UNKNOWN
        return self._by_label.get(label, UNKNOWN)
