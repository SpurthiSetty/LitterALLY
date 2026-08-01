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

    def category_for(self, label):
        """Category for a label, ignoring confidence. Returns None if unknown.

        Separate from resolve() on purpose: resolve answers "what did the
        camera just see", where a weak guess must be rejected. This answers
        "which bin does a banana go in", where there is no guess to doubt.
        """
        return self._by_label.get(label)

    def find_labels(self, query):
        """Labels matching a phrase, for questions asked in human words.

        Someone asks about a "bottle"; the vocabulary holds "pop bottle" and
        "water bottle". Exact matches come first so an exact word is never
        buried under the things that merely contain it.
        """
        needle = query.strip().lower()
        if not needle:
            return []

        exact = [label for label in self._by_label if label.lower() == needle]
        partial = [
            label
            for label in self._by_label
            if label.lower() != needle
            and (needle in label.lower() or label.lower() in needle)
        ]
        return exact + sorted(partial)

    def all_labels(self):
        return sorted(self._by_label)
