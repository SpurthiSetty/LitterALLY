"""What the chatbot is allowed to know.

Each function here is registered as a tool the language model can call. They
are deliberately plain Python over the event store and the rules file - no
Arduino imports - so the whole set runs and can be tested on a laptop against a
copy of the database.

Docstrings are not decoration: the model reads them to decide which tool to
call and with what arguments. Keep them concrete.
"""

from datetime import datetime, timedelta, timezone

from rules import Rules
from store import EventStore

# Recognised values for the `period` argument, and how far back each reaches.
_PERIODS = {
    "today": timedelta(days=1),
    "day": timedelta(days=1),
    "24 hours": timedelta(days=1),
    "week": timedelta(days=7),
    "month": timedelta(days=30),
    "all": None,
    "ever": None,
}


class ChatTools:
    def __init__(self, store=None, rules=None):
        self._store = store or EventStore()
        self._rules = rules or Rules()

    # -- helpers ----------------------------------------------------------

    def _since(self, period):
        window = _PERIODS.get((period or "all").strip().lower(), timedelta(days=1))
        return None if window is None else datetime.now(timezone.utc) - window

    def _coverage(self):
        """How far back the log actually goes.

        Every answer carries this. A bin running for a day should not imply it
        has been watching for months when it says it has never seen a battery.
        """
        first, last = self._store.span()
        return {"first_event": first, "last_event": last}

    # -- tools ------------------------------------------------------------

    def what_has_the_bin_seen(self, period: str = "today") -> dict:
        """Summarise what has been thrown away, grouped by disposal category.

        period: one of "today", "week", "month", "all". Defaults to today.
        Use for questions like "what have I thrown out this week" or "how much
        of my rubbish is recyclable".
        """
        since = self._since(period)
        return {
            "period": period,
            "total_items": self._store.total(since=since),
            "by_category": self._store.counts_by_category(since=since),
            "most_common": self._store.counts_by_label(since=since, limit=5),
            "coverage": self._coverage(),
        }

    def recent_items(self, count: int = 10) -> dict:
        """List the most recent individual items, newest first.

        Use for "what was the last thing I threw out" or "show me recent items".
        """
        count = max(1, min(int(count), 50))
        return {
            "items": self._store.recent(limit=count),
            "coverage": self._coverage(),
        }

    def when_did_i_last_throw_out(self, item: str) -> dict:
        """When a particular item was last seen, if ever.

        item: an everyday name such as "banana" or "bottle".
        Use for "have I thrown out any batteries recently".
        """
        matches = self._rules.find_labels(item)
        if not matches:
            return {
                "item": item,
                "known": False,
                "note": "not something this bin can recognise",
                "coverage": self._coverage(),
            }

        seen = {label: self._store.last_seen(label) for label in matches}
        seen = {label: ts for label, ts in seen.items() if ts}
        return {
            "item": item,
            "known": True,
            "matched_labels": matches,
            "last_seen": seen or None,
            "coverage": self._coverage(),
        }

    def how_do_i_dispose_of(self, item: str) -> dict:
        """Which bin an item belongs in, according to this bin's disposal rules.

        item: an everyday name such as "cardboard", "banana peel", "phone".
        Returns known=False when the item is outside the configured rules, which
        means the bin genuinely has no policy for it - not that it is rubbish.
        """
        matches = self._rules.find_labels(item)
        if not matches:
            return {
                "item": item,
                "known": False,
                "note": "no disposal rule configured for this item",
                "categories_available": list(self._rules.categories),
            }

        return {
            "item": item,
            "known": True,
            "categories": {
                label: self._rules.category_for(label) for label in matches
            },
        }
