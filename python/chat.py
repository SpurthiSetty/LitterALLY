"""Chat over the bin's own data.

Two kinds of question, per the architecture: what this bin has seen, and how to
dispose of a given thing. The first is answered from the event store, the
second from the rules file, and both are local by construction.

Backends, chosen with SMARTBIN_CHAT_BACKEND:

  offline  deterministic intent matching, no model at all. The default,
           because it runs on a laptop with nothing installed and makes the
           tools and the UI testable on their own. Also what the bin falls
           back to when the model is unavailable.
  local    the board's LLM brick, which does the tool calling itself.
  openai   any OpenAI-compatible endpoint, for developing against a real model
           on a laptop (Ollama, LM Studio) before touching the board.
"""

import os
import re

from chat_tools import ChatTools

BACKEND = os.environ.get("SMARTBIN_CHAT_BACKEND", "local")

SYSTEM_PROMPT = """You are the assistant built into a smart waste bin.

You can answer two kinds of question:
  1. What this bin has seen - use the tools to read its event log.
  2. How to dispose of something - use the tools to read its disposal rules.

Rules you must follow:
- Answer only from the tools. Never guess what the bin has seen.
- The log covers a limited period. Say so rather than implying a longer record.
- If a tool reports known=false, say plainly that this bin has no rule for that
  item. Do not invent one, and do not assume it is general rubbish.
- Be brief. One or two sentences unless asked for detail.
"""

# Used for the fallback, which runs without tools and so cannot see the log.
FALLBACK_PROMPT = """You are a helpful assistant built into a smart waste bin.

The bin's own rules did not cover this question, so answer it yourself.

- Answer disposal questions directly and practically. Say which bin or route
  you would use, and mention that local rules vary.
- Do not refuse ordinary questions about rubbish, and do not say you are unable
  to help - answering them is your job.
- The one thing you cannot do is look up what THIS bin has recorded. If asked
  what it has seen or counted, say you cannot check that. Never invent figures.
- Two sentences.
"""

# These answers are a sentence or two. The default of 512 lets the model run on
# long past the point of being useful, and generation time is linear in tokens.
MAX_TOKENS = int(os.environ.get("SMARTBIN_CHAT_MAX_TOKENS", "128"))


# Escalation. The rule is about privacy, not capability: the event log never
# leaves the device, so anything answerable from it stays local no matter how
# awkwardly it is phrased. Only questions the bin's own data cannot answer are
# candidates for a cloud model, and those carry the question alone.
def escalation_reason(question, results, understood=True):
    for result in results:
        if "by_category" in result or "items" in result:
            return None  # answered from the log; never leaves the device
    for result in results:
        if result.get("known") is False:
            return f"no local rule for {result.get('item', 'this item')!r}"
    if not results and understood:
        # A question we recognised as being about disposal but could not pin to
        # an item. Chatter that was never a question is not an escalation
        # candidate: there is nothing a cloud model would be answering.
        return "could not identify the item being asked about"
    return None


def _answered(results):
    """Did the router actually find something, as opposed to drawing a blank?

    A disposal lookup that came back known=false has not answered - the bin has
    no rule for that item - so it should fall through to the model rather than
    be reported as the final word.
    """
    for result in results:
        if result.get("total_items"):
            return True
        if result.get("items"):
            return True
        if result.get("known"):
            return True
    return False


_DISPOSAL = re.compile(
    r"\b(dispose|throw away|bin|recycle|compost|where does|which bin|what do i do with)\b",
    re.I,
)
_HISTORY = re.compile(r"\b(seen|thrown|threw|history|log|recent|last|how many|what have)\b", re.I)
_PERIOD = re.compile(r"\b(today|week|month|all|ever)\b", re.I)


def _subject(question):
    """Pull the thing being asked about out of a plain-English question."""
    text = re.sub(r"[?.!]", " ", question)
    text = re.sub(
        r"\b(how|do|i|dispose|of|a|an|the|my|this|that|where|does|go|goes|going|"
        r"should|put|which|bin|bins|for|what|to|into|in|on|with|is|are|am|can|"
        r"recycle|recycled|throw|thrown|throwing|away|out)\b",
        " ",
        text,
        flags=re.I,
    )
    return " ".join(text.split()).strip()


class Chat:
    def __init__(self, tools=None, backend=None):
        self.tools = tools or ChatTools()
        self.backend = (backend or BACKEND).lower()
        self._llm = None

    # -- offline backend ---------------------------------------------------

    def _offline(self, question):
        """Deterministic routing. No model, so it cannot invent anything."""
        results = []
        understood = False

        if _DISPOSAL.search(question):
            understood = True
            subject = _subject(question)
            if subject:
                results.append(self.tools.how_do_i_dispose_of(subject))
        elif _HISTORY.search(question):
            understood = True
            period = _PERIOD.search(question)
            results.append(
                self.tools.what_has_the_bin_seen(period.group(1) if period else "today")
            )

        return self._phrase(question, results), results, understood

    def _phrase(self, question, results):
        if not results:
            return (
                "I can tell you what this bin has thrown out, or which bin "
                "something belongs in. Try 'what have I thrown out this week' "
                "or 'how do I dispose of a carton'."
            )

        result = results[0]

        if "by_category" in result:
            if not result["total_items"]:
                return f"Nothing recorded for {result['period']}."
            parts = ", ".join(f"{n} {c}" for c, n in result["by_category"].items())
            first = (result["coverage"]["first_event"] or "")[:10]
            return (
                f"{result['total_items']} items {result['period']}: {parts}. "
                f"(Log starts {first}.)"
            )

        if result.get("known"):
            categories = result.get("categories", {})
            unique = sorted(set(categories.values()))
            if len(unique) == 1:
                return f"{result['item']} goes in {unique[0]}."
            listed = ", ".join(f"{lb} -> {ct}" for lb, ct in categories.items())
            return f"Depends which: {listed}."

        return (
            f"This bin has no rule for {result.get('item', 'that')}. "
            f"It sorts into: {', '.join(result.get('categories_available', []))}."
        )

    # -- model backends ----------------------------------------------------

    def _tool_list(self):
        """Wrap the tools for the LLM brick, which is LangChain underneath.

        Plain functions are rejected - the brick indexes them by a .name
        attribute that only a LangChain tool carries. The docstring becomes the
        description the model reads to choose between them, which is why those
        are written as instructions.
        """
        from langchain_core.tools import StructuredTool

        return [
            StructuredTool.from_function(
                func=method, name=method.__name__, description=method.__doc__
            )
            for method in (
                self.tools.what_has_the_bin_seen,
                self.tools.recent_items,
                self.tools.when_did_i_last_throw_out,
                self.tools.how_do_i_dispose_of,
            )
        ]

    def _model(self):
        """The fallback model, deliberately without tools.

        Binding tools cost two generation passes - one to choose a tool, one to
        write the answer - and each pass re-read every tool schema first. That
        was most of the 50 to 124 seconds. The model now only runs on questions
        the router already failed to answer using those same tools, so offering
        them again buys nothing and costs a whole pass.

        The prompt changes accordingly: with no tools it has no access to the
        bin's records, and must say so rather than invent figures. It once
        reported 20 items where the log held 303.
        """
        if self._llm is None:
            from arduino.app_bricks.llm import LargeLanguageModel

            self._llm = LargeLanguageModel(
                system_prompt=FALLBACK_PROMPT, max_tokens=MAX_TOKENS
            )
        return self._llm

    def _local(self, question):
        return self._model().chat(question), []

    def _local_stream(self, question):
        yield from self._model().chat_stream(question)

    def _openai(self, question):
        # Developing against a real model on a laptop: point at Ollama or
        # anything else speaking the OpenAI API. Tool calling is left to the
        # caller's model; without it this still exercises the prompt.
        from openai import OpenAI

        client = OpenAI(
            base_url=os.environ.get("SMARTBIN_LLM_URL", "http://localhost:11434/v1"),
            api_key=os.environ.get("SMARTBIN_LLM_KEY", "not-needed"),
        )
        reply = client.chat.completions.create(
            model=os.environ.get("SMARTBIN_LLM_MODEL", "qwen2.5:0.5b"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        return reply.choices[0].message.content, []

    # -- entry point -------------------------------------------------------

    def ask_stream(self, question):
        """Yield (kind, payload) as the answer is produced.

        Router answers arrive whole and instantly; model answers arrive a piece
        at a time. Total time is unchanged for the model, but words appearing
        after a few seconds reads as slow, whereas a blank screen for a minute
        reads as broken.
        """
        question = (question or "").strip()
        if not question:
            yield "meta", {"backend": "offline"}
            yield "text", "Ask me something about the bin."
            yield "done", {"escalate": None}
            return

        answer, results, understood = self._offline(question)

        if _answered(results) or self.backend not in ("local", "openai"):
            yield "meta", {"backend": "offline"}
            yield "text", answer
            yield "done", {
                "escalate": None if _answered(results)
                else escalation_reason(question, results, understood)
            }
            return

        yield "meta", {"backend": self.backend}
        produced = False
        try:
            for piece in self._local_stream(question):
                text = piece if isinstance(piece, str) else str(piece)
                if text:
                    produced = True
                    yield "text", text
        except Exception as exc:
            # The bin still answers when the model is unavailable: fall back to
            # whatever the router had to say rather than showing nothing.
            print(f"[chat] model unavailable ({exc})")
            if not produced:
                yield "text", answer
        yield "done", {"escalate": None}

    def ask(self, question):
        question = (question or "").strip()
        if not question:
            return {"answer": "Ask me something about the bin.", "escalate": None}

        # The router runs first whatever the backend, because when it can answer
        # it is both instant and incapable of misreporting a figure - the number
        # is read from the database and printed, with no step in between that
        # could corrupt it. Asked what had been thrown out today, the model said
        # 20 items where the log holds 303, after 124 seconds. That is the worst
        # kind of wrong: fluent, confident, and about the user's own data.
        answer, results, understood = self._offline(question)

        if _answered(results):
            return {
                "answer": answer,
                "backend": "offline",
                "escalate": None,
                "tool_results": results,
            }

        # Past here the router has nothing, so the model earns its latency:
        # unanticipated phrasings, and questions that need more than a lookup.
        if self.backend == "local":
            answer, _ = self._local(question)
        elif self.backend == "openai":
            answer, _ = self._openai(question)
        else:
            return {
                "answer": answer,
                "backend": "offline",
                "escalate": escalation_reason(question, results, understood),
                "tool_results": results,
            }

        return {
            "answer": answer,
            "backend": self.backend,
            # No escalation signal from a model backend: it does its own tool
            # calling, so there are no results here to judge. Reporting one
            # anyway meant every single answer claimed it would escalate.
            "escalate": None,
            "tool_results": [],
        }
