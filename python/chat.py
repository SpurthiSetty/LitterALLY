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
import threading

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

# The cloud model gets the tools, so it can read the log - but it is reached
# only for questions the router could not answer, which are mostly items the
# bin has no rule for. Given SYSTEM_PROMPT it treated the five categories as
# the only possible answers and put a mattress in recycle. A mattress does not
# go in a household bin at all, and saying so is the useful answer.
CLOUD_PROMPT = """You are the assistant built into a smart waste bin.

Use the tools to answer questions about what this bin has recorded and about
its own disposal rules. Report what a tool returns exactly - never adjust or
round a figure, and never state a count the tools did not give you.

When a tool reports known=false, this bin has no rule for that item. Say so,
then give genuinely useful real-world advice. Plenty of things - mattresses,
furniture, appliances, paint, medicines - belong in none of this bin's
categories and need bulky waste collection, a recycling centre, or a take-back
scheme. Never force an item into one of the five categories just because they
are the options on offer; recommending the wrong bin is worse than saying the
bin cannot take it.

Local collection rules vary, so say so where it matters. Two or three
sentences.
"""

# Used for the fallback, which runs without tools and so cannot see the log.
FALLBACK_PROMPT = """You are a helpful assistant built into a smart waste bin.

The bin's own rules did not cover this question, so answer it yourself.

- This bin sorts into exactly five categories: recycle, compost, trash,
  hazardous, ewaste. Those are the only ones that exist.
- NEVER refer to bins by position, number, colour or lid - there is no "top
  bin", "third bin" or "bin with the lid closed". Inventing them is the single
  worst thing you can do here.
- If the item does not clearly belong to one of the five, say the bin has no
  rule for it and suggest checking local council collection. That is a good
  answer, not a failure.
- You cannot look up what this bin has recorded. If asked what it has seen or
  counted, say you cannot check that. Never invent figures.
- Two sentences, plain language.
"""

# These answers are a sentence or two. The default of 512 lets the model run on
# long past the point of being useful, and generation time is linear in tokens.
MAX_TOKENS = int(os.environ.get("SMARTBIN_CHAT_MAX_TOKENS", "128"))

# Cloud. Off unless a key exists AND the caller allows it for that question:
# leaving the device is a decision the user makes, not a consequence of the
# local model struggling.
CLOUD_KEY = os.environ.get("SMARTBIN_CLOUD_KEY", os.environ.get("API_KEY", ""))

# Left empty so the brick chooses, which means Anthropic Claude. Naming a model
# here only pins it to whatever was current when this was written - the first
# attempt hardcoded claude-sonnet-4-5, which was already stale. Set this to
# override: the brick also understands openai: and google: prefixes.
CLOUD_MODEL = os.environ.get("SMARTBIN_CLOUD_MODEL", "")


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
        self._cloud_llm = None
        # The brick refuses a second stream while one is running - "a streaming
        # response is already in progress" - and the server is threaded, so two
        # browser tabs or an overlapping request were enough to fail both and
        # take the app down. One model call at a time.
        self._model_lock = threading.Lock()

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

    def cloud_available(self):
        """Whether a cloud answer is even possible - a key has to exist."""
        return bool(CLOUD_KEY)

    def _cloud(self):
        """The cloud model, which does get tools.

        Unlike the local fallback, a capable model can be trusted to read a
        tool result and repeat it faithfully - that is what the 0.8B could not
        do when it reported 20 items against 303. So the cloud path can answer
        questions about the log that the router's patterns did not match,
        rather than being limited to general knowledge.
        """
        if self._cloud_llm is None:
            from arduino.app_bricks.cloud_llm import CloudLLM

            options = {
                "api_key": CLOUD_KEY,
                "system_prompt": CLOUD_PROMPT,
                "tools": self._tool_list(),
            }
            if CLOUD_MODEL:
                options["model"] = CLOUD_MODEL

            self._cloud_llm = CloudLLM(**options)
        return self._cloud_llm

    def _cloud_stream(self, question):
        yield from self._cloud().chat_stream(question)

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

    def ask_stream(self, question, allow_cloud=False):
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

        # Cloud and local are alternatives, not escalating tiers. If the cloud
        # is permitted and configured it answers directly: running the 0.8B
        # first would add tens of seconds to arrive at a worse answer. Local is
        # what remains when the toggle is off, the key is missing, or the
        # network is down - so the bin always answers something.
        use_cloud = allow_cloud and self.cloud_available()
        stream = self._cloud_stream if use_cloud else self._local_stream
        chosen = "cloud" if use_cloud else self.backend

        if not self._model_lock.acquire(timeout=2):
            yield "meta", {"backend": "offline"}
            yield "text", ("The model is answering someone else's question. "
                           "Try again in a moment - or ask something the bin "
                           "can look up directly, which never waits.")
            yield "done", {"escalate": None}
            return

        yield "meta", {"backend": chosen}
        produced = False
        try:
            for piece in stream(question):
                text = piece if isinstance(piece, str) else str(piece)
                if text:
                    produced = True
                    yield "text", text
        except Exception as exc:
            print(f"[chat] {chosen} unavailable ({exc})")
            if not produced and use_cloud:
                # Reaching the cloud failed; the local model is still here.
                try:
                    for piece in self._local_stream(question):
                        text = piece if isinstance(piece, str) else str(piece)
                        if text:
                            produced = True
                            yield "text", text
                except Exception as inner:
                    print(f"[chat] local unavailable too ({inner})")
            if not produced:
                # Nothing answered, so say what the router had rather than
                # showing a blank reply.
                yield "text", answer
        finally:
            self._model_lock.release()
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
