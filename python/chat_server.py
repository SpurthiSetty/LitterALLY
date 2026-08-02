"""Laptop development server for the chatbot.

Standard library only, so it runs on a laptop with nothing installed but the
rules parser. This is not what ships on the board - there the web_ui brick
serves the page and the LLM brick answers - but it exercises the same
ChatTools and Chat code, so the parts worth getting right are shared.

    python3 chat_server.py [--db path/to/smartbin.db] [--port 8090]

Then open http://localhost:8090
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from chat import Chat
from chat_tools import ChatTools
from rules import Rules
from store import EventStore

_UI = Path(__file__).with_name("chat_ui.html")


class Handler(BaseHTTPRequestHandler):
    chat = None

    def _send(self, code, body, content_type="application/json"):
        payload = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.startswith("/api/summary"):
            tools = self.chat.tools
            self._send(200, json.dumps(tools.what_has_the_bin_seen("all")))
        else:
            self._send(200, _UI.read_bytes(), "text/html; charset=utf-8")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            question = json.loads(self.rfile.read(length) or b"{}").get("question", "")
        except json.JSONDecodeError:
            self._send(400, json.dumps({"error": "bad json"}))
            return

        try:
            self._send(200, json.dumps(self.chat.ask(question), default=str))
        except Exception as exc:  # a broken tool must not kill the server
            self._send(500, json.dumps({"error": str(exc)}))

    def log_message(self, *args):
        pass  # the default logger writes a line per request to stderr


def serve(db=None, rules=None, port=8090, backend=None):
    """Serve the chat UI. Blocks, so the orchestrator runs it on a thread.

    Its own EventStore rather than the orchestrator's: the chat path is meant
    to share nothing with the real-time path except the database file, and
    SQLite is happy with a second reader.
    """
    store = EventStore(db) if db else EventStore()
    rule_set = Rules(rules) if rules else Rules()
    Handler.chat = Chat(tools=ChatTools(store=store, rules=rule_set), backend=backend)

    print(f"[chat] backend {Handler.chat.backend}, {store.total()} events, port {port}")

    # Threading, because a local model takes the better part of a minute to
    # answer and a single-threaded server would refuse to serve the page - or
    # anything else - for the whole of it.
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=None, help="path to smartbin.db")
    parser.add_argument("--rules", default=None, help="path to disposal_rules.yaml")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    serve(db=args.db, rules=args.rules, port=args.port)


if __name__ == "__main__":
    main()
