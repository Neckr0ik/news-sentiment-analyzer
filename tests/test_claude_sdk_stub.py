"""Exercise the real anthropic SDK against a local stub server (no internet, no key)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import anthropic

from src.sentiment import DEFAULT_CLAUDE_MODEL, ClaudeBackend


class _Handler(BaseHTTPRequestHandler):
    received = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        _Handler.received.append((self.path, body))
        items = json.loads(body["messages"][0]["content"].split("Items:\n", 1)[1])
        text = json.dumps({"results": [{"id": it["id"], "score": -0.8} for it in items]})
        reply = json.dumps({
            "id": "msg_test", "type": "message", "role": "assistant", "model": body["model"],
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 10},
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args):
        pass


def test_real_sdk_request_shape():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = anthropic.Anthropic(api_key="test-key", base_url=f"http://127.0.0.1:{server.server_port}",
                                     max_retries=0)
        scores = ClaudeBackend(client=client).score_batch(["Huracán deja muertos", "Enchentes"], "es")
    finally:
        server.shutdown()
    assert scores == [-0.8, -0.8]
    path, body = _Handler.received[-1]
    assert path == "/v1/messages"
    assert body["model"] == DEFAULT_CLAUDE_MODEL
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["output_config"]["effort"] == "low"
