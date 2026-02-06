from http.server import BaseHTTPRequestHandler, HTTPServer
from utils.metrics import get_metrics

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404); self.end_headers(); return
        body = get_metrics().to_prometheus().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

def serve(port: int = 8000):
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()