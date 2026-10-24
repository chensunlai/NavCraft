"""Serve a generated public viewer directory without access to run internals."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


class PublicArtifactHandler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".json": "application/json", ".js": "text/javascript",
                      ".css": "text/css"}

    def send_head(self):
        self._artifact = None
        root = Path(self.directory).resolve()
        try:
            request_path = unquote(urlsplit(self.path).path, errors="strict")
            parts = request_path.split("/")
            if "\\" in request_path or "\0" in request_path or any(part.startswith(".") for part in parts):
                raise ValueError("Hidden paths and traversal are not public artifacts")
            target = root.joinpath(*(part for part in parts if part)).resolve()
            if target.is_dir():
                target = (target / "index.html").resolve()
            relative = target.relative_to(root)
            if any(part.startswith(".") for part in relative.parts) or not target.is_file():
                raise ValueError("Not a public artifact")
        except (ValueError, OSError):
            self.send_error(404, "Public artifact not found")
            return None
        self._artifact = target
        return super().send_head()

    def translate_path(self, path):
        return str(self._artifact)

    def list_directory(self, path):
        self.send_error(404, "Directory listing is disabled")
        return None

    def end_headers(self):
        artifact = getattr(self, "_artifact", None)
        self.send_header("Cache-Control", "no-store" if artifact is None or artifact.suffix.lower() in {
            ".html", ".htm", ".json"} else "public, max-age=60")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()


def create_server(directory, host="0.0.0.0", port=8765):
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError(f"Public viewer directory does not exist: {root}")
    return ThreadingHTTPServer((host, port), partial(PublicArtifactHandler, directory=str(root)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    with create_server(args.directory, args.host, args.port) as server:
        print(f"Serving {args.directory.resolve()} at http://{args.host}:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
