import base64
from http.client import HTTPConnection
import threading

import pytest

from src.viewer.server import create_server


@pytest.fixture
def public_server(tmp_path):
    root = tmp_path / "public"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text('<script src="assets/viewer.js"></script>')
    (root / "data.json").write_text('{"status":"completed"}')
    (root / "assets/viewer.js").write_text("console.log('viewer');")
    (root / "assets/viewer.css").write_text("body { color: #111; }")
    (root / "assets/frame.png").write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aB5cAAAAASUVORK5CYII="))
    (root / ".secret.json").write_text('{"private":true}')
    (tmp_path / "outside.json").write_text('{"private":true}')
    (root / "external.json").symlink_to(tmp_path / "outside.json")
    (root / "external-dir").symlink_to(tmp_path, target_is_directory=True)
    (root / "hidden-alias.json").symlink_to(root / ".secret.json")
    server = create_server(root, "127.0.0.1", 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def request(server, path, method="GET"):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


@pytest.mark.parametrize("path,mime,cache", [
    ("/", "text/html", "no-store"),
    ("/data.json?refresh=1", "application/json", "no-store"),
    ("/assets/viewer.js", "text/javascript", "public, max-age=60"),
    ("/assets/viewer.css", "text/css", "public, max-age=60"),
    ("/assets/frame.png", "image/png", "public, max-age=60"),
])
def test_public_artifacts_have_correct_types_and_cache_policy(public_server, path, mime, cache):
    status, headers, body = request(public_server, path)
    assert status == 200
    assert headers["Content-type"] == mime
    assert headers["Cache-Control"] == cache
    assert body


@pytest.mark.parametrize("path", [
    "/missing.json", "/assets/", "/assets", "/.secret.json", "/%2esecret.json",
    "/../outside.json", "/%2e%2e/outside.json", "/assets/../../outside.json",
    "/assets%2f..%2f..%2foutside.json", "/external.json", "/external-dir/outside.json",
    "/hidden-alias.json", "/assets\\..\\outside.json",
])
def test_missing_directory_hidden_and_escaping_paths_are_unavailable(public_server, path):
    status, _, body = request(public_server, path)
    assert status == 404
    assert b'"private":true' not in body
    assert b"Directory listing" not in body


def test_head_works_without_returning_the_image(public_server):
    status, headers, body = request(public_server, "/assets/frame.png", "HEAD")
    assert status == 200 and headers["Content-type"] == "image/png"
    assert int(headers["Content-Length"]) > 0 and body == b""


def test_server_requires_an_existing_public_directory(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        create_server(tmp_path / "missing", "127.0.0.1", 0)
