"""HTTP smoke test for the generated static website, when present."""
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "data" / "website_factory" / "site"
PAGES = [
    "index.html",
    "services.html",
    "about.html",
    "projects.html",
    "blog.html",
    "faq.html",
    "contact.html",
]


def main():
    if not SITE.is_dir():
        print("No generated site package; HTTP smoke test skipped.")
        return

    server = subprocess.Popen(
        ["python", "-m", "http.server", "8765", "--directory", str(SITE)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(40):
            if server.poll() is not None:
                raise RuntimeError("Static HTTP server exited before becoming ready")
            try:
                urllib.request.urlopen("http://127.0.0.1:8765/", timeout=1).close()
                break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.25)
        else:
            raise RuntimeError("Static HTTP server did not become ready")

        for page in PAGES:
            with urllib.request.urlopen(
                "http://127.0.0.1:8765/" + page, timeout=10
            ) as response:
                assert response.status == 200, (page, response.status)
                body = response.read().decode("utf-8")
                assert "<html" in body.lower(), page
                assert "<title>" in body.lower(), page
                assert 'rel="canonical"' in body, page
                assert 'class="skip-link"' in body, page
        print("Static HTTP smoke test passed for 7 pages.")
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)


if __name__ == "__main__":
    main()
