import re, hashlib
from pathlib import Path
from urllib.parse import urlparse, urljoin
from urllib.parse import ParseResult

import httpx
from playwright.async_api import async_playwright


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_HTML = DATA_DIR / "raw" / "html"
RAW_PDF = DATA_DIR / "raw" / "pdf"

for p in (RAW_HTML, RAW_PDF):
    p.mkdir(parents=True, exist_ok=True)


async def scrape_page(url: str, same_domain_only: bool = False):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        browser_page = await browser.new_page()
        await browser_page.goto(url, wait_until="domcontentloaded", timeout=90000)

        title = await browser_page.title()
        html = await browser_page.content()

        file_name = slugify(urlparse(url).path or "index") + ".html"
        file_path = RAW_HTML / file_name
        file_path.write_text(html, encoding="utf-8")

        valid_links_debug = await browser_page.eval_on_selector_all(
            "main a[href]",
            """
            els => els.map(e => ({href: e.href, html: e.outerHTML.substring(0,200)}))
            """
        )
        valid_links = [x["href"] for x in valid_links_debug]

        print(f"[DEBUG] Valid links ({len(valid_links)}):")
        for l in valid_links_debug:
            print(f"   [VALID] {l['href']} -- from: {l['html']}")

        valid_abs_links = normalize_links(valid_links, url, same_domain_only)

        return title, file_path, valid_abs_links
    

def normalize_links(links, url, same_domain_only):
    origin = urlparse(url).netloc
    seen, abs_links = set(), []

    for l in links:
        if l:
            try:
                u = urlparse(l)
                if not u.scheme:
                    l = urljoin(url, l)
                    u = urlparse(l)
                if (not same_domain_only) or (u.netloc and u.netloc == origin):
                    if l not in seen:
                        seen.add(l)
                        abs_links.append(l)
            except Exception as e:
                print(f"[WARN] Link scartato {l}: {e}")
    return abs_links
    

async def download_pdfs(urls: list[str]) -> list[Path]:
    saved = []

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf",
        "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://www.unipg.it/",
    }

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=90.0,
        headers=headers
    ) as client:
        for u in urls:
            try:
                parsed_url = urlparse(u)
                file_path = RAW_PDF / assign_pdf_file_name(parsed_url, u)

                if file_path.exists():
                    saved.append(file_path)
                    continue

                r = await client.get(u)
                r.raise_for_status()

                content_type = r.headers.get('content-type', '').lower()
                if 'application/pdf' not in content_type and not r.content.startswith(b'%PDF'):
                    print(f"[WARN] Il contenuto di {u} non sembra essere un PDF")
                    continue

                file_path.write_bytes(r.content)
                saved.append(file_path)
                print(f"[OK] Scaricato PDF: {u} -> {file_path.name}")

            except Exception as e:
                print(f"[WARN] PDF skip {u}: {e}")

    return saved


def assign_pdf_file_name(parsed_url: ParseResult, u: str) -> str:
    if parsed_url.query:
        file_name = slugify(parsed_url.path + "_" + parsed_url.query[:50]) or sha(u)[:12]
    else:
        file_name = slugify(parsed_url.path) or sha(u)[:12]

    if not file_name.lower().endswith(".pdf"):
        file_name += ".pdf"

    return file_name


def slugify(s: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", s.strip())
    return s[:120] if s else "file"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()