import asyncio
import httpx

from scraping import (
    download_pdfs,
    scrape_page
)

from parsing import (
    to_documents_from_html,
    to_documents_from_pdf
)


async def crawl(seed_url: str, max_depth, is_download_pdf_active):
    visited = set()
    to_visit = [(seed_url, 0)]
    all_docs = []
    
    while to_visit:
        url, depth = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        print(f"\n[INFO] Crawling ({depth}/{max_depth}): {url}")
        try:
            title, html_file_path, links = await scrape_page(url)
            print(f"[INFO] Controllo Content-Type per {len(links)} link validi...")
            pdf_links, html_links = await categorize_links(links)
            print(f"[DEBUG] Link trovati nel <main> della pagina: {url}")
            print(f"  - PDF: {len(pdf_links)}")
            for l in pdf_links:
                print(f"    [PDF] {l}")
            print(f"  - HTML: {len(html_links)}")
            for l in html_links:
                print(f"    [HTML] {l}")

        except Exception as e:
            print(f"[WARN] Skip {url}: {e}")
            continue

        html_docs = to_documents_from_html(html_file_path, source_url=url, page_title=title)
        all_docs.extend(html_docs)

        if is_download_pdf_active:
            pdf_files = await download_pdfs(pdf_links)
            pdf_docs = []
            for f in pdf_files:
                pdf_docs.extend(to_documents_from_pdf(f, source_url=url))
            all_docs.extend(pdf_docs)
        else:
            if pdf_links:
                print(f"[INFO] Download PDF disabilitato (--pdf=false). Skippati {len(pdf_links)} link PDF.")

        if depth < max_depth:
            for link in html_links:
                if link not in visited:
                    print(f"[DEBUG] -> Da visitare (depth {depth+1}): {link}")
                    to_visit.append((link, depth + 1))

    return all_docs


async def categorize_links(links: list[str]) -> tuple[list[str], list[str]]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
    }

    pdf_links, html_links = [], []

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=30.0,
        headers=headers
    ) as client:
        
        semaphore = asyncio.Semaphore(10)   # limita le richieste concorrenti

        async def check_link(link):
            async with semaphore:
                if await check_if_pdf_url(link, client):
                    pdf_links.append(link)
                elif link.startswith("http") and not link.lower().endswith(('.jpg', '.png', '.gif', '.jpeg')):
                    html_links.append(link)
            
        tasks = [check_link(link) for link in links]
        await asyncio.gather(*tasks, return_exceptions=True)

    return pdf_links, html_links


async def check_if_pdf_url(url: str, client: httpx.AsyncClient) -> bool:
    try:
        if url.lower().endswith('.pdf'):
            return True
        
        response = await client.head(url, timeout=10.0)
        content_type = response.headers.get('content-type', '').lower()

        if 'application/pdf' in content_type:
            return True
        
        if response.status_code == 405:
            response = await client.get(url, headers={'Range': 'bytes=0-1023'}, timeout=10.0)
            content_type = response.headers.get('content-type', '').lower()
            if 'application/pdf' in content_type:
                return True
    
    except Exception as e:
        print(f"[WARN] Errore nel controllo PDF per {url}: {e}")
        return False
    return False