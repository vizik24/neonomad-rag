"""Convert every HTML file in ./html to clean Markdown in ./md for RAG ingestion."""
import re
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import markdownify as md

IN_DIR = Path("html")
OUT_DIR = Path("md")
STRIP_TAGS = ["script", "style", "noscript", "svg", "nav", "footer", "header", "form", "button", "iframe", "video"]
SITE = "https://neonomad.co.uk"

# Per-page metadata: (doc_type, url path). Pages not listed default to ("notes", "/").
PAGES = {
    "bikerbed": ("product", "/product?item=bikerbed"),
    "kitchen": ("product", "/product?item=kitchen"),
    "mattress": ("product", "/product?item=mattress"),
    "faq": ("faq", "/faqs"),
    "shipping": ("policy", "/shipping"),
    "warranty_returns": ("policy", "/returns"),
    "terms_conditions": ("policy", "/termsandconditions"),
    "privacy": ("policy", "/privacy"),
    "about": ("about", "/about"),
    "contact": ("contact", "/contact"),
    "homepage": ("overview", "/"),
}

# Markdown blocks repeated across pages that add noise to retrieval
NOISE_PATTERNS = [
    r"\[[^\]]*\]\([^)]*\.mp4\)",                                  # video placeholders
    r"Frequently Purchased With.*?(?=\nFAQs|\Z)",                 # cross-sell product cards
    r"### Fast UK Shipping.*?Get in touch with our support team\.",  # trust footer
    r"(### Will it fit\?\n+)?Start typing to find your vehicle from our list\.",  # vehicle search widget
    r"^FAQs$",                                                    # section label above Q&As
]


def clean_html(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else "Untitled"

    for tag in soup(STRIP_TAGS):
        tag.decompose()
    # Common cookie/banner containers
    for el in soup.select('[class*="cookie"], [id*="cookie"], [class*="banner"]'):
        el.decompose()

    body = soup.find("main") or soup.body or soup
    # Page title is the first h1; the <title> tag is the same site-wide
    h1 = body.find("h1")
    if h1:
        title = h1.get_text(strip=True)
    return title, str(body)


def to_markdown(html: str) -> str:
    text = md(html, heading_style="ATX", strip=["img", "a"])  # keep link text, drop URLs/images
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.MULTILINE)
    text = re.sub(r"^(£[\d.,]+)\n\n\1$", r"\1", text, flags=re.MULTILINE)  # duplicated price
    # Standalone questions become headings so each Q&A chunks on its own
    text = re.sub(r"^(?!#)([^\n]{5,200}\?)$", r"## \1", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse blank lines
    return text.strip()


def drop_seen_sections(text: str, seen: set[str]) -> str:
    """Remove '## ' sections whose heading already appeared in an earlier file (e.g. FAQ repeated on contact)."""
    parts = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    kept = []
    for part in parts:
        heading = part.splitlines()[0] if part.startswith("## ") else None
        if heading and heading in seen:
            continue
        if heading:
            seen.add(heading)
        kept.append(part)
    return "".join(kept).strip()


def front_matter(title: str, source: Path) -> str:
    doc_type, path = PAGES.get(source.stem, ("notes", "/"))
    if "|" in title:  # no h1 found, so <title> fell through; use the file name instead
        title = source.stem.replace("_", " ").title()
    return (
        "---\n"
        f"title: {title}\n"
        f"source_url: {SITE}{path}\n"
        f"source_file: {source.as_posix()}\n"
        f"doc_type: {doc_type}\n"
        f"updated: {date.today()}\n"
        "---\n\n"
    )


def main() -> None:
    # Canonical pages first, so duplicated sections are kept there and dropped from later pages
    order = list(PAGES)
    files = sorted(IN_DIR.rglob("*.htm*"), key=lambda p: (order.index(p.stem) if p.stem in order else len(order), p.name))
    if not files:
        print(f"No HTML files found in {IN_DIR}/")
        return

    seen: set[str] = set()
    for src in files:
        title, body = clean_html(src.read_text(encoding="utf-8", errors="ignore"))
        text = drop_seen_sections(to_markdown(body), seen)
        out = OUT_DIR / src.relative_to(IN_DIR).with_suffix(".md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(front_matter(title, src) + text + "\n", encoding="utf-8")
        print(f"{src} -> {out}")

    print(f"Converted {len(files)} file(s).")


if __name__ == "__main__":
    main()