"""Convert every PDF in ./pdf to clean Markdown in ./md for RAG ingestion."""
import re
from datetime import date
from pathlib import Path

import pymupdf4llm

IN_DIR = Path("pdf")
OUT_DIR = Path("md")

# Per-file metadata: (title, doc_type, source_url). Unlisted files default to the file name.
PAGES = {
    "BikerBedInstructions": (
        "BikerBed Installation Instructions",
        "guide",
        "https://neonomad.co.uk/instructions",
    ),
    "DVLA_conversion": (
        "Converting a vehicle into a motor caravan (DVLA guidance)",
        "regulation",
        "https://www.gov.uk/government/publications/converting-a-vehicle-into-a-motor-caravan/converting-a-vehicle-into-a-motor-caravan",
    ),
}

# Lines/blocks that are print or layout artefacts rather than content
NOISE_PATTERNS = [
    r"<!-- Start of picture text -->.*?<!-- End of picture text -->",  # diagram labels (REAR, FRONT)
    r"^\d+(/\d+)? *$",                                               # page numbers: "3", "2/5"
    r"^https://www\.gov\.uk/\S+… *$",                                # browser print footer
    r"^Converting a vehicle into a motor caravan \(motor home\) - GOV\.UK *$",  # browser print header
    r"^\d{2}/\d{2}/\d{4}, \d{2}:\d{2} *$",                           # browser print timestamp
    r"^Contents *\n.*?^This publication is available at .*?$",      # contents list + licence boilerplate
    r"^All content is available under the Open Government Licence.*$",
    r"^© Crown copyright.*$",
]

# Exact fixes for extraction errors that a regex can't sensibly generalise
FIXUPS = {
    "DVLA_conversion": [
        (
            "seats and a table \n\n- sleeping accommodation, which may be converted from the seats cooking facilities",
            "- seats and a table\n- sleeping accommodation, which may be converted from the seats\n- cooking facilities",
        ),
        ("full defnition for required internal featuresi", "full definition for required internal features"),
        ("Swansea SA99 1BA", "Swansea\nSA99 1BA"),
        # URLs split across lines in the print layout
        ("uploads/atta chment_data", "uploads/attachment_data"),
        ("body-typeapplication", "body-type-application"),
        ("converting-a-vehicle-into-a-motorcaravan/", "converting-a-vehicle-into-a-motor-caravan/"),
        ("#motor-caravan-internal-</u> \n\n<u>features)</u>", "#motor-caravan-internal-features)</u>"),
    ],
    "BikerBedInstructions": [
        ("Installation Steps \n", "## Installation Steps\n"),
    ],
}


def to_markdown(src: Path) -> str:
    text = pymupdf4llm.to_markdown(str(src))
    for old, new in FIXUPS.get(src.stem, []):
        text = text.replace(old, new)
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.MULTILINE)
    text = re.sub(r"</?u>", "", text)                       # underline tags from hyperlinks
    text = re.sub(r"^(#+) \*\*(.+?)\*\*", r"\1 \2", text, flags=re.MULTILINE)  # bold inside headings
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)  # trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)                   # collapse blank lines
    return text.strip()


def front_matter(src: Path) -> str:
    title, doc_type, url = PAGES.get(src.stem, (src.stem.replace("_", " "), "notes", ""))
    return (
        "---\n"
        f"title: {title}\n"
        f"source_url: {url}\n"
        f"source_file: {src.as_posix()}\n"
        f"doc_type: {doc_type}\n"
        f"updated: {date.today()}\n"
        "---\n\n"
    )


def main() -> None:
    files = sorted(IN_DIR.glob("*.pdf"))
    if not files:
        print(f"No PDF files found in {IN_DIR}/")
        return

    for src in files:
        out = OUT_DIR / f"{src.stem}.md"
        out.write_text(front_matter(src) + to_markdown(src) + "\n", encoding="utf-8")
        print(f"{src} -> {out}")

    print(f"Converted {len(files)} file(s).")


if __name__ == "__main__":
    main()
