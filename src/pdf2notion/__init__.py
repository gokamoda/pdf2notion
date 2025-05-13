import os
import tempfile
from itertools import chain
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple
from urllib.request import urlopen

from argparse import ArgumentParser, Namespace
import gyazo
import pymupdf  # type: ignore[import]
import requests
from tqdm import tqdm


def parse_cli_args() -> Namespace:
    parser = ArgumentParser(
        description="Upload PDF file to Gyazo as images then convert to Scrapbox format."
    )
    parser.add_argument("url_or_filepath", help="URL or file path of the PDF.")
    parser.add_argument("title", help="Title of the Notion page.")
    parser.add_argument(
        "--database",
        help="Database ID of the Notion page.",
        required=True,
    )
    parser.add_argument(
        "--gyazo-token",
        default=os.getenv("GYAZO_ACCESS_TOKEN"),
        help="Access token for Gyazo.",
    )
    parser.add_argument(
        "--notion-token",
        default=os.getenv("NOTION_TOKEN"),
        help="Access token for Notion.",
    )
    parser.add_argument("--dpi", type=int, default=100, help="DPI of generating images.")
    parser.add_argument(
        "--spaces", type=int, default=2, help="Number of spaces after images."
    )
    parser.add_argument("--pages", help="PDF pages to upload.")
    parser.add_argument("--extract-links", action="store_true")

    return parser.parse_args()


def parse_range(expr: str) -> Iterator[tuple[int, int]]:
    """Yield start and end integer pairs from a range string like "1-9,12, 15-20,23".

    >>> list(parse_range("1-9,12, 15-20,23"))
    [(1, 9), (12, 12), (15, 20), (23, 23)]

    >>> list(parse_range("1-9,12, 15-20,2-3-4"))
    Traceback (most recent call last):
        ...
    ValueError: format error in 2-3-4
    """
    for x in expr.split(","):
        elem = x.split("-")
        if len(elem) == 1:  # a number
            yield int(elem[0]), int(elem[0])
        elif len(elem) == 2:  # a range inclusive
            yield int(elem[0]), int(elem[1])
        else:  # more than one hyphen
            raise ValueError(f"format error in {x}")


def extract_links_from_pdf(
    pdf_file: str, pages: list[int] | None = None
) -> Iterator[list[str]]:
    with pymupdf.open(pdf_file) as doc:
        pages = pages or list(range(1, len(doc) + 1))
        for i, page in enumerate(doc, 1):
            if i not in pages:
                continue

            yield [link["uri"] for link in page.get_links()]


def build_scrapbox_repr(
    gyazo_urls: list[str],
    expand: bool,
    n_spaces: int,
    links: Iterable[Iterable[str]] | None = None,
) -> str:
    if links is None:
        links = [[]] * len(gyazo_urls)

    blocks = []
    for gyazo_url, links_per_page in zip(gyazo_urls, links):
        block = []
        if expand:
            block.append(f"> [[{gyazo_url}]]\n")
        else:
            block.append(f"> [{gyazo_url}]\n")
        for link in links_per_page:
            block.append(f" {link}\n")
        block.append("\n" * n_spaces)
        blocks.append("".join(block))

    return "".join(blocks)


def download_pdf(url: str) -> str:
    """Return a path to the PDF file downloaded from `url`."""
    resp = urlopen(url)
    f = tempfile.NamedTemporaryFile(delete=False)
    f.write(resp.read())
    return f.name


def pdf2links(
    url_or_filepath: str,
    gyazo_access_token: str,
    dpi: int = 100,
    pages: str | None = None,
    extract_links: bool = False,
) -> str:
    """Upload PDF file to Gyazo as images then convert to Scrapbox format."""
    if url_or_filepath.startswith("http"):
        filepath = download_pdf(url_or_filepath)
    else:
        filepath = url_or_filepath

    client = gyazo.Api(access_token=gyazo_access_token)
    urls = []
    pixmaps = []

    with pymupdf.open(filepath) as doc:
        if pages is None:
            include_pages = set(range(1, len(doc) + 1))
        else:
            include_pages = set()
            for start, end in parse_range(pages):
                include_pages.update(range(start, end + 1))

        for i, page in enumerate(doc, start=1):
            if i in include_pages:
                pixmaps.append(page.get_pixmap(dpi=dpi))

    with tempfile.TemporaryDirectory() as tempdir:
        tempdir_p = Path(tempdir)
        for i, pixmap in enumerate(tqdm(pixmaps)):
            img_path = tempdir_p / f"{i}.png"
            pixmap.save(img_path)
            gyazoimg = client.upload_image(open(img_path, "rb"))
            urls.append(gyazoimg.to_dict()["url"])

    pages_l = None
    if pages is not None:
        # "1-4,6" --> [1, 2, 3, 4, 6]
        pages_l = list(
            chain.from_iterable(
                range(start, end + 1) for start, end in parse_range(pages)
            )
        )

    links = extract_links_from_pdf(filepath, pages_l) if extract_links else None

    return urls, links


def build_notion_repr(
    urls: list[str],
    links: Iterable[Iterable[str]] | None = None,
    n_spaces: int = 1,
) -> str:
    if links is None:
        links = [[]] * len(urls)

    blocks = []
    for gyazo_url, links_per_page in zip(urls, links):
        blocks.append(
            {
                "object": "block",
                "type": "image",
                "image": {
                    "type": "external",
                    "external": {"url": gyazo_url},
                },
            }
        )
        for link in links_per_page:
            blocks.append(
                {
                    "object": "block",
                    "type": "link_preview",
                    "link_preview": {
                        "url": link,
                    },
                }
            )

        for _ in range(n_spaces):  # add empty lines
            blocks.append(
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": ""},
                            }
                        ],
                    },
                }
            )

    return blocks



def get_database_id(database) -> Optional[str]:
    if database.startswith("https://www.notion.so/"):
        # Extract the database ID from the URL
        database_id = database.split("?")[0].split("/")[-1]
        return database_id
    else:
        return database

def add_row_to_database(url, api_key, title, blocks, database):
    database_id = get_database_id(database)
    headers = {
        "Notion-Version": "2022-06-28",
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
    }
    item_data = {
        "parent": {"database_id": f"{database_id}"},
        "properties": {"名前": {"title": [{"text": {"content": title}}]}},
        "children": blocks,
    }
    response = requests.post(url, headers=headers, json=item_data)
    if response.status_code == 200:
        print("Row created successfully.")
    else:
        print("Failed to create row:", response.text)

def cli():
    args = parse_cli_args()
    urls, links = pdf2links(
        url_or_filepath=args.url_or_filepath,
        gyazo_access_token=args.gyazo_token,
        dpi=args.dpi,
        pages=args.pages,
        extract_links=args.extract_links,
    )

    blocks = build_notion_repr(urls,links, n_spaces=args.spaces)
    add_row_to_database(
        url="https://api.notion.com/v1/pages/",
        api_key=args.notion_token,
        blocks=blocks,
        title=args.title,
        database=args.database
    )
