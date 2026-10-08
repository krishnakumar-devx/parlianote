"""Download the two public datasets Parlianote needs into data/ (skips files that are already there).

    python scripts/download_data.py

* Hansard corpus 1998-2025 (579 MB), Katz & Alexander (2025), https://doi.org/10.5281/zenodo.17351233, CC BY 4.0
* Australian postcodes with federal electorates, Matthew Proctor, https://www.matthewproctor.com/australian_postcodes
"""

import sys
from pathlib import Path

import requests

DATA = Path(__file__).resolve().parents[1] / "data"
FILES = {
    "corpus_1998_to_2025.parquet": "https://zenodo.org/records/17351233/files/corpus_1998_to_2025.parquet?download=1",
    "postcode_electorate.csv": "https://www.matthewproctor.com/Content/postcodes/australian_postcodes.csv",
}


def download(url: str, target: Path):
    """Stream a file to disk, printing progress. The postcode site rejects Python's default user agent, so we send a browser-like one."""
    with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "Mozilla/5.0"}) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        done = 0
        with open(target, "wb") as out:
            for chunk in response.iter_content(chunk_size=1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {target.name}: {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
    print()


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    for name, url in FILES.items():
        target = DATA / name
        if target.exists():
            print(f"already have {name}")
            continue
        print(f"downloading {name} ...")
        try:
            download(url, target)
        except requests.RequestException as error:
            target.unlink(missing_ok=True)  # never leave a half-downloaded file behind
            sys.exit(f"Could not download {name}: {error}")
    print("Done. Next step: see 'Rebuilding the data' in README.md.")
