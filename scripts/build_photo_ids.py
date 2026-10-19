"""Build src/ui/aph_ids.json: each MP's Parliament of Australia ID, so the app can show their official portrait.

    python scripts/build_photo_ids.py

The Hansard corpus records, for every speech, the speaker's ID on the Parliament's website (column "name.id", e.g. "R36" for
Anthony Albanese). The app knows MPs by its own key (uniqueID, or the display name for MPs elected from 2022, as in
src/prepare_app_data.py), so this writes one ID per app key: the one most of their speeches carry. Uses the local corpus in data/
if it is there; otherwise it reads just these three columns from Zenodo (a few MB, not the 579 MB file).

Photos: https://www.aph.gov.au/api/parliamentarian/<ID>/image, Parliament of Australia, CC BY-NC-ND 4.0 (shown unaltered, with credit).
"""

import io
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "data" / "corpus_1998_to_2025.parquet"
URL = "https://zenodo.org/records/17351233/files/corpus_1998_to_2025.parquet?download=1"
OUT = ROOT / "src" / "ui" / "aph_ids.json"
COLUMNS = ["uniqueID", "displayName", "name.id"]
ID = re.compile(r"[0-9A-Z]{2,7}")  # Parliament IDs look like R36, EZ5, 83M, 00AMT
PLACEHOLDERS = {"10000", "1000", "110000", "UNKNOWN"}  # the chair ("The SPEAKER") and unidentified speakers


class Remote(io.RawIOBase):
    """A read-only file over HTTP range requests: enough for pyarrow to read a few columns of a Parquet file in place."""

    def __init__(self, url: str):
        self.url = requests.head(url, allow_redirects=True, timeout=30).url
        self.size = int(requests.head(self.url, timeout=30).headers["Content-Length"])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, n=-1):
        end = self.size - 1 if n < 0 else min(self.pos + n, self.size) - 1
        if end < self.pos:
            return b""
        data = requests.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=60).content
        self.pos += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def main():
    source = LOCAL if LOCAL.exists() else Remote(URL)
    print(f"reading {', '.join(COLUMNS)} from {LOCAL.name if LOCAL.exists() else 'Zenodo'} ...")
    df = pq.read_table(source, columns=COLUMNS).to_pandas()
    df["mp_key"] = df["uniqueID"].fillna(df["displayName"])  # the app's key (src/prepare_app_data.py)
    df["id"] = df["name.id"].astype(str).str.strip().str.upper()
    df = df[df["id"].str.fullmatch(ID) & ~df["id"].isin(PLACEHOLDERS)]

    app_keys = set(pd.read_parquet(ROOT / "app" / "data" / "current_mp.parquet", columns=["mp_key"])["mp_key"])
    app_keys |= set(pd.read_parquet(ROOT / "app" / "data" / "speeches.parquet", columns=["mp_key"])["mp_key"])
    ids = df[df["mp_key"].isin(app_keys)].groupby("mp_key")["id"].agg(lambda s: s.value_counts().index[0]).sort_index()

    shared = ids[ids.duplicated(keep=False)]
    if len(shared):
        print("note: these MPs share an ID (left out):", shared.to_dict())
        ids = ids.drop(shared.index)
    missing = sorted(app_keys - set(ids.index))
    OUT.write_text(json.dumps(ids.to_dict(), indent=0, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(ids)} of {len(app_keys)} MPs have an ID; no ID for {len(missing)}: {missing[:10]}")


if __name__ == "__main__":
    main()
