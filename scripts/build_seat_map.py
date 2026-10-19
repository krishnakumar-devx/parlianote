"""Build src/ui/seat_map/seat_map.json, the electorate shapes behind the seat map (it sits beside the map component that loads it).

    python scripts/build_seat_map.py            (downloads the AEC file into data/ if it is not there yet)

Source: Australian Electoral Commission, federal electoral boundaries used at the 2025 election (150 seats),
https://www.aec.gov.au/Electorates/gis/gis_datadownload.htm  (AUS-March-2025-esri.zip, 22 MB).

Needs Node.js: the shapes are simplified with mapshaper (run through npx, pinned below), which keeps the borders shared by
neighbouring seats intact. Detail scales with each seat's size, so a small city seat stays recognisable when the map zooms in
while long outback coastlines stay light. The result is a few hundred KB and is committed, so the app never needs Node.
"""

import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data" / "AUS-March-2025-esri.zip"
URL = "https://www.aec.gov.au/Electorates/files/2025/AUS-March-2025-esri.zip"
OUT = ROOT / "src" / "ui" / "seat_map" / "seat_map.json"
MAPSHAPER = "mapshaper@0.7.72"

# Australian Albers equal-area projection, so the country is not stretched.
PROJ = "+proj=aea +lat_1=-18 +lat_2=-36 +lat_0=0 +lon_0=132 +ellps=GRS80 +units=m"
# Mainland and Tasmania only: drops Christmas and Cocos (Lingiari), Norfolk (Bean), Lord Howe (Sydney) and Macquarie Island (Clark).
CLIP = "bbox=112,-44.5,154.5,-9"
WIDTH = 1000  # the map's drawing units: 1000 across

# The AEC file has no state column, but its records come grouped by state and every state has a different number of seats.
STATE_BY_COUNT = {3: "ACT", 38: "VIC", 46: "NSW", 16: "WA", 5: "TAS", 10: "SA", 30: "QLD", 2: "NT"}
STATE_NAMES = {"ACT": "Australian Capital Territory", "NSW": "New South Wales", "NT": "Northern Territory", "QLD": "Queensland",
               "SA": "South Australia", "TAS": "Tasmania", "VIC": "Victoria", "WA": "Western Australia"}
SPOT_CHECK = {"Warringah": "NSW", "Kennedy": "QLD", "Melbourne": "VIC", "Perth": "WA", "Bean": "ACT", "Lingiari": "NT", "Clark": "TAS",
              "Sturt": "SA"}


def key(name: str) -> str:
    """Seat names compared without case, spaces or punctuation ("O'Connor" = "OConnor")."""
    return re.sub(r"[^a-z]", "", str(name).lower())


def mapshaper(*args):
    """Run the pinned mapshaper through npx and stop on any error."""
    subprocess.run(["npx", "--yes", MAPSHAPER, *args], check=True, capture_output=True, text=True)


def seat_states(work: Path, shp: Path) -> Path:
    """Write states.csv (seat name -> state) from the record order of the AEC file, and check it against known seats."""
    import csv

    attrs = work / "attrs.csv"
    mapshaper(str(shp.with_suffix(".dbf")), "-o", str(attrs), "format=csv")
    rows = list(csv.DictReader(open(attrs, encoding="utf-8")))
    blocks, current = [], []
    for row in rows:
        if current and int(row["E_div_numb"]) <= int(current[-1]["E_div_numb"]):
            blocks.append(current)
            current = []
        current.append(row)
    blocks.append(current)
    if sorted(len(b) for b in blocks) != sorted(STATE_BY_COUNT):
        sys.exit(f"Unexpected seats per state in the AEC file: {[len(b) for b in blocks]}")
    state_of = {row["Elect_div"]: STATE_BY_COUNT[len(b)] for b in blocks for row in b}
    wrong = {s: state_of.get(s) for s, st in SPOT_CHECK.items() if state_of.get(s) != st}
    if wrong:
        sys.exit(f"State check failed: {wrong}")
    out = work / "states.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Elect_div", "state"])
        w.writerows(state_of.items())
    return out


def simplified(work: Path, shp: Path, states_csv: Path, name: str, simplify: list, islands: str, dissolve: bool) -> dict:
    """Clip, project, simplify (and optionally merge seats into states) with mapshaper; returns the GeoJSON."""
    out = work / f"{name}.json"
    mapshaper(str(shp), "-clip", CLIP, "-proj", PROJ, "-join", str(states_csv), "keys=Elect_div,Elect_div", "-simplify", *simplify,
              "keep-shapes", "-filter-islands", f"min-area={islands}", "-filter-fields", "Elect_div,state",
              *(["-dissolve", "state"] if dissolve else []), "-o", str(out), "format=geojson", "precision=1")
    return json.load(open(out, encoding="utf-8"))


def rings(geometry):
    """Every ring of a Polygon or MultiPolygon."""
    polys = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    return [ring for poly in polys for ring in poly]


def main():
    if not ZIP.exists():
        print(f"downloading {URL} ...")
        r = requests.get(URL, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        ZIP.write_bytes(r.content)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        zipfile.ZipFile(ZIP).extractall(work)
        shp = next(work.glob("*.shp"))
        states_csv = seat_states(work, shp)
        detail = ["variable", "interval=Math.max(60, Math.sqrt(Area_SqKm) * 12)"]  # metres: about 1.2% of a seat's width
        layers = {
            "seats": simplified(work, shp, states_csv, "seats", detail, "1km2", False),
            "states": simplified(work, shp, states_csv, "states", detail, "1km2", True),
            "low_seats": simplified(work, shp, states_csv, "low_seats", ["interval=4000"], "150km2", False),
            "low_states": simplified(work, shp, states_csv, "low_states", ["interval=4000"], "150km2", True),
        }

    # one transform for every layer: projected metres -> drawing units, north up
    pts = [p for f in layers["low_states"]["features"] for ring in rings(f["geometry"]) for p in ring]
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
    margin = 12
    scale = (WIDTH - 2 * margin) / (x1 - x0)
    height = round((y1 - y0) * scale + 2 * margin)

    def xy(p, nd):
        return round((p[0] - x0) * scale + margin, nd), round((y1 - p[1]) * scale + margin, nd)

    def path(geometry, nd):
        """SVG path data with relative moves, rounded first so the steps add up exactly."""
        out = []
        for ring in rings(geometry):
            q = [xy(p, nd) for p in ring[:-1]]
            steps = [f"M{q[0][0]:g},{q[0][1]:g}"]
            steps += [f"l{round(b[0] - a[0], nd):g},{round(b[1] - a[1], nd):g}" for a, b in zip(q, q[1:])]
            out.append("".join(steps) + "z")
        return "".join(out)

    def box(geometry):
        q = [xy(p, 2) for ring in rings(geometry) for p in ring]
        xs, ys = [p[0] for p in q], [p[1] for p in q]
        return [round(min(xs), 2), round(min(ys), 2), round(max(xs) - min(xs), 2), round(max(ys) - min(ys), 2)]

    seat_map = {
        "source": "Australian Electoral Commission, federal electoral boundaries used at the 2025 election (mainland and Tasmania)",
        "w": WIDTH,
        "h": height,
        "low": {
            "seats": "".join(path(f["geometry"], 1) for f in layers["low_seats"]["features"]),
            "states": "".join(path(f["geometry"], 1) for f in layers["low_states"]["features"]),
        },
        "states": {
            f["properties"]["state"]: {"name": STATE_NAMES[f["properties"]["state"]], "d": path(f["geometry"], 2), "bbox": box(f["geometry"])}
            for f in layers["states"]["features"]
        },
        "seats": {
            key(f["properties"]["Elect_div"]): {
                "name": f["properties"]["Elect_div"].replace("Mcewen", "McEwen").replace("Mcmahon", "McMahon").replace("Mcpherson", "McPherson")
                .replace("O'connor", "O'Connor"),
                "state": f["properties"]["state"],
                "d": path(f["geometry"], 2),
                "bbox": box(f["geometry"]),
            }
            for f in layers["seats"]["features"]
        },
    }
    if len(seat_map["seats"]) != 150 or len(seat_map["states"]) != 8:
        sys.exit(f"Expected 150 seats in 8 states, got {len(seat_map['seats'])} in {len(seat_map['states'])}")
    OUT.write_text(json.dumps(seat_map, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1024:.0f} KB, 150 seats, drawing {WIDTH} x {height}")


if __name__ == "__main__":
    main()
