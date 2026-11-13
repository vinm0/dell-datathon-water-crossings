"""Rank Texas areas by need for a water crossing.

Pipeline
  1. build_pairs()     one row per flood-response image, with its 'before' image and measured context
  2. analyze_all()     VLM reads each image/pair -> structured JSON (cached, resumable, parallel)
  3. rank()            combine VLM findings with measured data into a 0-100 priority score
  4. neighborhoods(), plot_map(), show_candidates(), export_review_sheet() to inspect and verify

The VLM output is a screening aid. Check it with export_review_sheet() / agreement() before acting on it.
"""
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from texas_flood import STUDENT_DIR, ask_images

RESULTS = Path(__file__).resolve().parent / "results" / "crossing_results.jsonl"

_FIELDS = """Reply with ONLY a JSON object with these keys:
{"flooded_roads": <int estimate>,
 "isolated_buildings": <int estimate>,
 "road_cut_off": true|false,
 "crossing_need": "none"|"low"|"medium"|"high",
 "locations": [{"where": "<image region (top/center/bottom, left/center/right) and the feature>",
                "reason": "<visible clue>"}],  // [] if none
 "uncertainty": "<one sentence>"}"""

_GOAL = """Identify places where a water crossing (bridge, culvert or raised road) would help: roads that are
submerged or cut off, and houses/neighborhoods that floodwater isolates from the road network.
Use only visible evidence; if nothing is flooded, say none and use 0. """

PAIR_PROMPT = ("Image A is a top-down view before flooding; image B is the same area during/after flooding.\n"
               + _GOAL + _FIELDS)
SINGLE_PROMPT = ("The image is a top-down view of an area during/after a flood.\n" + _GOAL + _FIELDS)

# Score weights (sum to 1). Edit to change what "most needed" means.
WEIGHTS = {
    "need": 0.35,       # VLM's crossing_need level
    "severity": 0.20,   # VLM's flooded roads + isolated buildings
    "exposure": 0.10,   # buildings in the footprint (OSM)
    "waterway": 0.10,   # streams/channels present: a crossing needs water to cross
    "sfha": 0.10,       # FEMA Special Flood Hazard Area
    "vulnerability": 0.10,  # county share of households without a car, and poverty rate
    "hwm": 0.05,        # near a measured high-water mark
}
NEED_POINTS = {"none": 0.0, "low": 1 / 3, "medium": 2 / 3, "high": 1.0}
NO_FLOOD_MULTIPLIER = 0.3  # no visible flooding = no evidence of need; damp the context-only score

CONTEXT_COLUMNS = [
    "latitude", "longitude", "osm_road_or_path_count", "osm_building_count", "osm_waterway_count",
    "fema_sfha_tf", "fema_flood_zones", "county_name", "county_no_vehicle_households_percent",
    "county_poverty_percent", "nearest_high_water_mark_distance_km", "elevation_m",
    "terrain_slope_mean_degrees", "preceding_14d_precipitation_mm",
]
NUMERIC = ["latitude", "longitude", "osm_road_or_path_count", "osm_building_count", "osm_waterway_count",
           "county_no_vehicle_households_percent", "county_poverty_percent",
           "nearest_high_water_mark_distance_km", "elevation_m", "terrain_slope_mean_degrees",
           "preceding_14d_precipitation_mm"]


def parse_json(text):
    """Pull the first JSON object out of a model reply; return None if it is malformed."""
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------- 1. data
def build_pairs():
    """One row per response image: image paths plus measured context. 'before_path' is None if unpaired."""
    meta = pd.read_csv(STUDENT_DIR / "image_metadata.csv", keep_default_na=False)
    meta["path"] = meta["image_path"].map(lambda p: STUDENT_DIR / p)
    paths = meta.set_index("image_id")["path"]
    out = meta.loc[meta["image_role"] == "response"].copy()
    out["after_path"] = out["path"]
    out["before_path"] = out["paired_image_id"].map(lambda i: paths.get(i) if i else None)
    for column in NUMERIC:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    keep = ["image_id", "paired_image_id", "locality", "event_name", "acquisition_date",
            "before_path", "after_path"] + [c for c in CONTEXT_COLUMNS]
    return out[keep].reset_index(drop=True)


# ---------------------------------------------------------------- 2. VLM
def analyze_image(before_path, after_path):
    """Run the crossing prompt on a before/after pair (or a single image if before_path is None)."""
    if before_path is None or pd.isna(before_path):
        raw = ask_images([after_path], SINGLE_PROMPT, max_tokens=700)
    else:
        raw = ask_images([before_path, after_path], PAIR_PROMPT, max_tokens=700)
    return {"raw": raw, **(parse_json(raw) or {"crossing_need": "unparsed"})}


def analyze_all(pairs, limit=None, workers=4, cache=RESULTS):
    """Analyze every row, reusing cached answers. Safe to interrupt and rerun."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if cache.is_file():
        for line in cache.read_text().splitlines():
            record = json.loads(line)
            done[record["image_id"]] = record
    todo = pairs.loc[~pairs["image_id"].isin(done)]
    if limit is not None:
        todo = todo.head(limit)
    lock = threading.Lock()

    def work(row):
        try:
            result = analyze_image(row.before_path, row.after_path)
        except Exception as error:  # keep going; record why
            result = {"raw": str(error), "crossing_need": "error"}
        record = {"image_id": row.image_id, **result}
        with lock, cache.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        return record

    with ThreadPoolExecutor(workers) as pool:
        for record in pool.map(work, todo.itertuples()):
            done[record["image_id"]] = record
    return pd.DataFrame(done.values())


# ---------------------------------------------------------------- 3. score
def _percentile(series):
    return series.rank(pct=True).fillna(0.0)


def rank(pairs, results, weights=WEIGHTS):
    """Merge VLM results with context and compute a 0-100 priority score, highest first."""
    table = pairs.merge(results, on="image_id", how="inner")
    for column in ["flooded_roads", "isolated_buildings"]:
        table[column] = pd.to_numeric(table.get(column), errors="coerce")
    need = table["crossing_need"].map(NEED_POINTS)
    parts = pd.DataFrame({
        "need": need.fillna(0.0),
        "severity": (np.log1p(table["flooded_roads"].fillna(0) + table["isolated_buildings"].fillna(0))
                     / np.log1p(30)).clip(0, 1),
        "exposure": _percentile(table["osm_building_count"]),
        "waterway": (table["osm_waterway_count"].fillna(0) / 2).clip(0, 1),
        "sfha": (table["fema_sfha_tf"] == "T").astype(float),
        "vulnerability": (_percentile(table["county_no_vehicle_households_percent"])
                          + _percentile(table["county_poverty_percent"])) / 2,
        "hwm": (1 - table["nearest_high_water_mark_distance_km"].fillna(5).clip(0, 5) / 5),
    })
    score = sum(parts[name] * weight for name, weight in weights.items())
    score = score.where(need.fillna(0) > 0, score * NO_FLOOD_MULTIPLIER)
    table["priority"] = (100 * score).round(1)
    table["needs_review"] = ~table["crossing_need"].isin(NEED_POINTS)  # unparsed / error
    return table.sort_values("priority", ascending=False).reset_index(drop=True)


def neighborhoods(ranked):
    """Roll the tile-level ranking up to named localities."""
    flagged = ranked["crossing_need"].isin(["medium", "high"])
    return (ranked.assign(flagged=flagged)
            .groupby("locality")
            .agg(tiles=("image_id", "size"), flagged_tiles=("flagged", "sum"),
                 top_priority=("priority", "max"), mean_priority=("priority", "mean"),
                 flooded_roads=("flooded_roads", "sum"), isolated_buildings=("isolated_buildings", "sum"))
            .round(1).sort_values("top_priority", ascending=False))


# ---------------------------------------------------------------- 4. inspect
def plot_map(ranked, label_top=10):
    """Scatter of image centers coloured by priority; the top tiles are numbered."""
    geo = ranked.dropna(subset=["latitude", "longitude"])
    fig, ax = plt.subplots(figsize=(9, 7))
    points = ax.scatter(geo["longitude"], geo["latitude"], c=geo["priority"], cmap="YlOrRd", s=28,
                        edgecolor="k", linewidth=0.2)
    for rank_number, (_, row) in enumerate(geo.head(label_top).iterrows(), start=1):
        ax.annotate(str(rank_number), (row["longitude"], row["latitude"]), fontsize=9, weight="bold")
    fig.colorbar(points, label="Crossing priority (0-100)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Where a water crossing is most needed")
    plt.tight_layout()
    plt.show()


def show_candidates(ranked, n=3):
    """Show before/after images and the model's stated locations for the top n."""
    for rank_number, (_, row) in enumerate(ranked.head(n).iterrows(), start=1):
        panels = [("before", row["before_path"]), ("after", row["after_path"])]
        panels = [(name, path) for name, path in panels if path is not None and not pd.isna(path)]
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5), squeeze=False)
        for ax, (name, path) in zip(axes.flat, panels):
            with Image.open(path) as image:
                ax.imshow(image)
            ax.set_title(name)
            ax.axis("off")
        fig.suptitle(f"#{rank_number} {row['image_id']} · {row['locality']} · priority {row['priority']} "
                     f"· need {row['crossing_need']}")
        plt.tight_layout()
        plt.show()
        locations = row.get("locations")
        for place in locations if isinstance(locations, list) else []:
            print(f"  - {place.get('where')}: {place.get('reason')}")
        print("  Uncertainty:", row.get("uncertainty"), "\n")


# ---------------------------------------------------------------- 5. verify
def export_review_sheet(ranked, path="results/review_sheet.csv", top=20, middle=10, bottom=10, seed=0):
    """Write a CSV for a person to fill in 'human_need' (none/low/medium/high) for a mixed sample."""
    middle_rows = ranked.iloc[top:-bottom].sample(min(middle, max(len(ranked) - top - bottom, 0)), random_state=seed)
    sample = pd.concat([ranked.head(top), middle_rows, ranked.tail(bottom)])
    sheet = sample[["image_id", "locality", "crossing_need", "priority", "after_path"]].copy()
    sheet["human_need"] = ""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(path, index=False)
    return sheet


def agreement(path="results/review_sheet.csv"):
    """After 'human_need' is filled in: how often does the model's need level match a person's?"""
    sheet = pd.read_csv(path, keep_default_na=False)
    sheet = sheet.loc[sheet["human_need"] != ""]
    if sheet.empty:
        print("Fill in the human_need column first.")
        return None
    order = list(NEED_POINTS)
    exact = (sheet["crossing_need"] == sheet["human_need"]).mean()
    within_one = ((sheet["crossing_need"].map(order.index) - sheet["human_need"].map(order.index)).abs() <= 1).mean()
    flagged = sheet["crossing_need"].isin(["medium", "high"]) == sheet["human_need"].isin(["medium", "high"])
    print(f"{len(sheet)} reviewed: exact {exact:.0%}, within one level {within_one:.0%}, "
          f"medium/high-vs-not agreement {flagged.mean():.0%}")
    return pd.crosstab(sheet["human_need"], sheet["crossing_need"])
