import io
import json
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import pandas as pd
import PIL.Image
import PIL.ImageOps
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from pypdf import PdfReader, PdfWriter

try:  # iPhone photos are often HEIC; support them if pillow-heif is installed
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIC_OK = True
except ImportError:
    HEIC_OK = False

st.set_page_config(page_title="Wine List Value Scanner", page_icon="🍷", layout="wide")

MAX_IMAGE_SIDE = 2400  # downscale big phone photos; plenty for reading a menu
THIS_YEAR = date.today().year
# Default importance of each factor in the overall Value Score (user can change these)
DEFAULT_WEIGHTS = {
    "Price vs. retail": 35, "Critic standing": 30, "Drinking window": 20,
    "Vintage quality": 15, "My taste": 25,
}
# Tried in order if the key can't list models (or the listed pick fails)
# Lighter models are a last resort for analysis when Google's main models are overloaded
EXTRA_ANALYSIS_FALLBACKS = ["gemini-flash-lite-latest", "gemini-2.5-flash-lite"]
FALLBACK_MODELS = ["gemini-flash-latest", "gemini-3.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash"]


# ---------------------------------------------------------------- data model
# Step 1 reads the list (ListedWine); step 2 evaluates it in batches (Evaluation).
class ListedWine(BaseModel):
    producer: str
    name: str = Field(description="Cuvée / bottling name, or the varietal if none")
    vintage: str = Field(description="Year, or 'NV'")
    region: str = Field(description="Region / appellation and country")
    grape: str = Field(description="Main grape(s)")
    color: str = Field(description="One of: Red, White, Rosé, Sparkling, Dessert, Orange")
    format: str = Field(description="Bottle size in ml, e.g. '750ml', '375ml', '1500ml'; 'Glass' for a by-the-glass price. Use 750ml unless the wine's own line or its section heading explicitly says otherwise")
    list_price: float = Field(description="Price on the wine list, number only")


class ListedWines(BaseModel):
    restaurant: str = Field(description="Restaurant name if visible, else ''")
    wines: list[ListedWine]


class Evaluation(BaseModel):
    id: int = Field(description="The id number given for the wine")
    est_retail_price: float = Field(description="Typical retail shop price for this bottle and vintage, number only")
    retail_confidence: str = Field(description="High, Medium or Low")
    critic_score: int = Field(description="Estimated professional critic score, 100-point scale. Never 0")
    quality_basis: str = Field(description="Exactly one of: 'Web reviews', 'This wine', 'Producer', 'Region/style'")
    vintage_quality: int = Field(description="Vintage quality for this region, 1 (poor) to 5 (outstanding); 0 only for NV")
    drink_from: int = Field(description="First year of the drinking window. Never 0")
    drink_to: int = Field(description="Last year of the drinking window. Never 0")
    underrated: bool = Field(description="True only for genuine sommelier finds")
    taste_match: int = Field(description="How well it fits the diner's stated preferences, 1 (poor) to 5 (perfect); 0 if no preferences were given")
    note: str = Field(description="One short, specific sentence")


class Evaluations(BaseModel):
    evaluations: list[Evaluation]


class QuickEstimate(BaseModel):
    id: int
    retail: float = Field(description="Typical retail shop price, number only")
    critic: int = Field(description="Typical critic score, 100-point scale, never 0")


class QuickEstimates(BaseModel):
    estimates: list[QuickEstimate]


class Wine(ListedWine):
    est_retail_price: float = 0
    retail_confidence: str = "Low"
    critic_score: int = 0
    quality_basis: str = ""
    vintage_quality: int = 0
    drink_from: int = 0
    drink_to: int = 0
    underrated: bool = False
    taste_match: int = 0
    note: str = ""


class WineList(BaseModel):
    currency: str
    restaurant: str
    wines: list[Wine]
    sources: list[tuple[str, str]] = []  # (title, url) of web pages Google Search used
    total_listed: int = 0  # wines on the whole list
    researched: int = 0  # wines that got the full (deep) evaluation


# label -> (symbol, currency name, retail market to price against)
CURRENCIES = {
    "US Dollars ($)": ("$", "US dollars", "the United States"),
    "British Pounds (£)": ("£", "British pounds", "the United Kingdom"),
    "Euros (€)": ("€", "euros", "the eurozone (France, Germany, Italy, etc.)"),
    "Canadian Dollars (C$)": ("C$", "Canadian dollars", "Canada"),
    "Australian Dollars (A$)": ("A$", "Australian dollars", "Australia"),
}


def currency_note(label: str) -> str:
    sym, name, market = CURRENCIES[label]
    return (
        f"\nIMPORTANT: Give retail prices in {name}, as a shopper in {market} would pay at a "
        "wine shop. Restaurant prices are deliberately not shown: estimate retail independently "
        "from your knowledge of the wine market.\n"
    )


EXTRACT_PROMPT = """You are reading a restaurant wine list (all pages/images provided).

Extract EVERY wine on it. For each wine, record producer, name, vintage, region, grape,
color, format and list price exactly as printed (fill region/grape from your knowledge
if the list only implies them, e.g. by its section heading). If both a glass and a
bottle price are shown, use the bottle price. Skip beers, spirits, cocktails and
other non-wine items. Do not skip any wine, however long the list.

Bottle size: use 750ml unless that wine's own line, or the heading of the section it
sits in, explicitly gives another size (e.g. "Half Bottles" → 375ml, "Magnums" →
1500ml). A size heading applies only to the wines under it, not to the rest of the list."""

EVAL_TEMPLATE = """You are an expert sommelier and wine-pricing analyst. It is now {year}.

For EVERY wine below, return an evaluation with the same id. Do not skip any.

- est_retail_price: the typical current retail (shop) price of that bottle and vintage.
  Be realistic: a mainstream wine shop, not auction or release prices for rare wines.
  retail_confidence = Low if you are unsure of the producer or bottling.
- critic_score: ALWAYS give your best estimate on the 100-point scale; never 0.
  Work down this ladder and record which step you used in quality_basis:
    'This wine'    - you know reviews of this wine (this or nearby vintages)
    'Producer'     - you know the producer's usual level for this kind of bottling
    'Region/style' - otherwise, what a wine of this appellation, grape, price tier and
                     vintage typically scores (a $15 retail Côtes du Rhône ~87, a
                     good Premier Cru Burgundy ~91-93, etc.)
- vintage_quality: 1-5 from vintage charts for that region (0 only for NV). If the
  region is obscure, use the nearest major region's vintage as a guide.
- drink_from / drink_to: ALWAYS give a window (years). If you don't know the wine,
  use the norms for its style, grape, region and quality level. NV wines: drink now
  through the next 2-3 years.
- underrated: true only for genuine sommelier finds: lesser-known regions, grapes or
  producers that over-deliver, grower Champagne, well-aged back-vintages priced
  modestly, etc. Most wines should be false.
- taste_match: 0 unless the diner's preferences are given below; then 1-5 for how well
  the wine fits them.
- note: one short, specific sentence on quality, style and readiness (and the
  diner's taste, if given).

Wines (id | wine):
"""

EVAL_PROMPT = EVAL_TEMPLATE.replace("{year}", str(THIS_YEAR))

SEARCH_NOTE = """
Use Google Search to look up critic reviews/scores, vintage reports and typical retail
prices for these wines (e.g. Wine-Searcher, Vinous, Wine Spectator, Decanter,
JancisRobinson.com, CellarTracker). When a score comes from what you found, set
quality_basis to 'Web reviews'. Don't spend searches on wines you already know well.

Return ONLY a JSON object, no other text, in exactly this shape:
{"evaluations": [{"id": 0, "est_retail_price": 0.0, "retail_confidence": "Medium",
  "critic_score": 90, "quality_basis": "Web reviews", "vintage_quality": 4,
  "drink_from": 2024, "drink_to": 2032, "underrated": false, "taste_match": 0,
  "note": "..."}]}
"""


def prefs_note(prefs: str) -> str:
    if not prefs.strip():
        return ""
    return f"\nThe diner's preferences (use them for taste_match and the note):\n{prefs.strip()}\n"


# ---------------------------------------------------------------- helpers
def get_secret(name, default=None):
    """Read from Streamlit secrets, falling back to env vars (works locally without secrets.toml)."""
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip().strip("\"'")
    except Exception:
        pass
    return os.environ.get(name, default)


# Quick checks must fail fast: no silent retries, short timeout (the SDK otherwise retries
# rate-limit errors with backoff, which looks like the app hanging on "Connecting…").
QUICK = types.HttpOptions(timeout=20_000, retry_options=types.HttpRetryOptions(attempts=1))


def make_client(api_key: str, vertex: bool):
    opts = types.HttpOptions(timeout=180_000)  # full wine-list analysis can take a while
    if vertex:
        return genai.Client(vertexai=True, api_key=api_key, http_options=opts)
    return genai.Client(api_key=api_key, http_options=opts)


def _version_key(name: str):
    nums = re.findall(r"\d+(?:\.\d+)?", name)
    return float(nums[0]) if nums else 0.0


def pick_model(client) -> str | None:
    """Pick the newest stable 'flash' model this key can use (None if the key can't list models).

    Hard-coding a model name is what broke this app before: Google retires models regularly.
    """
    names = []
    for m in client.models.list(config=types.ListModelsConfig(http_options=QUICK)):
        actions = m.supported_actions or []
        name = (m.name or "").split("/")[-1]
        if actions and "generateContent" not in actions:
            continue
        if not name.startswith("gemini-") or "flash" not in name:
            continue
        if any(bad in name for bad in ("lite", "image", "tts", "audio", "live", "embedding", "exp")):
            continue
        names.append(name)
    stable = [n for n in names if "preview" not in n] or names
    if not stable:
        return None
    return max(stable, key=lambda n: (_version_key(n), "latest" not in n, -len(n)))


@st.cache_resource(show_spinner="Connecting to Gemini…")
def connect(api_key: str, model_override: str | None, force_vertex: bool):
    """Return (client, model, backend). Raises with every attempt's error if nothing works.

    Don't guess the backend from the key prefix: AI Studio now issues "AQ." keys too, and
    sending those to Vertex AI was why the app kept failing. Try the Gemini API first and
    fall back to Vertex AI express mode.
    """
    attempts = [True] if force_vertex else [False, True]
    errors = []
    for vertex in attempts:
        backend = "Vertex AI" if vertex else "Gemini API"
        try:
            client = make_client(api_key, vertex)
        except Exception as e:
            errors.append(f"{backend} (setup): {e}")
            continue
        if model_override:
            candidates = [model_override]
        else:
            try:  # some keys can generate but not list models; that's fine
                picked = pick_model(client)
            except Exception as e:
                picked = None
                errors.append(f"{backend} (list models): {e}")
            candidates = [m for m in (picked, *FALLBACK_MODELS) if m]
            candidates = list(dict.fromkeys(candidates))  # dedupe, keep order
        for model in candidates:
            try:  # a tiny real request: the only check that proves key + model both work
                client.models.generate_content(
                    model=model,
                    contents="Reply with OK.",
                    config=types.GenerateContentConfig(max_output_tokens=50, http_options=QUICK),
                )
                return client, model, backend
            except Exception as e:
                errors.append(f"{backend} / {model}: {e}")
    raise RuntimeError("\n\n".join(errors))


def prepare_image(data: bytes) -> types.Part:
    img = PIL.Image.open(io.BytesIO(data))
    img = PIL.ImageOps.exif_transpose(img)  # fix sideways phone photos
    img = img.convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


PDF_PAGES_PER_CHUNK = 2  # long lists overflow one answer, so read a couple of pages at a time


def split_pdf(data: bytes) -> list[tuple[str, bytes]]:
    """Split a PDF into small page groups: [(label, pdf_bytes), ...]."""
    try:
        reader = PdfReader(io.BytesIO(data))
        n = len(reader.pages)
        if n <= PDF_PAGES_PER_CHUNK:
            return [(f"pages 1–{n}", data)]
        out = []
        for start in range(0, n, PDF_PAGES_PER_CHUNK):
            writer = PdfWriter()
            end = min(start + PDF_PAGES_PER_CHUNK, n)
            for i in range(start, end):
                writer.add_page(reader.pages[i])
            buf = io.BytesIO()
            writer.write(buf)
            out.append((f"pages {start + 1}–{end}", buf.getvalue()))
        return out
    except Exception:  # unreadable by pypdf: let Gemini have the whole thing
        return [("whole PDF", data)]


def size_ml(fmt: str) -> int:
    """'375ml' -> 375, '1.5L' -> 1500, 'Magnum' -> 1500, 'Glass' -> 0; default 750."""
    f = (fmt or "").lower().replace(" ", "")
    if "glass" in f or "btg" in f:
        return 0
    if "half" in f:
        return 375
    if "magnum" in f:
        return 1500
    m = re.search(r"(\d+(?:\.\d+)?)(ml|cl|l)", f)
    if not m:
        return 750
    n = float(m.group(1))
    return int(n * {"ml": 1, "cl": 10, "l": 1000}[m.group(2)])


PRICED_LINE = re.compile(r"^(?P<body>.*?\S)\s+\$?(?P<price>\d{1,3}(?:,\d{3})+|\d{2,6})(?:\.\d{2})?\s*$")
VINTAGE_START = re.compile(r"^(?P<v>NV|N\.V\.|(?:19|20)\d{2})\b[\s,]*(?P<rest>.+)$", re.I)
SIZE_NOTE = re.compile(r"\((\d+(?:\.\d+)?)\s*(ml|cl|l)\)", re.I)


def color_of(heading: str) -> str:
    t = heading.lower()
    if any(k in t for k in ("sweet", "fortif", "dessert")):
        return "Dessert"
    if any(k in t for k in ("rosé", "rose ", "skin", "orange")):
        return "Rosé"
    if "red" in t:
        return "Red"
    if "white" in t:
        return "White"
    if "sparkling" in t or "champagne" in t:
        return "Sparkling"
    return ""


def parse_pdf_text(data: bytes) -> tuple[list[ListedWine], int]:
    """Read a text-based PDF list locally (instant, free, no page limits).

    Handles the common layout "VINTAGE Producer, Wine, Appellation ... PRICE", using each
    page's heading for color/region ("Red Wines of Piedmont", "Half Bottles | ...") and
    all-caps lines as sub-sections. Returns (wines, number_of_priced_lines) so the caller
    can tell whether this layout fits; scanned PDFs return ([], 0).
    """
    reader = PdfReader(io.BytesIO(data))
    wines, priced = [], 0
    for page in reader.pages:
        text = unicodedata.normalize("NFKC", page.extract_text() or "")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            continue
        heading = "" if PRICED_LINE.match(lines[0]) else lines[0]
        half = "half bottle" in heading.lower()
        color = color_of(heading)
        region = re.sub(r"^.*?\bof\b\s*", "", heading.split("|")[-1]).strip(" |")
        section = ""
        for line in lines[1:]:
            m = PRICED_LINE.match(line)
            if not m:
                if line.isupper() and len(line) < 60:  # e.g. "BAROLO", "CHENIN BLANC & BLENDS"
                    section = line.title()
                continue
            priced += 1
            v = VINTAGE_START.match(m.group("body"))
            if not v:
                continue
            body = v.group("rest")
            ml = 375 if half else 750
            sz = SIZE_NOTE.search(body)
            if sz:
                ml = int(float(sz.group(1)) * {"ml": 1, "cl": 10, "l": 1000}[sz.group(2).lower()])
            body = SIZE_NOTE.sub("", body)
            body = re.sub(r"\(\d{1,2}\.\d{2}\)", "", body)  # disgorgement dates like (7.21)
            body = re.sub(r"\s{2,}", " ", body).strip(" ,")
            producer, _, name = body.partition(",")
            wines.append(ListedWine(
                producer=producer.strip(),
                name=name.strip(),
                vintage="NV" if v.group("v").upper().startswith("N") else v.group("v"),
                region=", ".join(x for x in (section, region) if x),
                grape="",
                color=color,
                format=f"{ml}ml",
                list_price=float(m.group("price").replace(",", "")),
            ))
    return wines, priced


def size_label(ml: int) -> str:
    if ml == 0:
        return "Glass"
    if ml == 375:
        return "Half bottle (375ml)"
    if ml < 750:
        return f"Small ({ml}ml)"
    if ml == 750:
        return "Standard (750ml)"
    return f"Large ({ml / 1000:g}L)"


EVAL_BATCH = 15  # wines per evaluation request: small enough that every wine gets attention
TRIAGE_BATCH = 120  # wines per quick-estimate request (big lists only)
COLORS = ["Red", "White", "Sparkling", "Rosé", "Orange", "Dessert"]

TRIAGE_TEMPLATE = """You are an expert sommelier and wine-pricing analyst. It is now {year}.
For EVERY wine below give, quickly and from your own knowledge:
- retail: the typical current retail (shop) price of that bottle and vintage
- critic: the typical professional critic score (100-point scale). If you don't know
  the exact wine, use the producer's usual level, else what that appellation, vintage
  and price tier typically scores. Never 0.
Return one estimate per wine, using the same id.

Wines (id | wine):
"""
SEARCH_BATCH = 8  # smaller batches when searching, so each wine actually gets looked up


def generate_json(client, model: str, contents, schema):
    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.2,
            # we handle "busy" errors ourselves by switching model, so don't retry silently
            http_options=types.HttpOptions(
                timeout=180_000, retry_options=types.HttpRetryOptions(attempts=1)
            ),
        ),
    )
    if resp.parsed is not None:
        return resp.parsed
    return schema.model_validate_json(resp.text)


def generate_json_searched(client, model: str, contents, schema):
    """Same as generate_json but lets Gemini use Google Search first.

    Search can't be combined with forced-JSON mode on every model, so ask for JSON in the
    prompt and pull it out of the text. Returns (result, [(title, url), ...]).
    """
    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.2,
            http_options=types.HttpOptions(
                timeout=240_000, retry_options=types.HttpRetryOptions(attempts=1)
            ),
        ),
    )
    text = resp.text or ""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Search answer contained no JSON")
    result = schema.model_validate(json.loads(text[start : end + 1]))
    sources = []
    try:
        for chunk in resp.candidates[0].grounding_metadata.grounding_chunks or []:
            if chunk.web and chunk.web.uri:
                sources.append((chunk.web.title or chunk.web.domain or "source", chunk.web.uri))
    except (AttributeError, IndexError, TypeError):
        pass
    return result, sources


def is_busy_or_missing(msg: str) -> bool:
    """Errors worth trying another model for: overloaded, rate-limited, timed out or retired."""
    keys = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL",
            "504", "DEADLINE", "timed out", "timeout", "404", "NOT_FOUND")
    return any(k.lower() in msg.lower() for k in keys)


def with_fallback(client, models: list[str], contents, schema, log):
    """Try each model in turn so one overloaded model doesn't sink the scan. Returns (result, model)."""
    errors = []
    for m in models:
        try:
            return generate_json(client, m, contents, schema), m
        except Exception as e:
            msg = str(e)
            if not is_busy_or_missing(msg):
                raise
            errors.append(f"{m}: {msg}")
            log(f"⏭️ {m} is busy or unavailable, trying another model…")
    raise RuntimeError("ALL_BUSY\n\n" + "\n\n".join(errors))


def describe(w: ListedWine) -> str:
    bits = [w.producer, w.name, w.vintage, w.region, w.grape, w.color]
    fmt = "" if w.format.lower() in ("", "750ml", "750 ml", "bottle") else f" [{w.format}]"
    return " · ".join(b for b in bits if b) + fmt


def read_wines(client, models, files, status) -> tuple[str, list[ListedWine], str]:
    """Step 1: get every wine off the list. Returns (restaurant, wines, model_used)."""
    wines: list[ListedWine] = []
    ai_chunks: list[tuple[str, list[types.Part]]] = []
    for f in files:
        data = f.getvalue()
        if f.name.lower().endswith(".pdf"):
            try:
                parsed, priced = parse_pdf_text(data)
            except Exception:
                parsed, priced = [], 0
            if len(parsed) >= 15 and len(parsed) >= 0.6 * priced:
                status.write(f"📄 {f.name}: read {len(parsed):,} wines directly from the PDF text")
                wines += parsed
                continue
            for label, pdf in split_pdf(data):
                ai_chunks.append((f"{f.name} {label}",
                                  [types.Part.from_bytes(data=pdf, mime_type="application/pdf")]))
        else:
            ai_chunks.append((f.name, [prepare_image(data)]))
    if not ai_chunks:
        return "", wines, models[0]

    status.update(label=f"Step 1: reading the wine list ({len(ai_chunks)} part(s))… (20–60 s)")
    read: dict[int, ListedWines] = {}
    used_models, msgs = [], []

    def read_chunk(parts):
        out, m = with_fallback(client, models, [*parts, EXTRACT_PROMPT], ListedWines, msgs.append)
        used_models.append(m)
        return out

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(read_chunk, parts): i for i, (_, parts) in enumerate(ai_chunks)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                read[i] = fut.result()
                status.write(f"📄 {ai_chunks[i][0]}: {len(read[i].wines)} wines")
            except Exception as e:
                msgs.append(f"⚠️ Couldn't read {ai_chunks[i][0]}: {str(e)[:150]}")
    for m in dict.fromkeys(msgs):
        status.write(m)
    if not read and not wines:
        raise RuntimeError("ALL_BUSY\n\n" + "\n".join(msgs))
    restaurant = next((r.restaurant for _, r in sorted(read.items()) if r.restaurant), "")
    for _, r in sorted(read.items()):  # keep list order
        wines += r.wines
    return restaurant, wines, (used_models or models)[0]


def in_scope(w: ListedWine, scope: dict) -> bool:
    ml = size_ml(w.format)
    if ml < 750 and not scope["include_small"]:
        return False
    if scope["max_price"] and w.list_price > scope["max_price"]:
        return False
    color = "Rosé" if w.color == "Orange" and "Orange" not in scope["colors"] else w.color
    return not color or color not in COLORS or color in scope["colors"]


def analyze(client, model: str, files, cur_label: str, prefs: str, use_search: bool,
            scope: dict, status) -> tuple[WineList, str]:
    """Read the list, narrow it to what the diner wants, then evaluate.

    Big lists (hundreds or thousands of wines) get a quick price/critic estimate for every
    wine first, and only the most promising `deep_n` get the full evaluation, which keeps
    a 2,500-wine list within free-plan limits.
    """
    models = list(dict.fromkeys([model, *FALLBACK_MODELS, *EXTRA_ANALYSIS_FALLBACKS]))
    sym = CURRENCIES[cur_label][0]

    status.update(label="Step 1: reading the wine list…")
    restaurant, listed, used = read_wines(client, models, files, status)
    wines, seen = [], set()
    for w in listed:
        key = (w.producer.lower(), w.name.lower(), w.vintage, size_ml(w.format), w.list_price)
        if w.list_price > 0 and key not in seen:
            seen.add(key)
            wines.append(w)
    total = len(wines)
    wines = [w for w in wines if in_scope(w, scope)]
    status.write(f"🎯 {len(wines):,} of {total:,} wines match what you're looking for")
    if not wines:
        return WineList(currency=sym, restaurant=restaurant, wines=[], total_listed=total), used

    # Keep using whichever model worked, with the others as backups
    models = [used, *[m for m in models if m != used]]
    messages: list[str] = []
    notes = currency_note(cur_label)

    def line(i):
        # No list price here on purpose: shown the menu price, the model tends to echo it
        # back as the "retail" estimate, which made every markup come out at 1.0×.
        return f"{i} | {describe(wines[i])}"

    # --- quick pass (big lists only): rough retail + critic score for every wine
    quick: dict[int, QuickEstimate] = {}
    deep_ids = list(range(len(wines)))
    if len(wines) > scope["deep_n"]:
        batches = [deep_ids[i : i + TRIAGE_BATCH] for i in range(0, len(wines), TRIAGE_BATCH)]
        status.update(label=f"Step 2: quick price & critic check of {len(wines):,} wines…")
        prompt0 = TRIAGE_TEMPLATE.replace("{year}", str(THIS_YEAR))

        def run_quick(ids):
            prompt = prompt0 + "\n".join(line(i) for i in ids) + "\n" + notes
            out, _ = with_fallback(client, models, prompt, QuickEstimates, messages.append)
            return out.estimates

        done = 0
        with ThreadPoolExecutor(max_workers=3) as pool:
            for fut in as_completed([pool.submit(run_quick, ids) for ids in batches]):
                try:
                    for q in fut.result():
                        if 0 <= q.id < len(wines):
                            quick[q.id] = q
                except Exception as e:
                    messages.append(f"⚠️ A quick-check batch failed: {str(e)[:150]}")
                done += 1
                status.update(label=f"Step 2: quick check {done}/{len(batches)} batches…")

        def prelim(i):
            q = quick.get(i)
            if not q or q.retail <= 0:
                return -1
            markup_pts = 100 * _clamp01((4.0 - wines[i].list_price / q.retail) / 2.5)
            critic_pts = 100 * _clamp01((q.critic - 84) / 12)
            return 0.55 * markup_pts + 0.45 * critic_pts

        deep_ids = sorted(quick, key=prelim, reverse=True)[: scope["deep_n"]]
        status.write(f"🔬 Researching the top {len(deep_ids)} candidates in depth")

    # --- full evaluation of the shortlist
    size = SEARCH_BATCH if use_search else EVAL_BATCH
    batches = [deep_ids[i : i + size] for i in range(0, len(deep_ids), size)]
    evals: dict[int, Evaluation] = {}
    sources: dict[str, str] = {}
    search_ok = [use_search]  # flips off for the rest of the scan if search stops working

    def run_batch(ids):
        prompt = EVAL_PROMPT + "\n".join(line(i) for i in ids) + "\n" + notes + prefs_note(prefs)
        if search_ok[0]:
            for m in models[:2]:
                try:
                    out, found = generate_json_searched(client, m, prompt + SEARCH_NOTE, Evaluations)
                    for title, url in found:
                        sources.setdefault(url, title)
                    return out.evaluations
                except Exception as e:
                    err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "search" in err.lower():
                search_ok[0] = False  # quota used up or not supported: stop trying
            messages.append(f"🔎 Web search unavailable for some wines, used built-in knowledge instead ({err[:120]})")
        out, _ = with_fallback(client, models, prompt, Evaluations, messages.append)
        return out.evaluations

    how = "checking reviews online" if use_search else "evaluating"
    status.update(label=f"Step 3: {how} for {len(deep_ids)} wines… (1–3 min)")
    done = 0
    with ThreadPoolExecutor(max_workers=3) as pool:  # modest parallelism for free-tier rate limits
        for fut in as_completed([pool.submit(run_batch, ids) for ids in batches]):
            try:
                for e in fut.result():
                    evals[e.id] = e
            except Exception as e:  # one failed batch shouldn't lose the whole list
                messages.append(f"⚠️ A batch failed: {str(e)[:200]}")
            done += 1
            status.update(label=f"Step 3: evaluated {done}/{len(batches)} batches…")

    # Re-ask once for any shortlisted wines the model skipped
    missing = [i for i in deep_ids if i not in evals]
    if missing and len(missing) < len(deep_ids):
        status.write(f"🔁 Re-checking {len(missing)} wines that were skipped…")
        try:
            for e in run_batch(missing):
                evals[e.id] = e
        except Exception as e:
            messages.append(f"⚠️ Re-check failed: {str(e)[:200]}")
    for m in dict.fromkeys(messages):
        status.write(m)
    if not evals and not quick:
        raise RuntimeError("ALL_BUSY\n\n" + "\n".join(messages))

    merged = []
    for i, w in enumerate(wines):
        if i in evals:
            extra = evals[i].model_dump(exclude={"id"})
        elif i in quick:
            extra = {"est_retail_price": quick[i].retail, "critic_score": quick[i].critic,
                     "quality_basis": "Quick estimate", "retail_confidence": "Medium"}
        else:
            extra = {}
        merged.append(Wine(**w.model_dump(), **extra))
    return WineList(
        currency=sym, restaurant=restaurant, wines=merged,
        sources=[(t, u) for u, t in sources.items()],
        total_listed=total, researched=len(evals),
    ), used


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def drink_status(w: Wine) -> tuple[str, float]:
    """(label, 0-100 score) for where the wine is in its drinking window this year."""
    if not w.drink_from or not w.drink_to:
        return ("🟢 Ready", 80.0) if w.vintage.upper().startswith("NV") else ("❔ Unknown", 60.0)
    if THIS_YEAR < w.drink_from - 2:
        return "🔵 Too young", 20.0
    if THIS_YEAR < w.drink_from:
        return "🟡 A bit young", 55.0
    if THIS_YEAR <= w.drink_to:
        return "🟢 Drinking well", 100.0
    if THIS_YEAR <= w.drink_to + 2:
        return "🟠 Fading", 45.0
    return "🔴 Past peak", 10.0


def value_score(markup: float, w: Wine, drink_pts: float, weights: dict) -> float:
    """Blend the factors into one 0-100 score using the user's weights."""
    parts = {
        "Price vs. retail": 100 * _clamp01((4.0 - markup) / (4.0 - 1.5)),  # 1.5x → 100, 4x → 0
        "Critic standing": 100 * _clamp01((w.critic_score - 84) / 12) if w.critic_score else 50.0,  # 84 → 0, 96 → 100
        "Drinking window": drink_pts,
        "Vintage quality": 25 * (w.vintage_quality - 1) if w.vintage_quality else 60.0,
        "My taste": 25 * (w.taste_match - 1) if w.taste_match else None,  # None: no preferences given
    }
    used = {k: v for k, v in parts.items() if v is not None}
    total = sum(weights.get(k, 0) for k in used) or 1
    return sum(v * weights.get(k, 0) for k, v in used.items()) / total


def rating(score: float) -> str:
    if score >= 75:
        return "💎 Great buy"
    if score >= 60:
        return "👍 Good buy"
    if score >= 45:
        return "😐 Fair"
    return "🚩 Skip"


def to_frame(result: WineList, weights: dict) -> pd.DataFrame:
    rows = []
    for w in result.wines:
        if w.list_price <= 0:
            continue
        status, drink_pts = drink_status(w)
        if w.est_retail_price > 0:
            markup = w.list_price / w.est_retail_price
            score = value_score(markup, w, drink_pts, weights)
        else:  # the evaluation step didn't return this wine; still list it
            markup = score = None
        rows.append(
            {
                "Score": round(score) if score is not None else None,
                "Rating": rating(score) if score is not None else "❔ Not evaluated",
                "Find": "⭐" if w.underrated else "",
                "Wine": " ".join(x for x in (w.producer, w.name) if x).strip(),
                "Vintage": w.vintage,
                "Region": w.region,
                "Grape": w.grape,
                "Color": w.color,
                "Size": size_label(size_ml(w.format)),
                "List": w.list_price,
                "Est. Retail": w.est_retail_price or None,
                "Markup": round(markup, 2) if markup else None,
                "Critics": w.critic_score or None,
                "Based on": w.quality_basis,
                "Taste": "♥" * w.taste_match if w.taste_match else "",
                "Vintage ★": "★" * w.vintage_quality if w.vintage_quality else "",
                "Drink": f"{w.drink_from}–{w.drink_to}" if w.drink_from and w.drink_to else "",
                "Readiness": status,
                "Confidence": w.retail_confidence,
                "Note": w.note,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["Score", "Markup"], ascending=[False, True], na_position="last"
        ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------- UI
st.title("🍷 Wine List Value Scanner")
st.caption(
    "Snap or upload a restaurant wine list (photos or PDF, multiple pages OK) to find the best "
    "values and under-the-radar bottles. Retail prices are AI estimates. Use them as a guide, not gospel."
)

api_key = get_secret("GEMINI_API_KEY")
if not api_key:
    st.error(
        "No `GEMINI_API_KEY` found. On Streamlit Community Cloud, add it under **App → Settings → Secrets**:\n\n"
        '```toml\nGEMINI_API_KEY = "your-key-here"\n```\n'
        "Get a key at https://aistudio.google.com/apikey"
    )
    st.stop()

try:
    client, model, backend = connect(
        api_key,
        get_secret("GEMINI_MODEL"),
        (get_secret("GEMINI_USE_VERTEX", "") or "").lower() in ("1", "true", "yes"),
    )
except Exception as e:
    st.error(
        "Couldn't connect to Gemini with this API key. Common causes: a typo or extra quotes in "
        "the secret, the key was deleted, or the Generative Language API isn't enabled for its project. "
        "Create a fresh key at https://aistudio.google.com/apikey and paste it into Secrets.\n\n"
        f"The key this app is using is **{len(api_key)} characters** long and ends in **…{api_key[-4:]}**. "
        "Compare that with the key ending shown on the AI Studio API Keys page."
    )
    with st.expander("Technical details"):
        st.code(str(e))
    st.stop()

# --- sidebar: the diner's taste and search setting
with st.sidebar:
    st.header("🍷 My taste")
    default_prefs = st.query_params.get("taste") or get_secret("MY_PREFERENCES", "") or ""
    prefs = st.text_area(
        "What do you like and dislike?",
        value=default_prefs,
        height=160,
        placeholder="e.g. Love Northern Rhône Syrah, Barolo and dry Riesling. Prefer old-world, "
        "earthy, high-acid wines. Avoid oaky buttery Chardonnay and jammy Zinfandel.",
        help="Used to give each wine a 'My taste' match (1–5 ♥) that feeds the Value Score.",
    )
    if prefs.strip() != (st.query_params.get("taste") or "").strip():
        if prefs.strip():
            st.query_params["taste"] = prefs.strip()  # bookmark the page to keep your taste
        elif "taste" in st.query_params:
            del st.query_params["taste"]
    st.caption("💡 Bookmark this page after typing: your taste is saved in the link.")
    st.divider()
    use_search = st.toggle(
        "🔎 Check reviews online (Google Search)",
        value=True,
        help="Looks up critic reviews, vintage reports and shop prices on the web. More accurate "
        "but slower (1–3 min). Free Gemini keys get a limited number of searches per day; if "
        "they run out, the app falls back to the AI's own knowledge.",
    )

img_types = ["jpg", "jpeg", "png", "webp"] + (["heic", "heif"] if HEIC_OK else [])
tab_upload, tab_camera = st.tabs(["📄 Upload photos / PDF", "📷 Use camera"])
with tab_upload:
    uploads = st.file_uploader(
        "Wine list pages", type=img_types + ["pdf"], accept_multiple_files=True
    )
with tab_camera:
    snap = st.camera_input("Take a photo of the list")

files = list(uploads or []) + ([snap] if snap else [])
files_key = None

if files:
    previews = [f for f in files if not f.name.lower().endswith(".pdf")]
    pdfs = [f.name for f in files if f.name.lower().endswith(".pdf")]
    if pdfs:
        st.write("📄 " + ", ".join(pdfs))
    if previews:
        with st.expander(f"Preview ({len(previews)} image{'s' if len(previews) > 1 else ''})"):
            cols = st.columns(min(len(previews), 4))
            for i, f in enumerate(previews):
                try:
                    cols[i % len(cols)].image(
                        PIL.ImageOps.exif_transpose(PIL.Image.open(f)), width="stretch"
                    )
                except Exception:
                    cols[i % len(cols)].warning(f"Can't preview {f.name}")

    cur_label = st.selectbox(
        "Prices on this list are in",
        list(CURRENCIES),
        key="currency",
        help="Retail prices are estimated for this country, so pick where the restaurant is.",
    )
    with st.expander("🎯 What to look at", expanded=True):
        named = re.findall(r"(?:focus on|only|just|stick to)\s+(?:the\s+)?(red|white|sparkling|champagne|ros[eé]|orange|sweet|dessert)", prefs, re.I)
        alias = {"champagne": "Sparkling", "rose": "Rosé", "rosé": "Rosé", "sweet": "Dessert"}
        pref_colors = list(dict.fromkeys(alias.get(c.lower(), c.title()) for c in named))
        c1, c2, c3 = st.columns([3, 2, 2])
        scope_colors = c1.multiselect(
            "Wine types", COLORS, default=pref_colors or COLORS,
            key="scope_colors_" + "_".join(pref_colors),  # re-defaults when My taste names a type
        )
        max_price = c2.number_input(
            f"Max list price ({CURRENCIES[cur_label][0]}, 0 = any)", min_value=0, value=0, step=25,
            key="scope_max",
        )
        include_small = c3.checkbox(
            "Include half bottles & small formats",
            value=bool(re.search(r"375|half[- ]?bottle|halves|500\s*ml", prefs, re.I)),
            key="scope_small",
        )
        deep_n = st.slider(
            "How many top candidates to research in depth", 10, 100, 40, step=10, key="scope_deep",
            help="Big lists get a quick price & critic check for every wine, then only the most "
            "promising ones get the full treatment (web reviews, drinking window, your taste). "
            "Higher = more thorough but slower and uses more of your free Gemini quota.",
        )
    scope = {"colors": scope_colors, "max_price": max_price, "include_small": include_small, "deep_n": deep_n}
    files_key = (
        tuple((f.name, f.size) for f in files), cur_label, prefs.strip(), use_search,
        tuple(sorted(scope_colors)), max_price, include_small, deep_n,
    )
    if st.button("🔍 Analyze Wine List", type="primary"):
        with st.status("Analyzing…", expanded=False) as status:
            try:
                result, used = analyze(
                    client, model, files, cur_label, prefs, use_search, scope, status
                )
                st.session_state["result"] = result
                st.session_state["files_key"] = files_key
                st.session_state["used_model"] = used
                status.update(label=f"Done, analyzed with {used}", state="complete")
            except Exception as e:
                status.update(label="Analysis failed", state="error")
                msg = str(e)
                if msg.startswith("ALL_BUSY"):
                    st.error(
                        "Google's Gemini servers are overloaded right now (this happens to everyone on "
                        "busy days, especially on the free plan). Wait a minute or two and press "
                        "**Analyze** again."
                    )
                    with st.expander("Technical details"):
                        st.code(msg.removeprefix("ALL_BUSY").strip())
                elif "API key" in msg or "401" in msg or "403" in msg:
                    st.error(f"Gemini rejected the API key. Check `GEMINI_API_KEY`.\n\nDetails: {msg}")
                else:
                    st.error(f"Analysis failed: {msg}")
                st.stop()

result = st.session_state.get("result")
current_key = files_key
if result and files and st.session_state.get("files_key") != current_key:
    if st.session_state.get("files_key", (None,))[0] == current_key[0]:
        st.info("Settings changed. Press **Analyze Wine List** again to update the results.")
elif result and files:
    with st.expander("⚖️ What matters to you? Adjust how the Value Score is weighted"):
        has_taste = any(w.taste_match for w in result.wines)
        factors = {k: v for k, v in DEFAULT_WEIGHTS.items() if has_taste or k != "My taste"}
        wcols = st.columns(len(factors))
        weights = {
            k: wc.slider(k, 0, 100, v, step=5, key=f"w_{k}")
            for wc, (k, v) in zip(wcols, factors.items())
        }
        st.caption(
            "Price vs. retail: restaurant markup over shop price · Critic standing: typical "
            "professional score · Drinking window: is it ready to drink now · Vintage quality: "
            "how good the year was in that region · My taste: fit with the preferences in the sidebar."
        )
    df = to_frame(result, weights)
    cur = result.currency or "$"
    if df.empty:
        st.warning("No priced wines were found. Try a sharper, well-lit photo with the prices visible.")
        st.stop()

    st.subheader(f"📊 Value Analysis{' — ' + result.restaurant if result.restaurant else ''}")
    if result.total_listed and result.researched < len(result.wines):
        st.caption(
            f"📋 {result.total_listed:,} wines on the list · {len(result.wines):,} matched your choices · "
            f"🔬 top {result.researched} researched in depth. The rest have a quick price & critic "
            "estimate only ('Based on: Quick estimate')."
        )

    # --- filters
    f1, f2, f3 = st.columns([2, 2, 2])
    colors = sorted(df["Color"].dropna().unique())
    pick_colors = f1.multiselect("Color", colors, default=colors)
    lo, hi = float(df["List"].min()), float(df["List"].max())
    budget = f2.slider(f"Max list price ({cur})", lo, hi, hi) if hi > lo else hi
    sizes = sorted(df["Size"].unique(), key=lambda x: (x != "Standard (750ml)", not x.startswith("Large"), x))
    wants_small = bool(st.session_state.get("scope_small"))
    default_sizes = [x for x in sizes if x.startswith(("Standard", "Large")) or (wants_small and x.startswith(("Small", "Half")))]
    pick_sizes = f3.multiselect("Bottle size", sizes, default=default_sizes or sizes)

    view = df[df["Color"].isin(pick_colors) & (df["List"] <= budget) & df["Size"].isin(pick_sizes)]

    near_one = df["Markup"].between(0.97, 1.03).mean()
    if near_one > 0.4:
        st.warning(
            f"{near_one:.0%} of retail estimates came back equal to the list price, which isn't "
            "realistic. Treat the markups as unreliable and try **Analyze** again."
        )

    # --- headline metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Wines found", len(df))
    m2.metric("Median markup", f"{df['Markup'].median():.1f}×")
    m3.metric("Great buys", int((df["Score"] >= 75).sum()))
    m4.metric("Drinking well now", int((df["Readiness"] == "🟢 Drinking well").sum()))

    # --- top picks
    picks = view[
        (view["Confidence"] != "Low") & (view["Score"] >= 60) & (view["Based on"] != "Quick estimate")
    ].head(3)
    if not picks.empty:
        st.markdown("#### 🏆 Sommelier's picks")
        for col, (_, w) in zip(st.columns(len(picks)), picks.iterrows()):
            with col.container(border=True):
                st.markdown(f"**{w['Wine']} {w['Vintage']}**  \n{w['Region']}")
                st.markdown(
                    f"**Value Score {w['Score']}** · {cur}{w['List']:,.0f} on list · "
                    f"~{cur}{w['Est. Retail']:,.0f} retail ({w['Markup']:.1f}×)"
                )
                details = [
                    f"~{w['Critics']:.0f} pts" if pd.notna(w["Critics"]) else "",
                    f"vintage {w['Vintage ★']}" if w["Vintage ★"] else "",
                    f"{w['Readiness']} ({w['Drink']})" if w["Drink"] else w["Readiness"],
                    f"taste {w['Taste']}" if w["Taste"] else "",
                ]
                st.markdown(" · ".join(d for d in details if d))
                st.caption(("⭐ Underrated find. " if w["Find"] else "") + w["Note"])

    # --- full table
    st.markdown("#### All wines (best overall value first)")
    st.dataframe(
        view,
        hide_index=True,
        width="stretch",
        column_config={
            "Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
            "Critics": st.column_config.NumberColumn(format="%d pts"),
            "Based on": st.column_config.TextColumn(
                help="Web reviews = found via Google Search · Quick estimate = big-list first pass only · This wine = known reviews · Producer = producer's usual level · "
                "Region/style = typical for that appellation, grape and price (least certain)"
            ),
            "List": st.column_config.NumberColumn(format=f"{cur}%.0f"),
            "Est. Retail": st.column_config.NumberColumn(format=f"{cur}%.0f"),
            "Markup": st.column_config.NumberColumn(format="%.2f×"),
            "Note": st.column_config.TextColumn(width="large"),
        },
    )
    if result.sources:
        with st.expander(f"🔎 Web sources checked ({len(result.sources)})"):
            st.markdown("\n".join(f"- [{t}]({u})" for t, u in result.sources[:60]))
    st.download_button(
        "⬇️ Download CSV", view.to_csv(index=False), "wine_list_analysis.csv", "text/csv"
    )
    st.caption(
        "Value Score (0–100) blends price vs. retail, critic standing, drinking window, vintage "
        "quality and your taste using the weights above. 💎 75+ · 👍 60–74 · 😐 45–59 · 🚩 under 45. "
        + (
            "Scores marked 'Web reviews' were checked with Google Search; the rest are the AI's own "
            "knowledge of reviews and vintage charts. "
            if result.sources
            else "Critic scores, vintage ratings and drinking windows are the AI's recollection of "
            "published reviews and vintage charts, not a live lookup. "
        )
        + "'Based on' shows how specific each estimate is. Typical restaurant markup is 2.5–3×. "
        "Low-confidence price estimates are excluded from picks. "
        f"Model: {st.session_state.get('used_model', model)} via {backend}"
    )
