import io
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import pandas as pd
import PIL.Image
import PIL.ImageOps
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

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
DEFAULT_WEIGHTS = {"Price vs. retail": 35, "Critic standing": 30, "Drinking window": 20, "Vintage quality": 15}
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
    format: str = Field(description="Bottle size, e.g. '750ml'; 'Glass' if it is a by-the-glass price")
    list_price: float = Field(description="Price on the wine list, number only")


class ListedWines(BaseModel):
    restaurant: str = Field(description="Restaurant name if visible, else ''")
    wines: list[ListedWine]


class Evaluation(BaseModel):
    id: int = Field(description="The id number given for the wine")
    est_retail_price: float = Field(description="Typical retail shop price for this bottle and vintage, number only")
    retail_confidence: str = Field(description="High, Medium or Low")
    critic_score: int = Field(description="Estimated professional critic score, 100-point scale. Never 0")
    quality_basis: str = Field(description="Exactly one of: 'This wine', 'Producer', 'Region/style'")
    vintage_quality: int = Field(description="Vintage quality for this region, 1 (poor) to 5 (outstanding); 0 only for NV")
    drink_from: int = Field(description="First year of the drinking window. Never 0")
    drink_to: int = Field(description="Last year of the drinking window. Never 0")
    underrated: bool = Field(description="True only for genuine sommelier finds")
    note: str = Field(description="One short, specific sentence")


class Evaluations(BaseModel):
    evaluations: list[Evaluation]


class Wine(ListedWine):
    est_retail_price: float = 0
    retail_confidence: str = "Low"
    critic_score: int = 0
    quality_basis: str = ""
    vintage_quality: int = 0
    drink_from: int = 0
    drink_to: int = 0
    underrated: bool = False
    note: str = ""


class WineList(BaseModel):
    currency: str
    restaurant: str
    wines: list[Wine]


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
        f"\nIMPORTANT: All list prices are in {name}, even if no currency symbol is printed. "
        f"Estimate retail prices as a shopper in {market} would pay, in {name}.\n"
    )


EXTRACT_PROMPT = """You are reading a restaurant wine list (all pages/images provided).

Extract EVERY wine on it. For each wine, record producer, name, vintage, region, grape,
color, format and list price exactly as printed (fill region/grape from your knowledge
if the list only implies them, e.g. by its section heading). If both a glass and a
bottle price are shown, use the bottle price. Skip beers, spirits, cocktails and
other non-wine items. Do not skip any wine, however long the list."""

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
- note: one short, specific sentence weighing price, quality and readiness.

Wines (id | wine | list price):
"""

EVAL_PROMPT = EVAL_TEMPLATE.replace("{year}", str(THIS_YEAR))


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


def to_parts(files) -> list[types.Part]:
    parts = []
    for f in files:
        data = f.getvalue()
        if f.name.lower().endswith(".pdf"):
            # Send the PDF itself — Gemini reads layout and scanned pages, unlike text extraction.
            parts.append(types.Part.from_bytes(data=data, mime_type="application/pdf"))
        else:
            parts.append(prepare_image(data))
    return parts


EVAL_BATCH = 15  # wines per evaluation request: small enough that every wine gets attention


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


def analyze(client, model: str, parts, cur_label: str, status) -> tuple[WineList, str]:
    """Step 1: read the list. Step 2: evaluate it in batches (in parallel)."""
    models = list(dict.fromkeys([model, *FALLBACK_MODELS, *EXTRA_ANALYSIS_FALLBACKS]))
    sym = CURRENCIES[cur_label][0]

    status.update(label="Step 1 of 2: reading the wine list… (20–60 s)")
    listed, used = with_fallback(client, models, [*parts, EXTRACT_PROMPT], ListedWines, status.write)
    wines = [w for w in listed.wines if w.list_price > 0]
    if not wines:
        return WineList(currency=sym, restaurant=listed.restaurant, wines=[]), used
    status.write(f"📋 Found {len(wines)} wines")

    # Keep using whichever model worked, with the others as backups
    models = [used, *[m for m in models if m != used]]
    batches = [list(range(i, min(i + EVAL_BATCH, len(wines)))) for i in range(0, len(wines), EVAL_BATCH)]
    evals: dict[int, Evaluation] = {}
    messages: list[str] = []

    def run_batch(ids):
        lines = "\n".join(f"{i} | {describe(wines[i])} | {sym}{wines[i].list_price:g}" for i in ids)
        prompt = EVAL_PROMPT + lines + "\n" + currency_note(cur_label)
        out, _ = with_fallback(client, models, prompt, Evaluations, messages.append)
        return out.evaluations

    status.update(label=f"Step 2 of 2: evaluating {len(wines)} wines… (30–90 s)")
    done = 0
    with ThreadPoolExecutor(max_workers=3) as pool:  # modest parallelism for free-tier rate limits
        futures = [pool.submit(run_batch, ids) for ids in batches]
        for fut in as_completed(futures):
            try:
                for e in fut.result():
                    evals[e.id] = e
            except Exception as e:  # one failed batch shouldn't lose the whole list
                messages.append(f"⚠️ A batch failed: {str(e)[:200]}")
            done += 1
            status.update(label=f"Step 2 of 2: evaluated {done}/{len(batches)} batches…")

    # Re-ask once for any wines the model skipped
    missing = [i for i in range(len(wines)) if i not in evals]
    if missing and len(missing) < len(wines):
        status.write(f"🔁 Re-checking {len(missing)} wines that were skipped…")
        try:
            for e in run_batch(missing):
                evals[e.id] = e
        except Exception as e:
            messages.append(f"⚠️ Re-check failed: {str(e)[:200]}")
    for m in dict.fromkeys(messages):
        status.write(m)
    if not evals:
        raise RuntimeError("ALL_BUSY\n\n" + "\n".join(messages))

    merged = []
    for i, w in enumerate(wines):
        e = evals.get(i)
        extra = e.model_dump(exclude={"id"}) if e else {}
        merged.append(Wine(**w.model_dump(), **extra))
    return WineList(currency=sym, restaurant=listed.restaurant, wines=merged), used


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
    }
    total = sum(weights.values()) or 1
    return sum(parts[k] * weights.get(k, 0) for k in parts) / total


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
                "Format": w.format,
                "List": w.list_price,
                "Est. Retail": w.est_retail_price or None,
                "Markup": round(markup, 2) if markup else None,
                "Critics": w.critic_score or None,
                "Based on": w.quality_basis,
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

img_types = ["jpg", "jpeg", "png", "webp"] + (["heic", "heif"] if HEIC_OK else [])
tab_upload, tab_camera = st.tabs(["📄 Upload photos / PDF", "📷 Use camera"])
with tab_upload:
    uploads = st.file_uploader(
        "Wine list pages", type=img_types + ["pdf"], accept_multiple_files=True
    )
with tab_camera:
    snap = st.camera_input("Take a photo of the list")

files = list(uploads or []) + ([snap] if snap else [])

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
    files_key = (tuple((f.name, f.size) for f in files), cur_label)
    if st.button("🔍 Analyze Wine List", type="primary"):
        with st.status("Analyzing…", expanded=False) as status:
            try:
                result, used = analyze(client, model, to_parts(files), cur_label, status)
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
if result and files and st.session_state.get("files_key") == (
    tuple((f.name, f.size) for f in files),
    st.session_state.get("currency"),
):
    with st.expander("⚖️ What matters to you? Adjust how the Value Score is weighted"):
        wcols = st.columns(len(DEFAULT_WEIGHTS))
        weights = {
            k: wc.slider(k, 0, 100, v, step=5, key=f"w_{k}")
            for wc, (k, v) in zip(wcols, DEFAULT_WEIGHTS.items())
        }
        st.caption(
            "Price vs. retail: restaurant markup over shop price · Critic standing: typical "
            "professional score · Drinking window: is it ready to drink now · Vintage quality: "
            "how good the year was in that region."
        )
    df = to_frame(result, weights)
    cur = result.currency or "$"
    if df.empty:
        st.warning("No priced wines were found. Try a sharper, well-lit photo with the prices visible.")
        st.stop()

    st.subheader(f"📊 Value Analysis{' — ' + result.restaurant if result.restaurant else ''}")

    # --- filters
    f1, f2, f3 = st.columns([2, 2, 1])
    colors = sorted(df["Color"].dropna().unique())
    pick_colors = f1.multiselect("Color", colors, default=colors)
    lo, hi = float(df["List"].min()), float(df["List"].max())
    budget = f2.slider(f"Max list price ({cur})", lo, hi, hi) if hi > lo else hi
    bottles_only = f3.checkbox("Bottles only", value=True)

    view = df[df["Color"].isin(pick_colors) & (df["List"] <= budget)]
    if bottles_only:
        view = view[~view["Format"].str.contains("glass", case=False, na=False)]

    # --- headline metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Wines found", len(df))
    m2.metric("Median markup", f"{df['Markup'].median():.1f}×")
    m3.metric("Great buys", int((df["Score"] >= 75).sum()))
    m4.metric("Drinking well now", int((df["Readiness"] == "🟢 Drinking well").sum()))

    # --- top picks
    picks = view[(view["Confidence"] != "Low") & (view["Score"] >= 60)].head(3)
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
                help="This wine = known reviews · Producer = producer's usual level · "
                "Region/style = typical for that appellation, grape and price (least certain)"
            ),
            "List": st.column_config.NumberColumn(format=f"{cur}%.0f"),
            "Est. Retail": st.column_config.NumberColumn(format=f"{cur}%.0f"),
            "Markup": st.column_config.NumberColumn(format="%.2f×"),
            "Note": st.column_config.TextColumn(width="large"),
        },
    )
    st.download_button(
        "⬇️ Download CSV", view.to_csv(index=False), "wine_list_analysis.csv", "text/csv"
    )
    st.caption(
        "Value Score (0–100) blends price vs. retail, critic standing, drinking window and vintage "
        "quality using the weights above. 💎 75+ · 👍 60–74 · 😐 45–59 · 🚩 under 45. "
        "Critic scores, vintage ratings and drinking windows are the AI's recollection of published "
        "reviews and vintage charts, not a live lookup. "
        "'Based on' shows how specific each estimate is. Typical restaurant markup is 2.5–3×. "
        "Low-confidence price estimates are excluded from picks. "
        f"Model: {st.session_state.get('used_model', model)} via {backend}"
    )
