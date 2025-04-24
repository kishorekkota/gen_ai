"""FastAPI app for document upload, OCR classification, metadata extraction,
   **and basic forgery heuristics with dynamic bank‑logo hashing**.

🔄 **NEW FEATURES**
• Automatically computes MD5 hashes for every logo image placed in the `logos/` folder (PNG/JPG).
• Stores them in `BANK_LOGO_HASHES` at startup and logs the mapping.
• Adds `/logo-hashes` endpoint to return the current dictionary as JSON for easy copy‑paste.

Place files like `chase.png`, `boa.png`, `discover.png`, etc., into `logos/`.
Each image should be the corporate logo on a white background, at least 250 × 250 px.
"""

from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from llm_classifier import classify_document_with_gemini
from extractor import extract_metadata
import os, hashlib, logging, cv2
from PIL import Image, ImageChops
import pikepdf

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ─────────────────────────────── config ──────────────────────────────
MAX_MB   = 10
LOGO_DIR = Path("logos")  # put logo images here
CACHE_DIR = Path("/tmp/.cache_doctr"); CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(CACHE_DIR))

# ─────────────────────── dynamic logo hashing ────────────────────────

def _logo_hash(img: Image.Image) -> str:
    """Return MD5 of top‑left 250×250 grayscale, down‑sampled to 64×64."""
    crop = img.crop((0, 0, 250, 250)).convert("L").resize((64, 64))
    return hashlib.md5(crop.tobytes()).hexdigest()


def _build_logo_hashes(directory: Path) -> dict:
    mapping = {}
    if not directory.exists():
        logging.warning("Logo directory %s does not exist", directory)
        return mapping
    for img_path in directory.glob("*.[pj][pn]g"):
        try:
            digest = _logo_hash(Image.open(img_path))
            bank_name = img_path.stem.replace("_", " ").title()
            mapping[bank_name] = digest
        except Exception as e:
            logging.error("Failed hashing %s: %s", img_path, e)
    return mapping

BANK_LOGO_HASHES = _build_logo_hashes(LOGO_DIR)
logging.info("Loaded %d logo hashes", len(BANK_LOGO_HASHES))

# ─────────────────────── app & OCR predictor ─────────────────────────
app = FastAPI()
templates = Jinja2Templates(directory="templates")
ocr = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True)

# ───────────────────── forgery‑detection helpers ─────────────────────

def _logo_hash_match(img: Image.Image) -> bool:
    return _logo_hash(img) in BANK_LOGO_HASHES.values()


def _ela_score(img_path: Path) -> float:
    img = Image.open(img_path).convert("RGB")
    tmp_path = img_path.with_suffix(".ela.jpg")
    img.save(tmp_path, "JPEG", quality=95)
    ela = ImageChops.difference(img, Image.open(tmp_path))
    std = float(cv2.cvtColor(cv2.imread(str(tmp_path)), cv2.COLOR_BGR2GRAY).std())
    tmp_path.unlink(missing_ok=True)
    return std


def detect_forgery(tmp_path: Path, ext: str, doc_type: str, metadata: dict) -> dict:
    issues = []

    # 1️⃣ PDF metadata sanity
    if ext == ".pdf":
        with pikepdf.open(tmp_path) as pdf:
            info = pdf.docinfo or {}
            creation = str(info.get("/CreationDate", ""))
            if len(creation) >= 6 and creation[2:6] > "2030":
                issues.append("Future creation date")
            if str(info.get("/Producer", "")).lower().startswith("word"):
                issues.append("Producer = Word")

    # 2️⃣ Logo hash (Bank Statement images)    
    if doc_type == "Bank Statement" and ext in {".jpg", ".jpeg", ".png"}:
        if not _logo_hash_match(Image.open(tmp_path)):
            issues.append("Bank logo hash mismatch")

    # 3️⃣ ELA noise
    if ext in {".jpg", ".jpeg", ".png"}:
        if _ela_score(tmp_path) > 25:
            issues.append("High ELA noise")

    # 4️⃣ Routing sanity
    if doc_type == "Bank Statement" and metadata.get("routing_number", "").startswith("0") is False:
        issues.append("Routing number unusual")

    return {"is_forged": bool(issues), "issues": issues}

# ─────────────────────────── OCR helper ──────────────────────────────

def _extract_text(result) -> str:
    doc_json = result.export()
    return " ".join(w["value"]
                     for p in doc_json["pages"]
                     for b in p["blocks"]
                     for l in b["lines"]
                     for w in l["words"])

# ───────────────────────────── routes ────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/logo-hashes")
async def logo_hashes():
    """Return dictionary of bank → MD5 logo hashes."""
    return JSONResponse(BANK_LOGO_HASHES)


@app.post("/upload/")
async def handle_upload(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "Missing filename")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".pdf"}:
        raise HTTPException(415, "Unsupported type")

    content = await file.read()
    if len(content) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, "File too large")

    tmp_path = Path(f"temp{ext}")
    try:
        tmp_path.write_bytes(content)
        doc = DocumentFile.from_pdf(str(tmp_path)) if ext == ".pdf" else DocumentFile.from_images([str(tmp_path)])
        extracted_text = _extract_text(ocr(doc))
    finally:
        pass  # keep temp for forgery check

    # classify
    prompt = (
        "You are a strict document‑type classifier. Return **one** label from: "
        "[Bank Statement, School Enrollment, Payslip, W‑2, 1099, Cell Phone Bill].\n\n" + extracted_text[:4000]
    )
    doc_type = classify_document_with_gemini(prompt)

    metadata = extract_metadata(extracted_text)
    forgery  = detect_forgery(tmp_path, ext, doc_type, metadata)
    valid    = bool(doc_type and metadata) and not forgery["is_forged"]

    tmp_path.unlink(missing_ok=True)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": doc_type,
        "metadata": metadata,
        "forgery": forgery,
        "valid": valid,
    })
