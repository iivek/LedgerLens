"""Synthetic invoice/receipt generator: Faker content -> Jinja2 HTML -> PNG -> augment.

Gold labels are constructed *before* rendering, so they are exact by construction.
Displayed number strings deliberately vary in separator style ("1,234.56",
"1.234,56", "1 234,56", zero-decimal JPY/IDR) to train separator robustness —
gold always stores canonical floats.

Rendering backend: Playwright/Chromium if installed (`uv sync --extra render` +
`playwright install chromium`), else a system Chrome/Chromium binary in headless
screenshot mode.
"""

from __future__ import annotations

import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from faker import Faker
from jinja2 import Environment, FileSystemLoader
from PIL import Image

from ..unified import write_split
from .augment import augment_photo

TEMPLATES_DIR = Path(__file__).parent / "templates"

# (code, symbol, decimals, separator style)  — style: us "1,234.56", eu "1.234,56", space "1 234,56"
CURRENCIES = [
    ("USD", "$", 2, "us"), ("EUR", "€", 2, "eu"), ("GBP", "£", 2, "us"),
    ("IDR", "Rp", 0, "eu"), ("INR", "₹", 2, "us"), ("MYR", "RM", 2, "us"),
    ("PHP", "₱", 2, "us"), ("VND", "₫", 0, "eu"), ("TRY", "₺", 2, "eu"),
    ("BRL", "R$", 2, "eu"), ("NGN", "₦", 2, "us"), ("KES", "KSh", 2, "us"),
]
FONTS_RECEIPT = ["Courier New", "Menlo", "monospace"]
FONTS_INVOICE = ["Helvetica", "Arial", "Georgia", "Verdana", "Trebuchet MS"]
ACCENTS = ["#1a4d8f", "#7a1f1f", "#14532d", "#4a3b8f", "#8a5a00", "#222222"]
ITEM_NOUNS = ["Coffee", "Nasi Goreng", "Notebook", "USB-C Cable", "T-Shirt", "Consulting (hr)",
              "Delivery Fee", "Laundry kg", "Phone Case", "Chicken Rice", "Design Services",
              "Screen Repair", "Bottled Water", "Fabric (m)", "Printing B/W", "Airtime Top-up",
              "Data Bundle 5GB", "Motorbike Parking", "Hosting (mo)", "Photo Prints"]
FOOTERS = ["THANK YOU! PLEASE COME AGAIN", "Goods sold are not returnable",
           "Payment due within 14 days", "Terima kasih!", "Keep receipt for warranty",
           "* prices include service charge *"]
PAYMENTS = ["CASH", "VISA ****{n}", "MASTERCARD ****{n}", "QRIS", "M-PESA", "GOPAY"]


def fmt_money(value: float, decimals: int, style: str) -> str:
    q = round(value, decimals)
    ipart = int(abs(q))
    dec = abs(q) - ipart
    groups = f"{ipart:,}"
    if style == "eu":
        groups = groups.replace(",", ".")
        dsep = ","
    elif style == "space":
        groups = groups.replace(",", " ")
        dsep = ","
    else:
        dsep = "."
    sign = "-" if q < 0 else ""
    if decimals == 0:
        return sign + groups
    return sign + groups + dsep + f"{dec:.{decimals}f}"[2:]


@dataclass
class Renderer:
    backend: str  # "playwright" | "chrome"
    chrome_path: str | None = None
    _pw: object = None
    _browser: object = None

    @classmethod
    def create(cls) -> "Renderer":
        try:
            from playwright.sync_api import sync_playwright
            pw = sync_playwright().start()
            browser = pw.chromium.launch()
            return cls(backend="playwright", _pw=pw, _browser=browser)
        except Exception:
            pass
        for cand in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                     "/Applications/Chromium.app/Contents/MacOS/Chromium",
                     shutil.which("google-chrome") or "", shutil.which("chromium") or ""):
            if cand and Path(cand).exists():
                return cls(backend="chrome", chrome_path=cand)
        raise SystemExit(
            "No HTML renderer found. Either `uv sync --extra render && uv run playwright "
            "install chromium`, or install Google Chrome."
        )

    def render(self, html: str, out_png: Path, width: int) -> None:
        if self.backend == "playwright":
            page = self._browser.new_page(viewport={"width": width, "height": 800})
            page.set_content(html)
            page.screenshot(path=str(out_png), full_page=True)
            page.close()
        else:
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
                f.write(html)
                tmp = f.name
            subprocess.run(
                [self.chrome_path, "--headless=new", f"--screenshot={out_png}",
                 f"--window-size={width},2400", "--default-background-color=FFFFFFFF",
                 "--hide-scrollbars", "--disable-gpu", f"file://{tmp}"],
                check=True, capture_output=True,
            )
            Path(tmp).unlink(missing_ok=True)

    def close(self) -> None:
        if self.backend == "playwright":
            self._browser.close()
            self._pw.stop()


def _trim_white(png_path: Path) -> None:
    """Crop trailing white space below the document (chrome fixed-window renders)."""
    img = np.array(Image.open(png_path).convert("RGB"))
    nonwhite = np.where((img < 245).any(axis=2).any(axis=1))[0]
    if len(nonwhite):
        bottom = min(int(nonwhite[-1]) + 20, img.shape[0])
        Image.fromarray(img[:bottom]).save(png_path)


def make_sample(rng: random.Random, faker: Faker) -> tuple[str, dict, str, int]:
    """Returns (template_name, jinja context, gold-free) and gold dict embedded in ctx."""
    code, symbol, decimals, style = rng.choice(CURRENCIES)
    is_invoice = rng.random() < 0.4
    n_items = rng.randint(1, 8 if not is_invoice else 6)
    scale = {"USD": 1, "EUR": 1, "GBP": 1, "IDR": 15000, "INR": 80, "MYR": 4.5,
             "PHP": 55, "VND": 24000, "TRY": 30, "BRL": 5, "NGN": 1400, "KES": 130}[code]
    items, gold_items, subtotal = [], [], 0.0
    for _ in range(n_items):
        qty = float(rng.choice([1, 1, 1, 2, 2, 3, 4, 5])) if rng.random() < 0.85 else round(rng.uniform(0.2, 3.0), 2)
        unit = round(rng.uniform(0.5, 120) * scale, decimals)
        total = round(qty * unit, decimals)
        desc = rng.choice(ITEM_NOUNS)
        subtotal += total
        items.append({
            "description": desc,
            "qty_str": f"{qty:g}", "unit_price_str": fmt_money(unit, decimals, style),
            "total_str": fmt_money(total, decimals, style),
        })
        gold_items.append({"description": desc, "qty": qty, "unit_price": unit, "total": total})
    subtotal = round(subtotal, decimals)
    tax_rate = rng.choice([0.0, 0.05, 0.06, 0.07, 0.08, 0.10, 0.11, 0.18, 0.20])
    tax = round(subtotal * tax_rate, decimals)
    total = round(subtotal + tax, decimals)

    vendor_name = rng.choice([faker.company(), f"{faker.last_name().upper()} {rng.choice(['MART','CAFE','STORE','TRADING','SERVICES'])}"])
    date = faker.date_between(start_date="-3y", end_date="today")
    date_str = date.strftime(rng.choice(["%d/%m/%Y", "%Y-%m-%d", "%d %b %Y", "%d-%m-%y", "%b %d, %Y"]))
    ctx = {
        "vendor": {"name": vendor_name, "address": faker.address().replace("\n", ", "),
                   "phone": faker.phone_number(), "email": faker.company_email()},
        "customer": {"name": faker.name(), "address": faker.address().replace("\n", ", ")},
        "date_str": date_str, "time_str": faker.time(pattern="%H:%M"),
        "due_str": faker.date_this_year().strftime("%d/%m/%Y"),
        "receipt_no": faker.bothify(text=rng.choice(["INV-####", "R######", "??-#####"])).upper(),
        "cashier": faker.first_name().upper(),
        "items": items,
        "subtotal_str": fmt_money(subtotal, decimals, style),
        "tax_rate_str": f"{tax_rate * 100:g}%",
        "tax_str": fmt_money(tax, decimals, style),
        "total_str": fmt_money(total, decimals, style),
        "tendered_str": fmt_money(total, decimals, style),
        "currency_code": code, "currency_symbol": symbol,
        "payment_method": rng.choice(PAYMENTS).format(n=rng.randint(1000, 9999)),
        "footer": rng.choice(FOOTERS),
        "font": rng.choice(FONTS_INVOICE if is_invoice else FONTS_RECEIPT),
        "font_size": rng.randint(12, 15) if is_invoice else rng.randint(13, 16),
        "accent": rng.choice(ACCENTS),
    }
    gold = {
        "vendor": vendor_name, "date": date.strftime("%Y-%m-%d"), "currency": code,
        "total": total, "tax": tax if tax_rate > 0 else None, "line_items": gold_items,
    }
    template = "invoice_a4.html" if is_invoice else "receipt_thermal.html"
    width = 800 if is_invoice else 400
    ctx["__gold__"] = gold
    return template, ctx, code, width


def generate_dataset(n: int, out_dir: Path, seed: int = 17, augment: bool = True) -> None:
    rng = random.Random(seed)
    Faker.seed(seed)
    faker = Faker(["en_US", "en_GB", "id_ID", "en_IN", "pt_BR", "tr_TR"])
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
    renderer = Renderer.create()
    img_dir = out_dir / "images" / "synth"
    img_dir.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        for i in range(n):
            template, ctx, _code, width = make_sample(rng, faker)
            gold = ctx.pop("__gold__")
            sample_id = f"synth_{i:05d}"
            png = img_dir / f"{sample_id}.png"
            renderer.render(env.get_template(template).render(**ctx), png, width)
            _trim_white(png)
            if augment:
                img = cv2.imread(str(png))
                cv2.imwrite(str(png), augment_photo(img, rng))
            records.append({"id": sample_id, "image": str(png.relative_to(out_dir)),
                            "source": "synthetic", "gold": gold})
            if (i + 1) % 50 == 0:
                print(f"  rendered {i + 1}/{n}")
    finally:
        renderer.close()
    # 90/10 train/eval split, deterministic
    rng.shuffle(records)
    cut = max(1, int(len(records) * 0.9)) if len(records) > 1 else 1
    write_split(records[:cut], out_dir / "synth_train.jsonl")
    if records[cut:]:
        write_split(records[cut:], out_dir / "synth_eval.jsonl")
    print(f"wrote {cut} train / {len(records) - cut} eval samples under {out_dir}")
