# LedgerLens — Offline On-Device Invoice & Receipt Extractor
Project spec, v1. Target: Tether "AI Research Engineer — Model Compression & Quantization" (and partial coverage of the Multi-Modal & Vision role's OSS requirement).

## 1. The pitch (use this framing everywhere)
Freelancers and micro-merchants — especially in emerging markets, on mid-range Android phones — need to digitize receipts and invoices. Financial documents should never leave the device, and connectivity can't be assumed. So the extractor must run **offline, on-device, private** — which makes model compression a *requirement of the use case*, not an academic exercise. Input: a photo of a receipt/invoice. Output: structured JSON (vendor, date, currency, total, tax, line items) in ~1–2 seconds on a phone.

## 2. JD coverage map (Tether Compression role)
| Their requirement | Project component |
|---|---|
| Knowledge distillation (teacher→student) | VLM teacher labels/corrects → 2B student SFT |
| PTQ hands-on | AWQ/GPTQ int8 + int4 on student |
| QAT hands-on | torchao QAT on student (LoRA-QAT to bound compute) |
| Pruning | Structured attention-head/layer pruning + recovery fine-tune |
| Trade-off analysis (size/latency/memory vs accuracy) | The Pareto grid — headline deliverable |
| Low-latency AI on smartphones | llama.cpp/MLC build running on candidate's Android, demo video |
| Dataset curation | CORD/SROIE + synthetic invoice pipeline, documented |
| Documentation/reproducibility | Repo + writeup + HF model card |

## 3. Models
- **Teacher:** Qwen2.5-VL-7B-Instruct (strong doc understanding; runs on 24 GB in 4-bit for inference)
- **Student:** Qwen2-VL-2B-Instruct (primary). Fallback/comparison: SmolVLM-2B or Moondream2 if Qwen mobile runtime support disappoints
- **Runtime target:** llama.cpp multimodal (Qwen2-VL supported) → Android via termux/JNI, or MLC-LLM. Fallback demo: laptop CPU-only (still "edge", still honest — label it as such)

## 4. Data
- **CORD** (~1k receipts, line-item annotations) — train/eval
- **SROIE** (973 receipts; company/date/address/total) — eval + train supplement
- **Synthetic invoices:** HTML/CSS templates × randomized vendors/currencies/layouts → render → augment (perspective warp, blur, shadows, low-light) to mimic phone photos. ~2–5k images. Generation script in repo (this is the "data curation" evidence)
- **Personal holdout:** ~50 real receipts photographed by the candidate — the honest "does it actually work" set. Redact/exclude from publication if privacy-sensitive
- License note in repo: CORD/SROIE are research-use; model weights trained on them labeled accordingly

## 5. Evaluation protocol (fixed BEFORE training — this is the research discipline signal)
- **Field-level metrics:** exact-match and F1 per field (vendor, date, total, currency, tax); numeric tolerance rule for totals (normalize separators)
- **Structural:** JSON validity rate (parses + schema-conforms)
- **Line items:** F1 over (description, qty, price) tuples, Hungarian-matched
- **Efficiency:** model size (GB), peak RAM, tokens/sec, end-to-end latency per receipt — measured on (a) GPU, (b) laptop CPU, (c) phone
- **Report everything as Pareto plots:** accuracy vs bits, accuracy vs size, accuracy vs latency; every compression variant is one point

## 6. Weekend-by-weekend plan
**Weekend 1 — Baselines + harness.** Eval harness first (metrics above, one command, CSV out). Run teacher-7B and stock student-2B on CORD/SROIE eval splits. Deliverable: baseline table. *If the stock 2B is already near the 7B, pivot the story to "how small can we go" and add a ~1B target.*

**Weekend 2 — Distillation.** Teacher generates structured outputs + corrections on train sets and synthetic invoices; SFT the student (LoRA, then merge) on teacher outputs blended with ground truth. Deliverable: distilled-2B point on the plot; ablation of teacher-data vs ground-truth-only.

**Weekend 3 — Compression grid.** PTQ: int8 and int4 via AWQ/GPTQ. QAT: torchao on the LoRA-merged student. Pruning: structured head/layer pruning (e.g., 10–25%) + short recovery fine-tune. Combos: prune→QAT, distill→int4. Deliverable: the full Pareto grid, ~8–12 points.

**Weekend 4 — Ship.** Best small variant → GGUF → llama.cpp on Android; measure on-phone latency/RAM; record demo video (photo → JSON, airplane mode ON — the airplane-mode toggle is the money shot). Publish: GitHub repo (harness, data-gen, training scripts), HF model + card, writeup (~1,500 words, honest about what didn't work). 

## 7. Deliverables checklist
- [ ] GitHub repo (MIT/Apache, clean README, one-command eval)
- [ ] HuggingFace: distilled+quantized model with model card (starts the public OSS record the Vision role demands)
- [ ] Writeup with Pareto plots + failure analysis
- [ ] 60–90s demo video: airplane mode, photo, JSON, timer visible
- [ ] Baseline-vs-final table in README top

## 8. Budget & hardware
Single 24 GB GPU (own 3090/4090, or rented A10/A5000 ≈ $0.30–0.80/hr; expect $30–80 total). Phone: candidate's own Android. Total calendar: 4 weekends.

## 9. Risks & fallbacks
- Mobile VLM runtime flakiness → laptop CPU demo, labeled honestly
- QAT compute heavy → QAT only on the 2B with LoRA; PTQ carries the grid breadth
- CORD/SROIE ceiling effects → lean on synthetic hard set + personal holdout for differentiation
- Time overrun → Weekends 1–3 alone are already application-worthy; the phone demo is the cherry, not the cake

## 10. Truthfulness rules (inherited from master tracker — non-negotiable)
- CV entry only AFTER the artifact exists, under Independent Projects, e.g.: "Built LedgerLens, an offline on-device receipt/invoice extractor: distilled a vision-language teacher into a quantized (QAT/PTQ), pruned 2B student running on Android; published model, code, and accuracy–latency trade-off analysis."
- Interview framing: "hands-on at small scale — here are my measurements." Never imply production deployment or real users unless that becomes true.
- All numbers on the CV/letter must come from the eval harness, reproducibly.
- The Tether letter gets rewritten only once the repo link is live.
