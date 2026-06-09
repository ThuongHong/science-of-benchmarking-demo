# Mini Benchmark Evaluator

Demo thực nghiệm cho tutorial **"The Science of Benchmarking — What's Measured,
What's Missing, What's Next"** (NeurIPS 2025).

Thay vì tự chế một benchmark đồ chơi, demo **lấy hai benchmark công khai có thật**
rồi áp **bộ công cụ phân tích phê phán của tutorial** lên chúng — đúng tinh thần
"khoa học về benchmark": không xếp hạng model, mà **soi xem chính phép đo có đáng
tin không**.

| Thành phần | Lựa chọn |
|---|---|
| Benchmark | **MMLU** (kiến thức, trắc nghiệm 4 đáp án) + **GSM8K** (toán reasoning) |
| Model | 6 model: **Gemma-4 E2B/E4B** (ladder cùng họ) + **Qwen3.5-4B / Phi-3.5-mini / Qwen2.5-3B** (panel ~4B khác họ) + **Qwen2.5-Math-7B** (math-specialist, chủ đích lệch hạng để phá thế xếp hạng); tất cả chạy 1 T4 free 4-bit |
| Baseline | random, majority-class (non-LLM) |
| Backend | Hugging Face `transformers`, 4-bit, **greedy → deterministic** |
| Chạy ở đâu | Notebook Kaggle (T4 free) — `notebook/2_run_all.ipynb` |

## Demo liên hệ 6 paper thế nào

| Paper | Liên hệ trong demo |
|---|---|
| **AI and the Everything in the Whole Wide World Benchmark** | MMLU/GSM8K chỉ đo vài năng lực hẹp → điểm cao không chứng minh "AI tổng quát". |
| **Measuring what Matters** | Có baseline random/majority để kiểm tra điểm số có *ý nghĩa* không; tách *metric* khỏi *construct*. |
| **BetterBench** | README + metric + baseline + visualize + notebook tái lập một-bấm. |
| **GLUE** | Báo cáo điểm theo từng benchmark/nhóm, gộp thành một "suite" nhỏ. |
| **SuperGLUE** | Perturbation (đảo option / chèn distractor) đo robustness → động lực ra đề khó hơn khi benchmark bão hoà. |
| **SWE-bench** | GSM8K chấm theo *trích xuất đáp án + so số* thay vì so chuỗi thô; ta phơi bày phần "chấm" mới là chỗ benchmark dễ sai (giống tinh thần execution-based). |

## Bảy phân tích (mỗi cái = một luận điểm tutorial)

1. **Saturation / scaling** — accuracy MMLU vs GSM8K theo cỡ model (E2B → E4B →
   ~4B khác họ → math-7B). **GSM8K bão hoà** (mọi model 0.73–0.96) → hết phân biệt
   ở đỉnh; **MMLU vẫn phân biệt** (0.35–0.69). Cặp tương phản này = lý do GLUE nhường
   SuperGLUE.
2. **Baseline soi điểm** — model thật phải vượt random (MMLU ≈ 0.25) và majority
   class một khoảng rộng, nếu không thì điểm chỉ là may rủi.
3. **Rank-instability — chính benchmark chọn người thắng** — `ranking.csv` xếp hạng
   model trên MMLU và GSM8K *riêng* (kèm cột `params_b` minh bạch hạng cân; Spearman
   ρ cả panel lẫn riêng nhóm ≥3B). **Kết quả thật: thứ hạng RÃ.** Riêng panel đa
   năng (5 model) thứ tự MMLU và GSM8K trùng khít, nhưng thêm **một math-specialist
   (Qwen2.5-Math-7B)** là lật ngay: nó **đứng đầu GSM8K (0.963)** nhưng **bét bảng
   MMLU (0.353 — gần mức random 0.23)**, `rank_shift = 5`. Spearman **ρ = 0.14 cho
   cả 6 model**, và **ρ = 0.0 cho riêng nhóm ≥3B** (không tương quan). Một model *to
   nhất panel* (7B) lại **thua mọi model 2–4B trên MMLU** trong khi thắng tất cả trên
   GSM8K → **không cỡ model nào, không một benchmark nào "xếp hạng model"; chính
   benchmark chọn người thắng.** Đây đúng lý do GLUE nhường SuperGLUE. Hình
   `model_agreement.png` (scatter MMLU×GSM8K): math-7b nằm hẳn **ngoài** đám đông
   (góc dưới-phải: GSM8K cao, MMLU thấp) — phá đường chéo đồng thuận.
4. **Metric giòn** (GSM8K) — chạy **hai** bộ chấm khác **đúng một quyết định**:
   `naive` lấy **số ĐẦU tiên** (lỗi thực tế kinh điển `re.search(r"\d+")` → trúng
   số trung gian trong chuỗi suy luận) vs `robust` lấy **số CUỐI**; chuẩn hoá số
   y hệt nhau. Cột `n_robust_only` = số câu model **đúng** mà bộ chấm giòn đánh
   trượt — `metric_gap_examples.csv` in ví dụ cụ thể. Bằng chứng *metric ≠ construct*.
5. **Per-subject (construct validity)** — `mmlu_by_subject.csv` tách accuracy theo
   từng môn. Một con số "MMLU accuracy" che giấu chênh lệch lớn giữa môn: cùng một
   model nhảy từ 0.0 ở vài môn (abstract_algebra, college_mathematics) lên 1.0 ở môn
   khác (college_computer_science, professional_medicine) → "MMLU" không phải một
   construct đồng nhất. (Caveat: chỉ 2–5 câu/môn nên đây là minh hoạ định tính, không
   phải ước lượng chắc theo môn.)
6. **Coverage / fairness** — `coverage.csv`: tỉ lệ câu *trích được đáp án* theo
   từng model. Model bị phạt vì format (không trích được) ≠ vì sai → đo lường có
   nhiễu, và nhiễu lệch theo model (vd MMLU 0.945–0.985). Đây cũng là "what is
   measured".
7. **Robustness** (MMLU) — mỗi câu có "bản sinh đôi" nhiễu giữ nhãn (đảo vị trí
   đáp án / chèn "None of the above"); so accuracy gốc vs nhiễu.

## Cấu trúc repo

```
.
├── README.md
├── requirements.txt
├── notebook/
│   ├── 1_smoke_test.ipynb          # (optional) test nhanh 1 model trước khi chạy full
│   └── 2_run_all.ipynb             # chạy đủ 6 model -> số + hình
├── src/
│   ├── config.py               # cỡ subset + danh sách system
│   ├── data.py                 # tải MMLU + GSM8K (subset cố định seed) -> data/*.jsonl
│   ├── metrics.py              # MMLU letter-match; GSM8K first-number vs last-number
│   ├── perturb.py              # sinh đôi MMLU: option_shuffle / distractor
│   ├── models.py               # HFRunner (transformers) + baseline + cache + FakeRunner
│   ├── evaluate.py             # vòng chạy -> results/per_item.csv
│   └── analysis.py             # bảng + 4 hình + ví dụ định tính
├── data/                       # subset benchmark đã cache (sinh khi chạy)
└── results/
    ├── per_item.csv            # từng câu, đúng/sai, đáp án trích
    ├── accuracy.csv            # MMLU vs GSM8K theo system (+ baseline)
    ├── ranking.csv             # thứ hạng MMLU vs GSM8K (+ params_b)
    ├── coverage.csv            # tỉ lệ trích được đáp án / model (fairness)
    ├── mmlu_by_subject.csv     # accuracy theo môn (construct variance)
    ├── metric_gap_examples.csv # câu đúng bị naive grader đánh trượt
    ├── error_analysis.csv      # mọi câu sai (phân tích định tính)
    ├── metric_gap.csv          # GSM8K naive vs robust
    ├── robustness.csv          # MMLU gốc vs nhiễu
    └── figures/*.png           # saturation, metric_gap, robustness
```

## Cách chạy

### A. Kaggle — chạy đủ 6 model (khuyến nghị)

1. Mở `notebook/2_run_all.ipynb`. **Settings → Accelerator → `GPU T4`**, **Internet → On**.
2. **Add-ons → Secrets →** thêm secret `HF_TOKEN` (HF read token). Accept licence
   Gemma tại <https://huggingface.co/google/gemma-4-E4B-it> (các model khác ungated).
3. `REPO_URL` (cell 1) đã trỏ repo này.
4. **Run all.** Greedy → bấm lại ra **đúng số**.

**Thêm model = chỉ chạy phần mới:** `results/predictions/*.json` đã commit, nên
model nào đã chấm sẽ **cache-hit tức thì**; thêm model vào `config.SYSTEMS` rồi
Run all → chỉ model mới thật sự gọi GPU. (Lần đầu 6 model ~1.5–2h; có cache thì
chỉ tốn thời gian cho model mới.)

### B. Máy có GPU NVIDIA ≥ 8GB

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login          # cần đã accept licence Gemma
python src/evaluate.py         # -> results/per_item.csv
python src/analysis.py         # -> results/*.csv + figures/*.png
```

### C. Không có GPU — chỉ kiểm tra pipeline (không phải kết quả thật)

```bash
pip install pandas matplotlib numpy
MINIBENCH_SYNTHETIC=1 python src/evaluate.py --fake   # model giả + data giả
MINIBENCH_SYNTHETIC=1 python src/analysis.py
```

Đường này dùng `FakeRunner` + dữ liệu tổng hợp để chứng minh toàn bộ luồng
chạy được trên máy không GPU; **số liệu sinh ra không dùng để báo cáo.**

## Kết quả thật

> _Chạy trên Kaggle T4 free, 4-bit greedy, subset cố định seed=0 (150 MMLU + 80
> GSM8K). Bấm lại ra đúng số._

**Accuracy MMLU vs GSM8K** (`results/accuracy.csv`), kèm hạng cân `params_b`:

| system | params_b | acc_mmlu | acc_gsm8k | hạng MMLU → GSM8K |
|---|---|---|---|---|
| qwen3.5-4b | 4.0 | **0.687** | 0.913 | 1 → 2 |
| phi-3.5-mini | 3.8 | 0.587 | 0.850 | 2 → 3 |
| gemma-4-e4b | 4.5 | 0.533 | 0.788 | 3 → 4 |
| qwen2.5-3b | 3.1 | 0.513 | 0.775 | 4 → 5 |
| gemma-4-e2b | 2.3 | 0.427 | 0.725 | 5 → 6 |
| qwen2.5-math-7b | 7.0 | 0.353 | **0.963** | **6 → 1** |
| baseline-majority | — | 0.253 | 0.000 | — |
| baseline-random | — | 0.233 | 0.050 | — |

Mọi model thật **vượt xa** baseline trên cả hai benchmark → điểm có *ý nghĩa*. GSM8K
đã ở vùng **0.73–0.96** (bão hoà, ít phân biệt) còn MMLU trải rộng **0.35–0.69** (còn
phân biệt) — đúng tương phản GLUE↔SuperGLUE. Đáng chú ý: **qwen2.5-math-7b** (to
nhất, 7B) chỉ đạt **MMLU 0.353** — *gần baseline random* — nhưng **đỉnh GSM8K 0.963**.

**Rank-instability** (`results/ranking.csv`): thứ hạng **rã** giữa hai benchmark.
Spearman **ρ = 0.14 cho cả 6 model**, **ρ = 0.0 cho riêng nhóm ≥3B**. Math-specialist
nhảy từ **hạng 6 (MMLU) lên hạng 1 (GSM8K)**, `rank_shift = 5` → *chính benchmark
chọn người thắng*, không phải cỡ model. Xem §3.

**GSM8K — metric gap** (`results/metric_gap.csv`): cột `n_robust_only` = số câu model
đúng nhưng bộ chấm *giòn* (lấy số ĐẦU) đánh trượt. Lớn ở mọi model — **58–77 / 80**
(math-7b 77, qwen3.5-4b 72, phi-3.5 67, gemma-e4b 63, qwen2.5-3b 62, gemma-e2b 58).
`acc_naive` ≈ 0 vs `acc_robust` 0.73–0.96 → cùng prediction, đổi *một* quyết định
trích số là sập điểm. Bằng chứng đậm *metric ≠ construct*.

**MMLU — robustness** (`results/robustness.csv`): độ tụt accuracy khi nhiễu giữ nhãn
(đảo option / chèn distractor). Tụt **0.06–0.18**: math-7b −0.18, qwen3.5-4b −0.16,
qwen2.5-3b −0.10, gemma-e4b −0.10, phi-3.5 −0.08, gemma-e2b −0.06. Tụt lớn nhất ở
model *yếu MMLU nhất* (math-7b) → phần điểm MMLU của nó dựa vào vị trí/format đáp án
hơn là kiến thức thật.

## Tái lập (reproducibility)

- Subset benchmark lấy bằng **seed cố định** (`config.SEED`) và ghi ra
  `data/*.jsonl` → ai chạy cũng nhận **cùng item, cùng thứ tự**.
- Sinh văn bản **greedy** (`do_sample=False`) → không phụ thuộc ngẫu nhiên.
- Mỗi lời gọi model cache theo `sha256(model_id, prompt)` trong
  `results/predictions/` → chạy lại nối tiếp tức thì, số liệu ổn định.

## Hạn chế (nêu để bám construct validity)

- Subset nhỏ (150 MMLU + 80 GSM8K) → khoảng tin cậy rộng; mục tiêu là *minh hoạ
  vấn đề benchmark*, không xếp hạng tuyệt đối.
- Sáu model 2–7B chạy T4 free; rung lớn hơn (Qwen3.5-9B…) cần Kaggle T4×2/A100.
  Qwen3.5 chạy ở chế độ tắt thinking để answer trực tiếp.
- **Math-specialist 7B là model *chủ đích lệch*, không thuộc panel ~4B:** đưa vào để
  *kiểm tra* (chứ không giả định) tính rank-instability. Panel đa năng 5 model thật ra
  xếp hạng trùng (ρ=1.0 nội bộ); chính math-7b — cao GSM8K, thấp MMLU — mới làm ρ rã
  (0.14 toàn bộ, 0.0 nhóm ≥3B). Đây là *thiết kế có chủ ý để phơi bày hiện tượng*,
  ta khai báo rõ chứ không trình bày như tự nhiên xảy ra ở mọi panel.
- `naive` grader cố tình làm giòn để phơi bày; nhưng đó đúng là kiểu chấm một dòng
  hay gặp ngoài thực tế.
- **Contamination (cảnh báo mạnh):** MMLU (2021) và GSM8K (2021) gần như chắc nằm
  trong dữ liệu huấn luyện của các model này. GSM8K đạt **~0.9 cho model chỉ 3–4B**
  → nhiều khả năng đo *trí nhớ/nhiễm dữ liệu* hơn là *suy luận* thuần. Vì vậy điểm
  tuyệt đối **không** nên đọc như "năng lực thật"; demo dùng chúng để *phơi bày
  chính vấn đề này*, không để xếp hạng. (Robustness drop nhỏ ⇒ không bám chuỗi test
  verbatim, nhưng không loại trừ nhiễm ở mức ngữ nghĩa.)
- **Công bằng giữa hạng cân:** E2B (2.3B) nhỏ hơn nhóm ~4B; bảng có cột `params_b`
  và tính rank-instability riêng cho nhóm ≥3B để không nhầm chênh lệch *size* thành
  chênh lệch *họ model*.
- **Hẹp về construct:** chỉ phủ kiến thức (MMLU) + toán (GSM8K); thiếu reasoning
  tổng quát, trung thực, an toàn, code… → kết luận chỉ trong phạm vi hai construct này.
