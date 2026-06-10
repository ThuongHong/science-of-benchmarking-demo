# Science of Benchmarking

Một **nghiên cứu + công cụ** về *tính hợp lệ của phép đo* trong benchmark mô hình
ngôn ngữ: thay vì hỏi "model nào điểm cao nhất", dự án hỏi **"chính phép đo này có
đáng tin không"**.

Repo gồm hai phần độc lập nhưng bổ trợ:

- **`report/`** — bản báo cáo đầy đủ (LaTeX) tổng hợp lý thuyết: construct validity,
  chất lượng benchmark, lịch sử GLUE → SuperGLUE, SWE-bench, và đâu là chỗ benchmark
  hay sai.
- **`src/` + `notebook/`** — **Mini Benchmark Evaluator**: pipeline chạy thật, lấy
  hai benchmark công khai (MMLU, GSM8K) rồi áp một bộ phân tích phê phán lên chúng để
  *phơi bày* các vấn đề đo lường bằng số liệu thật.

## Mini Benchmark Evaluator — công cụ làm gì

Cho một tập model + một tập benchmark, công cụ sinh ra số liệu trả lời sáu câu hỏi
về *chất lượng phép đo* (không phải về xếp hạng model):

| Phân tích | Câu hỏi nó trả lời | Output |
|---|---|---|
| **Saturation** | Benchmark còn phân biệt được model mạnh không, hay đã đụng trần? | `accuracy.csv`, `saturation.png` |
| **Baseline** | Điểm có *ý nghĩa* hơn đoán mò không? | `accuracy.csv` (random / majority) |
| **Rank-instability** | Đổi benchmark có đổi luôn người thắng không? | `ranking.csv`, `model_agreement.png` |
| **Metric ≠ construct** | Đổi *cách chấm* (không đổi đầu ra model) có sập điểm không? | `metric_gap.csv`, `metric_gap_examples.csv` |
| **Construct variance** | Một con số "accuracy" có che giấu chênh lệch giữa các môn? | `mmlu_by_subject.csv` |
| **Robustness** | Điểm gốc dựa vào *kiến thức* hay dựa vào *format câu hỏi*? | `robustness.csv`, `robustness.png` |

Mọi phân tích đều minh bạch hạng cân (`params_b`) để không nhầm chênh lệch *kích
thước* với chênh lệch *họ model*.

## Cấu hình mặc định

| Thành phần | Lựa chọn |
|---|---|
| Benchmark | **MMLU** (kiến thức, trắc nghiệm 4 đáp án, 150 câu) + **GSM8K** (toán reasoning, 80 câu) |
| Model | Gemma-4 E2B/E4B (ladder cùng họ) · Qwen3.5-4B / Phi-3.5-mini / Qwen2.5-3B (panel ~4B khác họ) · Qwen2.5-Math-7B (math-specialist, đầu dò lệch chuyên môn) |
| Baseline | random · majority-class (non-LLM) |
| Backend | Hugging Face `transformers`, 4-bit, **greedy → tất định** |

Đổi model/benchmark = sửa `src/config.py`. Pipeline không gắn cứng vào danh sách này.

## Cấu trúc repo

```
.
├── README.md
├── requirements.txt
├── report/                     # báo cáo LaTeX đầy đủ
│   ├── main.tex                #   biên dịch bằng XeLaTeX/LuaLaTeX
│   ├── preamble.tex
│   ├── sections/               #   01_introduction … 20_reproducibility
│   └── ref/ref.bib
├── notebook/
│   ├── 1_smoke_test.ipynb      # test nhanh 1 model
│   └── 2_run_all.ipynb         # chạy đủ panel → số + hình
├── src/
│   ├── config.py               # subset + danh sách system (điểm chỉnh chính)
│   ├── data.py                 # tải MMLU + GSM8K (subset cố định seed)
│   ├── metrics.py              # MMLU letter-match; GSM8K first-number vs last-number
│   ├── perturb.py              # sinh bản nhiễu giữ nhãn (option_shuffle / distractor)
│   ├── models.py               # HFRunner + baseline + cache + FakeRunner
│   ├── evaluate.py             # vòng chạy → results/per_item.csv
│   └── analysis.py             # bảng + hình + ví dụ định tính
├── data/                       # subset benchmark đã cache (sinh khi chạy)
└── results/
    ├── per_item.csv            # từng câu: đúng/sai, đáp án trích
    ├── accuracy.csv            # MMLU vs GSM8K theo system (+ baseline)
    ├── ranking.csv             # thứ hạng MMLU vs GSM8K (+ params_b)
    ├── coverage.csv            # tỉ lệ trích được đáp án / model
    ├── mmlu_by_subject.csv     # accuracy theo môn
    ├── metric_gap.csv          # GSM8K naive vs robust grader
    ├── robustness.csv          # MMLU gốc vs nhiễu
    ├── predictions/*.json      # cache generation (commit để tái lập)
    └── figures/*.png
```

## Chạy demo

### A. Kaggle — đủ panel (khuyến nghị)

1. Mở `notebook/2_run_all.ipynb`. **Settings → Accelerator → `GPU T4`**, **Internet → On**.
2. **Add-ons → Secrets →** thêm `HF_TOKEN` (HF read token). Accept licence Gemma tại
   <https://huggingface.co/google/gemma-4-E4B-it> (model khác ungated).
3. **Run all.** Greedy → bấm lại ra **đúng số**.

Cache generation (`results/predictions/*.json`) đã commit: model đã chấm sẽ
**cache-hit tức thì**, thêm model vào `config.SYSTEMS` rồi Run all → chỉ model mới
gọi GPU thật. (Lần đầu cả panel ~1.5–2h; có cache thì chỉ tốn cho model mới.)

### B. Máy có GPU NVIDIA ≥ 8GB

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login          # cần đã accept licence Gemma
python src/evaluate.py         # → results/per_item.csv
python src/analysis.py         # → results/*.csv + figures/*.png
```

### C. Không GPU — chỉ kiểm tra pipeline (không phải kết quả thật)

```bash
pip install pandas matplotlib numpy
MINIBENCH_SYNTHETIC=1 python src/evaluate.py --fake   # model giả + data giả
MINIBENCH_SYNTHETIC=1 python src/analysis.py
```

`FakeRunner` + dữ liệu tổng hợp chứng minh toàn bộ luồng chạy được trên máy không
GPU; **số sinh ra không dùng để báo cáo.**

## Biên dịch báo cáo

```bash
cd report
latexmk -xelatex -outdir=output main.tex   # output/main.pdf
```

Cần XeLaTeX hoặc LuaLaTeX (hỗ trợ tiếng Việt). Trên Overleaf: Menu → Compiler → XeLaTeX.

## Kết quả thật (panel mặc định)

> _Kaggle T4 free, 4-bit greedy, subset seed=0 (150 MMLU + 80 GSM8K). Bấm lại ra đúng số._

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

Vài quan sát rút ra từ bảng trên:

- **Saturation:** GSM8K nằm gọn ở **0.73–0.96** (ít phân biệt ở đỉnh) còn MMLU trải
  **0.35–0.69** (còn phân biệt) — đúng tương phản từng đẩy GLUE nhường SuperGLUE.
  Với n=80, "bão hoà" là quan sát định tính, không phải kết luận thống kê.
- **Rank-instability:** panel 5 model đa năng xếp hạng nhất quán (ρ=1.0), nhưng thêm
  **một** math-specialist lệch chủ đích là đủ kéo Spearman xuống **ρ=0.14** (ρ=0.0
  cho riêng nhóm ≥3B): nó **bét MMLU (0.353)** mà **đỉnh GSM8K (0.963)**,
  `rank_shift=5`. Một construct đủ lệch là đủ biến "ai thắng" thành hàm của benchmark.
- **Metric ≠ construct:** cùng đầu ra model, đổi bộ chấm GSM8K từ *lấy số cuối* sang
  *lấy số đầu* làm accuracy rơi từ 0.73–0.96 xuống ≈0 (`n_robust_only` = 58–77 / 80).
- **Robustness:** nhiễu giữ nhãn làm mọi model tụt 0.06–0.18; tụt nhiều nhất ở model
  *yếu MMLU nhất* → điểm gốc một phần dựa vào format hơn kiến thức.

## Tái lập

- Subset lấy bằng **seed cố định** (`config.SEED`) → cùng item, cùng thứ tự.
- Generation **greedy** (`do_sample=False`) → không phụ thuộc ngẫu nhiên.
- Mỗi lời gọi model cache theo `sha256(model_id, prompt)` → chạy nối tiếp tức thì,
  số liệu ổn định, không cần GPU lại nếu đã có cache.

## Hạn chế

- **Cỡ mẫu nhỏ** (150 MMLU + 80 GSM8K) → khoảng tin cậy rộng. Mục tiêu là *minh hoạ
  vấn đề đo lường*, không xếp hạng tuyệt đối.
- **Math-specialist là đầu dò chủ đích lệch**, không thuộc panel ~4B. Đưa vào để
  *kiểm tra* rank-instability, không giả định nó. Panel đa năng tự nó ổn định (ρ=1.0).
- **Contamination:** MMLU (2021) và GSM8K (2021) gần như chắc nằm trong dữ liệu huấn
  luyện các model này; điểm tuyệt đối phản ánh *trí nhớ* nhiều hơn *suy luận*. Demo
  dùng chúng để phơi bày chính vấn đề này, không để đánh giá năng lực thật.
- **Hẹp về construct:** chỉ phủ kiến thức + toán; thiếu reasoning tổng quát, trung
  thực, an toàn, code… → kết luận chỉ trong phạm vi hai construct này.
