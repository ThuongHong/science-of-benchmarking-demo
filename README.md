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
| Model | **Gemma-4 E2B-it** và **E4B-it** (cùng họ, 2 cỡ; chạy free Colab T4 4-bit) |
| Baseline | random, majority-class (non-LLM) |
| Backend | Hugging Face `transformers`, 4-bit, **greedy → deterministic** |
| Chạy ở đâu | Notebook Colab (T4 free) — `notebook/benchmark_demo.ipynb` |

## Demo liên hệ 6 paper thế nào

| Paper | Liên hệ trong demo |
|---|---|
| **AI and the Everything in the Whole Wide World Benchmark** | MMLU/GSM8K chỉ đo vài năng lực hẹp → điểm cao không chứng minh "AI tổng quát". |
| **Measuring what Matters** | Có baseline random/majority để kiểm tra điểm số có *ý nghĩa* không; tách *metric* khỏi *construct*. |
| **BetterBench** | README + metric + baseline + visualize + notebook tái lập một-bấm. |
| **GLUE** | Báo cáo điểm theo từng benchmark/nhóm, gộp thành một "suite" nhỏ. |
| **SuperGLUE** | Perturbation (đảo option / chèn distractor) đo robustness → động lực ra đề khó hơn khi benchmark bão hoà. |
| **SWE-bench** | GSM8K chấm theo *trích xuất đáp án + so số* thay vì so chuỗi thô; ta phơi bày phần "chấm" mới là chỗ benchmark dễ sai (giống tinh thần execution-based). |

## Năm phân tích (mỗi cái = một luận điểm tutorial)

1. **Saturation** — accuracy MMLU vs GSM8K. Khi model mạnh dồn cục sát trần ở một
   benchmark, nó hết phân biệt được → lý do GLUE phải nhường SuperGLUE.
2. **Baseline soi điểm** — model thật phải vượt random (MMLU ≈ 0.25) và majority
   class một khoảng rộng, nếu không thì điểm chỉ là may rủi.
3. **Metric giòn** (GSM8K) — chạy **hai** bộ chấm trên *cùng output*:
   `naive` (so nguyên dòng cuối) vs `robust` (trích số cuối, bỏ `$ , %`). Cột
   `n_robust_only` = số câu model trả lời **đúng** nhưng bộ chấm giòn đánh trượt.
   Đây là bằng chứng *metric ≠ construct*.
4. **Robustness** (MMLU) — mỗi câu có một "bản sinh đôi" bị nhiễu giữ nhãn
   (đảo vị trí đáp án / chèn "None of the above"); so accuracy gốc vs nhiễu.
5. **Contamination / construct validity** — thảo luận định tính trong báo cáo
   (rò rỉ dữ liệu của benchmark cũ, ý nghĩa của "MMLU accuracy").

## Cấu trúc repo

```
.
├── README.md
├── requirements.txt
├── notebook/
│   ├── benchmark_demo_kaggle.ipynb  # entry Kaggle (T4 x2): Run all -> số + hình
│   └── benchmark_demo.ipynb         # entry Colab (T4 đơn)
├── src/
│   ├── config.py               # cỡ subset + danh sách system
│   ├── data.py                 # tải MMLU + GSM8K (subset cố định seed) -> data/*.jsonl
│   ├── metrics.py              # MMLU letter-match; GSM8K naive vs robust
│   ├── perturb.py              # sinh đôi MMLU: option_shuffle / distractor
│   ├── models.py               # GemmaRunner (transformers) + baseline + cache + FakeRunner
│   ├── evaluate.py             # vòng chạy -> results/per_item.csv
│   └── analysis.py             # 5 bảng + 3 hình
├── data/                       # subset benchmark đã cache (sinh khi chạy)
└── results/
    ├── per_item.csv            # từng câu, đúng/sai, đáp án trích
    ├── accuracy.csv            # MMLU vs GSM8K theo system (+ baseline)
    ├── metric_gap.csv          # GSM8K naive vs robust
    ├── robustness.csv          # MMLU gốc vs nhiễu
    └── figures/*.png           # saturation, metric_gap, robustness
```

## Cách chạy

### A. Kaggle (khuyến nghị — `T4 x2` free, chạy được cả rung 26B)

1. Mở `notebook/benchmark_demo_kaggle.ipynb`. **Settings → Accelerator → `GPU T4 x2`**, **Internet → On**.
2. **Add-ons → Secrets →** thêm secret `HF_TOKEN` (HF read token). Accept licence
   Gemma tại <https://huggingface.co/google/gemma-4-E4B-it>.
3. Sửa `REPO_URL` (cell 1) thành link repo này (hoặc attach repo làm Kaggle Dataset).
4. Đặt `ADD_26B = True` (cell 5) nếu muốn thêm rung **26B-A4B MoE** (cần T4×2).
5. **Run all.** Greedy → bấm lại ra **đúng số**.

### B. Colab (T4 đơn — chỉ E2B + E4B)

1. Mở `notebook/benchmark_demo.ipynb`. **Runtime → T4 GPU**.
2. Sửa `REPO_URL` (cell 1) thành link repo này.
3. Accept licence Gemma rồi dán HF token ở cell login.
4. **Runtime → Run all.**

### C. Máy có GPU NVIDIA ≥ 8GB

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login          # cần đã accept licence Gemma
python src/evaluate.py         # -> results/per_item.csv
python src/analysis.py         # -> results/*.csv + figures/*.png
```

### D. Không có GPU — chỉ kiểm tra pipeline (không phải kết quả thật)

```bash
pip install pandas matplotlib numpy
MINIBENCH_SYNTHETIC=1 python src/evaluate.py --fake   # model giả + data giả
MINIBENCH_SYNTHETIC=1 python src/analysis.py
```

Đường này dùng `FakeRunner` + dữ liệu tổng hợp để chứng minh toàn bộ luồng
chạy được trên máy không GPU; **số liệu sinh ra không dùng để báo cáo.**

## Kết quả thật

> _Điền sau khi chạy `notebook/benchmark_demo.ipynb` trên Colab (T4)._
> _Không khai số liệu chưa chạy._

**Accuracy MMLU vs GSM8K** (`results/accuracy.csv`):

| system | acc_mmlu | acc_gsm8k |
|---|---|---|
| gemma-4-e4b | … | … |
| gemma-4-e2b | … | … |
| baseline-majority | … | … |
| baseline-random | … | … |

**GSM8K — metric gap** (`results/metric_gap.csv`): cột `n_robust_only` = số câu
đúng bị bộ chấm giòn đánh trượt → …

**MMLU — robustness** (`results/robustness.csv`): độ tụt accuracy khi nhiễu → …

## Tái lập (reproducibility)

- Subset benchmark lấy bằng **seed cố định** (`config.SEED`) và ghi ra
  `data/*.jsonl` → ai chạy cũng nhận **cùng item, cùng thứ tự**.
- Sinh văn bản **greedy** (`do_sample=False`) → không phụ thuộc ngẫu nhiên.
- Mỗi lời gọi model cache theo `sha256(model_id, prompt)` trong
  `results/predictions/` → chạy lại nối tiếp tức thì, số liệu ổn định.

## Hạn chế (nêu để bám construct validity)

- Subset nhỏ (150 MMLU + 80 GSM8K) → khoảng tin cậy rộng; mục tiêu là *minh hoạ
  vấn đề benchmark*, không xếp hạng tuyệt đối.
- Chỉ hai cỡ Gemma-4 chạy được trên T4 free; rung lớn hơn (26B MoE) cần Kaggle/A100.
- `naive` grader cố tình làm giòn để phơi bày; nhưng đó đúng là kiểu chấm một dòng
  hay gặp ngoài thực tế.
- MMLU/GSM8K là benchmark cũ, có nguy cơ nhiễm dữ liệu huấn luyện → bàn ở mục
  contamination của báo cáo.
