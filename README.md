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
| Model | 5 model nhỏ: **Gemma-4 E2B/E4B** (ladder cùng họ) + **Qwen3.5-4B / Phi-3.5-mini / Qwen2.5-3B** (panel ~4B khác họ); tất cả chạy 1 T4 free 4-bit |
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

## Năm phân tích (mỗi cái = một luận điểm tutorial)

1. **Saturation / scaling** — accuracy MMLU vs GSM8K theo cỡ model (E2B → E4B →
   26B). Khi model mạnh dồn cục sát trần ở một benchmark, nó hết phân biệt được →
   lý do GLUE phải nhường SuperGLUE.
2. **Baseline soi điểm** — model thật phải vượt random (MMLU ≈ 0.25) và majority
   class một khoảng rộng, nếu không thì điểm chỉ là may rủi.
3. **Rank-instability (panel ~4B khác họ)** — `ranking.csv` xếp hạng các model
   trên MMLU và GSM8K *riêng*. Nếu thứ tự **lật** (Spearman ρ < 1) → không một
   benchmark / một con số nào "xếp hạng model"; chính benchmark chọn người thắng.
   Hình `model_agreement.png` (scatter MMLU×GSM8K) cho thấy lệch khỏi đường chéo.
3. **Metric giòn** (GSM8K) — chạy **hai** bộ chấm khác **đúng một quyết định**:
   `naive` lấy **số ĐẦU tiên** (lỗi thực tế kinh điển `re.search(r"\d+")` → trúng
   số trung gian trong chuỗi suy luận) vs `robust` lấy **số CUỐI**; chuẩn hoá số
   y hệt nhau. Cột `n_robust_only` = số câu model **đúng** mà bộ chấm giòn đánh
   trượt — `metric_gap_examples.csv` in ví dụ cụ thể. Bằng chứng *metric ≠ construct*.
4. **Per-subject (construct validity)** — `mmlu_by_subject.csv` tách accuracy theo
   từng môn. Một con số "MMLU accuracy" che giấu chênh lệch lớn giữa môn (marketing
   cao ↔ moral_scenarios ~ random) → "MMLU" không phải một construct đồng nhất.
5. **Robustness** (MMLU) — mỗi câu có "bản sinh đôi" nhiễu giữ nhãn (đảo vị trí
   đáp án / chèn "None of the above"); so accuracy gốc vs nhiễu. Drop ≈ 0 cũng là
   tín hiệu: model không bám chuỗi test verbatim → ít dấu hiệu **contamination** bề mặt.
6. **Contamination / construct** — thảo luận thêm trong báo cáo (rò rỉ dữ liệu
   benchmark cũ), neo vào kết quả robustness ở trên.

## Cấu trúc repo

```
.
├── README.md
├── requirements.txt
├── notebook/
│   ├── 1_smoke_test.ipynb          # test nhanh 1 model trước khi chạy full
│   ├── 2_run_all.ipynb             # chạy đủ 5 model -> số + hình
│   └── 3_add_and_compare.ipynb     # incremental: chạy model mới + merge + analyze
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
    ├── ranking.csv             # thứ hạng MMLU vs GSM8K (rank-instability)
    ├── mmlu_by_subject.csv     # accuracy theo môn (construct variance)
    ├── metric_gap_examples.csv # câu đúng bị naive grader đánh trượt
    ├── error_analysis.csv      # mọi câu sai (phân tích định tính)
    ├── metric_gap.csv          # GSM8K naive vs robust
    ├── robustness.csv          # MMLU gốc vs nhiễu
    └── figures/*.png           # saturation, metric_gap, robustness
```

## Cách chạy

### A. Kaggle — chạy đủ 5 model (khuyến nghị)

1. Mở `notebook/2_run_all.ipynb`. **Settings → Accelerator → `GPU T4`**, **Internet → On**.
2. **Add-ons → Secrets →** thêm secret `HF_TOKEN` (HF read token). Accept licence
   Gemma tại <https://huggingface.co/google/gemma-4-E4B-it> (các model khác ungated).
3. `REPO_URL` (cell 1) đã trỏ repo này.
4. **Run all.** Greedy → bấm lại ra **đúng số**. (~1.5–2h cho 5 model.)

### B. Kaggle — thêm model & tổng hợp (incremental)

Khi đã chạy vài model ở lần trước và chỉ muốn chạy **model mới** rồi gộp:
`notebook/3_add_and_compare.ipynb` chạy `config.PEERS`, **merge** với `per_item.csv`
cũ (attach dưới dạng Kaggle Dataset), rồi chạy lại toàn bộ analysis cho cả panel.

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

> _Điền sau khi chạy notebook Kaggle. Không khai số liệu chưa chạy._

**Accuracy MMLU vs GSM8K** (`results/accuracy.csv`):

| system | acc_mmlu | acc_gsm8k |
|---|---|---|
| gemma-4-e4b | … | … |
| gemma-4-e2b | … | … |
| qwen3.5-4b | … | … |
| phi-3.5-mini | … | … |
| qwen2.5-3b | … | … |
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
- Ba model nhỏ (E2B/E4B/Qwen3.5-4B) chạy T4 free; rung lớn hơn (Qwen3.5-9B…) cần
  Kaggle T4×2/A100. Qwen3.5 chạy ở chế độ tắt thinking để answer trực tiếp.
- `naive` grader cố tình làm giòn để phơi bày; nhưng đó đúng là kiểu chấm một dòng
  hay gặp ngoài thực tế.
- MMLU/GSM8K là benchmark cũ, có nguy cơ nhiễm dữ liệu huấn luyện → bàn ở mục
  contamination của báo cáo.
