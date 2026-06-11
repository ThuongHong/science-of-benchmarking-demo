# Mini Benchmark Evaluator

Một demo thực nghiệm về **science of benchmarking**: thay vì chỉ hỏi model nào
đạt điểm cao nhất, pipeline kiểm tra xem kết luận đó thay đổi thế nào khi đổi
benchmark, cách chấm điểm hoặc format câu hỏi.

Demo chạy sáu language model và hai baseline trên:

- **MMLU**: kiến thức tổng quát, trắc nghiệm bốn lựa chọn.
- **GSM8K**: bài toán có lời văn và chuỗi suy luận.

Mục tiêu là minh họa các vấn đề đo lường, **không phải tạo leaderboard model**.

## Chạy Trên Kaggle

### Yêu cầu

- Kaggle account với Internet bật.
- GPU T4.
- Hugging Face read token lưu trong Kaggle Secret với tên `HF_TOKEN`.
- Đã chấp nhận license của
  [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it).

### Chạy demo

1. Upload hoặc import [`notebook/2_run_all.ipynb`](notebook/2_run_all.ipynb)
   vào Kaggle.
2. Chọn **Settings → Accelerator → GPU T4**, bật Internet và thêm secret
   `HF_TOKEN`.
3. Chọn **Run All**.

Notebook sẽ:

```text
load deterministic benchmark subsets
        ↓
evaluate models and baselines
        ↓
write per-item predictions
        ↓
build analysis tables and figures
```

Kết quả nằm trong thư mục `results/` của notebook output. Prediction cache đã
được commit nên lần chạy lại không cần sinh lại các prompt đã có; model vẫn cần
được tải và load lần lượt lên GPU.

Để kiểm tra một model mới trước khi chạy toàn bộ panel, dùng
[`notebook/1_smoke_test.ipynb`](notebook/1_smoke_test.ipynb).

## Kết Quả Demo

Thiết lập mặc định dùng `seed=0`, gồm **150 câu MMLU**, **80 câu GSM8K** và
**50 câu MMLU bị perturb**. Model chạy bằng 4-bit quantization và greedy decoding.

| System | Params | MMLU | GSM8K | Hạng MMLU → GSM8K |
|---|---:|---:|---:|---:|
| qwen3.5-4b | 4.0B | **0.687** | 0.913 | 1 → 2 |
| phi-3.5-mini | 3.8B | 0.587 | 0.850 | 2 → 3 |
| gemma-4-e4b | 4.5B | 0.533 | 0.788 | 3 → 4 |
| qwen2.5-3b | 3.1B | 0.513 | 0.775 | 4 → 5 |
| gemma-4-e2b | 2.3B | 0.427 | 0.725 | 5 → 6 |
| qwen2.5-math-7b | 7.0B | 0.353 | **0.963** | **6 → 1** |
| baseline-majority | — | 0.253 | 0.000 | — |
| baseline-random | — | 0.233 | 0.050 | — |

### Demo cho thấy gì?

**1. “Model tốt nhất” phụ thuộc benchmark**

`qwen3.5-4b` đứng đầu MMLU nhưng `qwen2.5-math-7b` đứng đầu GSM8K. Math
specialist dịch từ hạng 6 lên hạng 1 khi chuyển từ benchmark kiến thức tổng quát
sang benchmark toán. Spearman correlation của toàn panel chỉ còn **ρ = 0.14**.

Artifact: `results/ranking.csv`, `results/figures/model_agreement.png`.

**2. Một cách chấm điểm dễ vỡ có thể chấm sai câu trả lời đúng**

GSM8K thường tạo output chứa nhiều con số trung gian. Khi grader lấy **số đầu
tiên** thay vì **số cuối cùng**, accuracy của cùng một tập output giảm gần về
0. Tùy model, **58–77 trên 80** câu chỉ được tính đúng bởi grader lấy số cuối.

Artifact: `results/metric_gap.csv`, `results/metric_gap_examples.csv`,
`results/figures/metric_gap.png`.

**3. Điểm số nhạy với thay đổi format**

Demo tạo các bản MMLU giữ nguyên đáp án đúng nhưng đảo thứ tự lựa chọn hoặc thêm
distractor. Accuracy giảm khoảng **0.06–0.18** tùy model. Điều này cho thấy một
phần điểm số phụ thuộc cách trình bày câu hỏi, không chỉ kiến thức cần đo.

Artifact: `results/robustness.csv`, `results/figures/robustness.png`.

**4. Một điểm tổng hợp che giấu khác biệt bên trong**

`acc_mmlu` gộp nhiều môn thành một số duy nhất. Phân tích theo subject cho thấy
năng lực của cùng một model có thể khác đáng kể giữa các môn.

Artifact: `results/mmlu_by_subject.csv`,
`results/figures/mmlu_by_subject.png`.

**5. Baseline giúp kiểm tra điểm có ý nghĩa hay không**

Hai baseline random và majority tạo mốc tham chiếu không dùng language model.
Trên MMLU, majority baseline đạt `0.253`, gần mức đoán ngẫu nhiên `0.25`.

Artifact: `results/accuracy.csv`, `results/figures/saturation.png`.

## Output

Các CSV và figure phân tích được tái tạo mỗi lần chạy và không được commit.
Prediction cache được commit để giảm chi phí chạy lại.

| Output | Nội dung |
|---|---|
| `results/per_item.csv` | Một dòng cho mỗi cặp system–item, gồm prediction, đáp án trích xuất và correctness |
| `results/accuracy.csv` | Accuracy MMLU/GSM8K và baseline |
| `results/ranking.csv` | Thứ hạng model trên từng benchmark |
| `results/coverage.csv` | Tỷ lệ output trích xuất được đáp án |
| `results/metric_gap.csv` | So sánh grader lấy số đầu và số cuối |
| `results/metric_gap_examples.csv` | Ví dụ model đúng nhưng grader dễ vỡ chấm sai |
| `results/robustness.csv` | Accuracy trước và sau perturbation |
| `results/mmlu_by_subject.csv` | Accuracy MMLU theo subject |
| `results/error_analysis.csv` | Các prediction bị chấm sai để phân tích định tính |
| `results/figures/*.png` | Biểu đồ tổng hợp |

## Cách Hoạt Động

```text
src/data.py
  tải và cache subset MMLU/GSM8K với seed cố định

src/perturb.py
  tạo MMLU twins bằng option shuffle và distractor

src/models.py
  chạy Hugging Face model, baseline hoặc fake model; cache generation

src/evaluate.py
  chấm từng system trên từng item → results/per_item.csv

src/analysis.py
  tạo bảng, biểu đồ, ranking và error analysis
```

Panel model và kích thước subset được cấu hình tập trung trong
[`src/config.py`](src/config.py).

## Tùy Chỉnh

Thêm một Hugging Face instruction model vào `SYSTEMS`:

```python
{
    "kind": "hf",
    "name": "my-model",
    "model_id": "organization/model-id",
    "params_b": 4.0,
}
```

Model cần hỗ trợ `AutoTokenizer`, `AutoModelForCausalLM` và chat template.
Nếu model cần cấu hình riêng, có thể thêm `max_new_tokens`,
`chat_template_kwargs` hoặc `trust_remote_code`.

Đổi quy mô demo bằng:

```python
N_MMLU = 150
N_GSM8K = 80
N_PERTURB = 50
SEED = 0
```

Khi prompt hoặc model ID thay đổi, cache key cũng thay đổi và model sẽ được chạy
lại cho các item tương ứng.

## Chạy Local

### GPU NVIDIA

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
hf auth login

python src/evaluate.py
python src/analysis.py
```

Panel mặc định chạy từng model một bằng 4-bit quantization. Cần GPU hỗ trợ
`bitsandbytes`; T4 16 GB là cấu hình đã dùng cho demo.

### Không GPU: kiểm tra pipeline

Chế độ này dùng dữ liệu và model giả có seed cố định. Nó chỉ kiểm tra luồng xử
lý, không tạo kết quả nghiên cứu.

```bash
pip install pandas matplotlib numpy
MINIBENCH_SYNTHETIC=1 python src/evaluate.py --fake
MINIBENCH_SYNTHETIC=1 python src/analysis.py
```

## Tái Lập

- Benchmark subset và perturbation dùng seed cố định.
- Generation dùng greedy decoding (`do_sample=False`).
- Mỗi generation cache theo hash của `model_id` và prompt.
- `results/per_item.csv` lưu prediction ở cấp từng item để audit lại cách chấm.
- `coverage.csv` cho biết khác biệt điểm có thể bị ảnh hưởng bởi lỗi extraction
  hay không.

## Giới Hạn

- Subset nhỏ nên kết quả có độ bất định cao; các chênh lệch nhỏ không nên được
  diễn giải như kết luận thống kê.
- Math specialist được thêm có chủ đích để kiểm tra rank instability và không
  thuộc panel model đa năng cùng hạng cân.
- MMLU và GSM8K có thể đã xuất hiện trong dữ liệu huấn luyện của các model.
- Demo chỉ phủ kiến thức tổng quát và toán, không đại diện cho mọi năng lực model.
- Greedy decoding và một prompt cố định không đo độ nhạy theo prompt hoặc sampling.

## Cấu Trúc Repo

```text
.
├── notebook/
│   ├── 1_smoke_test.ipynb
│   └── 2_run_all.ipynb
├── src/
│   ├── analysis.py
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── models.py
│   └── perturb.py
├── results/
│   ├── predictions/
│   └── figures/
├── requirements.txt
└── README.md
```
