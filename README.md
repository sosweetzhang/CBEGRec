# CBEGRec

Core code for the paper *CBEGRec: Learning Path Recommendation via Concept Bundling and Exercise Generation*, published at `KDD 2026` [[Paper](https://doi.org/10.1145/3770855.3818035)]. 

Authors: [Haotian Zhang](https://scholar.google.com.hk/citations?user=N3V-QjAAAAAJ&hl=zh-CN), [Jinze Wu](https://orcid.org/0000-0001-9957-5733), [Qi Liu*](https://scholar.google.com.hk/citations?user=5EoHAFwAAAAJ&hl=zh-CN),et al.

Email: sosweetzhang@mail.ustc.edu.cn


## Datasets

The repository includes `Logistics`, `Mechanical_Physics`, and `PHP`.
The `Math / XES3G5M` dataset is not bundled because it is too large (https://github.com/ai4ed/XES3G5M).

## Main entry points

```bash
python scripts/generate_kg.py
python scripts/run_eval.py --config config/config_physics.yaml --variant full
python scripts/run_experiments.py --datasets Logistics Mechanical_Physics PHP --steps 5 10 20 --variants full wo_cb wo_eg wo_cbeg
python scripts/run_llm_effects.py --dataset Logistics --models qwen-plus gpt-4 claude-3 --steps 5 10 20
python scripts/export_human_eval_cases.py --datasets Logistics Mechanical_Physics PHP --num_cases 20
```

## Environment

```bash
pip install -r requirements.txt
```

Set `LLM_API_KEY` and `LLM_BASE_URL` for a real OpenAI-compatible backend.



