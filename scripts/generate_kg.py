from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config, get_kg_llm_config
from src.main import DeterministicLLMClient
from src.kg.hybrid_kg_construction import HybridKGConstruction, is_dag
from src.utils.logger import AppLogger
from src.utils.llm_client import OpenAILLMClient


DATASETS = ["Logistics", "Mechanical_Physics", "PHP"]


def generate_kg_for_domain(domain: str, logger: AppLogger, mock_llm: bool = False):
    config = load_config(domain=domain)
    config = {**config, "llm": {**config.get("llm", {}), "mock": mock_llm}}
    data_config = config["data"]
    base_path = Path(data_config["base_path"]) / domain
    records_path = base_path / data_config["records_file"]
    problem_info_path = base_path / data_config["problem_info_file"]
    concept2id_path = base_path / data_config["concept2id_file"]
    kg_path = base_path / data_config["kg_structure_file"]

    llm_client = None
    llm_cfg = config.get("llm", {})
    if llm_cfg.get("mock", False):
        llm_client = DeterministicLLMClient()
    else:
        env_cfg = get_kg_llm_config()
        api_key = llm_cfg.get("api_key") or env_cfg.get("api_key")
        if api_key:
            from openai import OpenAI

            model = llm_cfg.get("model") or env_cfg.get("model")
            base_url = llm_cfg.get("base_url") or env_cfg.get("base_url")
            llm_client = OpenAILLMClient(OpenAI(api_key=api_key, base_url=base_url or None), model)

    builder = HybridKGConstruction(config, logger=logger, llm_client=llm_client)
    kg = builder.construct_hybrid_kg(
        text_data=[],
        records_path=str(records_path),
        problem_info_path=str(problem_info_path),
        concept2id_path=str(concept2id_path),
        existing_concepts=None,
    )
    kg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(kg_path, "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)
    logger.info(f"{domain}: saved KG to {kg_path} (DAG={is_dag(kg)})")


def main():
    parser = argparse.ArgumentParser(description="Generate CBEGRec knowledge graphs")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--mock_llm", action="store_true")
    args = parser.parse_args()

    logger = AppLogger("KG_Generator")
    for domain in args.datasets:
        generate_kg_for_domain(domain, logger, mock_llm=args.mock_llm)


if __name__ == "__main__":
    main()
