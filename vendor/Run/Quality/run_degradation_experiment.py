"""
run_degradation_experiment.py
==============================
Phase 3 experiment: systematically degrade corpora and measure how
quality features change across degradation levels.

For each dataset x degradation type x rate combination:
  1. Apply the degradation
  2. Compute quality features on the degraded corpus
  3. Save results

Output
------
Dataset/EvaluationData/QualityFeatures/degradation/{dataset}/{type}_{rate}.json

Usage
-----
python Run_Quality/run_degradation_experiment.py --dataset quality
python Run_Quality/run_degradation_experiment.py --dataset all
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import pickle
from pathlib import Path

from QualityCore import (
    QualityAssessor,
    TextTruncation, EntityVariation,
    GraphEdgeRemoval, DocumentDropout, CombinedDegradation,
)

DATASETS     = ["musique", "quality"]
RATES        = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
DEGRADERS    = {
    "text_truncation":   TextTruncation,
    "entity_variation":  EntityVariation,
    "graph_edge_removal": GraphEdgeRemoval,
    "document_dropout":  DocumentDropout,
    "combined":          CombinedDegradation,
}


def _load_corpus(dataset_name: str) -> list:
    path = Path("Dataset/RawData") / dataset_name / "Corpus.json"
    with open(path) as f:
        return json.load(f)


def _load_graph(dataset_name: str):
    path = Path("Dataset/ProcessedData/Graph") / dataset_name / "graph.gpickle"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def run_dataset(dataset_name: str) -> None:
    print(f"\n{'#'*60}")
    print(f"  Degradation experiment: {dataset_name}")
    print(f"{'#'*60}")

    docs  = _load_corpus(dataset_name)
    graph = _load_graph(dataset_name)

    all_results = {}

    for deg_name, DegClass in DEGRADERS.items():
        print(f"\n  Degradation type: {deg_name}")
        type_results = {}

        for rate in RATES:
            degrader = DegClass(rate=rate, seed=42)
            deg_result = degrader.apply(docs, graph)

            # Compute quality features on the degraded corpus
            assessor = QualityAssessor(
                dataset_name,
                project_root=".",
                degraded_docs=deg_result.docs,
                degraded_graph=deg_result.graph,
            )
            quality_result = assessor.compute()

            entry = {
                "rate": rate,
                "degradation_type": deg_name,
                "degradation_stats": deg_result.stats,
                "quality_features": quality_result.to_dict(),
            }
            type_results[str(rate)] = entry

            print(f"    rate={rate:.1f}  overall_quality={quality_result.overall:.3f}"
                  f"  completeness={quality_result.completeness:.3f}"
                  f"  consistency={quality_result.consistency:.3f}")

            # Save individual result
            out_path = Path(
                f"Dataset/EvaluationData/QualityFeatures/degradation/"
                f"{dataset_name}/{deg_name}_{int(rate*100):03d}.json"
            )
            quality_result.save(out_path)

        all_results[deg_name] = type_results

    # Save combined summary for this dataset
    summary_path = Path(
        f"Dataset/EvaluationData/QualityFeatures/degradation/"
        f"{dataset_name}/summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Summary saved -> {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    args = parser.parse_args()

    targets = DATASETS if args.dataset == "all" else [args.dataset]
    for name in targets:
        corpus_path = Path("Dataset/RawData") / name / "Corpus.json"
        if corpus_path.exists():
            run_dataset(name)
        else:
            print(f"Skipping {name} (corpus not found)")

    print("\nDone. Results in Dataset/EvaluationData/QualityFeatures/degradation/")


if __name__ == "__main__":
    main()
