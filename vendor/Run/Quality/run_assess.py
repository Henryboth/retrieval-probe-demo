"""
run_assess.py
=============
Compute and save the quality feature vector for one or all datasets.

Usage
-----
# Assess a single dataset
python Run_Quality/run_assess.py --dataset quality

# Assess all datasets
python Run_Quality/run_assess.py --dataset all

Output
------
Dataset/EvaluationData/QualityFeatures/{dataset}_clean.json
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from QualityCore import QualityAssessor

DATASETS = ["musique", "quality", "ultraDomain_legal", "graphragBench_medical"]


def assess(dataset_name: str) -> None:
    assessor = QualityAssessor(dataset_name, project_root=".")
    result = assessor.compute()
    print(result.report())
    out_path = Path(f"Dataset/EvaluationData/QualityFeatures/{dataset_name}_clean.json")
    result.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Compute quality features for a dataset")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name, or 'all' for all datasets")
    args = parser.parse_args()

    if args.dataset == "all":
        for name in DATASETS:
            corpus_path = Path("Dataset/RawData") / name / "Corpus.json"
            if corpus_path.exists():
                print(f"\n{'='*50}\nAssessing: {name}\n{'='*50}")
                assess(name)
            else:
                print(f"Skipping {name} (corpus not found)")
    else:
        assess(args.dataset)


if __name__ == "__main__":
    main()
