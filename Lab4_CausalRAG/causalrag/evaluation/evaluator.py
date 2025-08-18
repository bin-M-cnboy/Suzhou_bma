import os
import json
import re
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass

try:
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_relevancy,
        context_recall
    )
    from ragas.metrics.critique import harmfulness
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logging.warning("Ragas package not installed. Install with: pip install ragas")

# Import LLM interface for critique-based evaluation
try:
    from ..generator.llm_interface import LLMInterface
except ImportError:
    try:
        # Alternative import path
        from causalrag.generator.llm_interface import LLMInterface
    except ImportError:
        # Handle case where module is used standalone
        LLMInterface = None

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    metrics: Dict[str, float]
    detailed_scores: Dict[str, List[float]]
    error_analysis: Optional[Dict[str, Any]] = None
    raw_evaluations: Optional[Dict[str, Any]] = None

class CausalEvaluator:

    def __init__(
        self,
        llm_interface=None,
        metrics: List[str] = None,
        use_ragas: bool = True,
        results_dir: Optional[str] = None
    ):
        self.llm_interface = llm_interface
        self.use_ragas = use_ragas and RAGAS_AVAILABLE

        self.default_metrics = [
            "faithfulness",
            "answer_relevancy",
            "context_relevancy",
            "context_recall",
            "causal_consistency",
            "causal_completeness"
        ]

        self.metrics = metrics or self.default_metrics
        if results_dir:
            self.results_dir = Path(results_dir)
            os.makedirs(self.results_dir, exist_ok=True)
        else:
            self.results_dir = None

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        causal_paths: List[List[List[str]]] = None,
        ground_truths: Optional[List[str]] = None
    ) -> EvaluationResult:
        all_metrics = {}
        detailed_scores = {}
        error_analysis = {}
        raw_results = {}

        if len(questions) != len(answers) or len(questions) != len(contexts):
            raise ValueError("Number of questions, answers, and contexts must match")
        if causal_paths is not None and len(questions) != len(causal_paths):
            raise ValueError("Number of questions and causal paths must match")
        if self.use_ragas:
            ragas_results = self._run_ragas_evaluation(
                questions, answers, contexts, ground_truths
            )
            if ragas_results:
                all_metrics.update(ragas_results["metrics"])
                detailed_scores.update(ragas_results["detailed"])
                raw_results["ragas"] = ragas_results["raw"]
        if causal_paths:
            causal_results = self._evaluate_causal_reasoning(
                questions, answers, contexts, causal_paths, ground_truths
            )
            if causal_results:
                all_metrics.update(causal_results["metrics"])
                detailed_scores.update(causal_results["detailed"])
                error_analysis.update(causal_results.get("errors", {}))
                raw_results["causal"] = causal_results["raw"]
        if self.llm_interface:
            llm_results = self._run_llm_evaluations(
                questions, answers, contexts, causal_paths, ground_truths
            )
            if llm_results:
                all_metrics.update(llm_results["metrics"])
                detailed_scores.update(llm_results["detailed"])
                error_analysis.update(llm_results.get("errors", {}))
                raw_results["llm_critique"] = llm_results["raw"]
        result = EvaluationResult(
            metrics=all_metrics,
            detailed_scores=detailed_scores,
            error_analysis=error_analysis,
            raw_evaluations=raw_results
        )
        if self.results_dir:
            self._save_results(result)
        return result

    def _run_ragas_evaluation(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]]
    ) -> Dict[str, Any]:
        if not self.use_ragas:
            return {}

        try:
            flattened_contexts = [" ".join(ctx) for ctx in contexts]
            ragas_metrics = []

            if "faithfulness" in self.metrics:
                ragas_metrics.append(faithfulness)
            if "answer_relevancy" in self.metrics:
                ragas_metrics.append(answer_relevancy)
            if "context_relevancy" in self.metrics:
                ragas_metrics.append(context_relevancy)
            if "context_recall" in self.metrics and ground_truths:
                ragas_metrics.append(context_recall)
            if not ragas_metrics:
                return {}

            dataset_dict = {
                "question": questions,
                "answer": answers,
                "contexts": [[ctx] for ctx in flattened_contexts],
            }

            if ground_truths:
                dataset_dict["ground_truth"] = ground_truths
            result = evaluate(dataset_dict, metrics=ragas_metrics)

            metrics_dict = {}
            detailed_dict = {}

            for column in result.columns:
                if column in {"question", "answer", "contexts", "ground_truth"}:
                    continue
                metrics_dict[column] = float(result[column].mean())
                detailed_dict[column] = result[column].tolist()
            return {
                "metrics": metrics_dict,
                "detailed": detailed_dict,
                "raw": result.to_dict()
            }

        except Exception as e:
            logger.error(f"Error in Ragas evaluation: {e}")
            return {}

    def _evaluate_causal_reasoning(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        causal_paths: List[List[List[str]]],
        ground_truths: Optional[List[str]]
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "metrics": {},
            "detailed": {},
            "errors": {},
            "raw": {}
        }

        if not self.llm_interface:
            return results

        try:
            if "causal_consistency" in self.metrics:
                consistency_scores = []

                for i, (question, answer, paths) in enumerate(
                    zip(questions, answers, causal_paths)
                ):
                    if not paths:
                        consistency_scores.append(1.0)
                        continue
                    paths_text = "\n".join([" -> ".join(path) for path in paths])    ###
                    prompt = (
                        f"Evaluate if the answer respects the causal relationships provided.\n\n"
                        f"Question: {question}\n\n"
                        f"Causal relationships that should be respected:\n{paths_text}\n\n"
                        f"Answer to evaluate:\n{answer}\n\n"
                        "On a scale of 0-10, how well does the answer respect these causal relationships?\n"
                        "- Score 0: Answer directly contradicts the causal relationships\n"
                        "- Score 5: Answer is neutral or doesn't address the causal relationships\n"
                        "- Score 10: Answer perfectly aligns with and explains the causal relationships\n\n"
                        "Provide your rating as a number from 0-10, followed by a brief explanation.\nRating:"
                    )
                    response = self.llm_interface.generate(prompt, temperature=0.1)

                    try:
                        first_line = response.strip().splitlines()[0]
                        score_match = re.search(r"(\d+(?:\.\d+)?)", first_line)
                        score = float(score_match.group(1)) / 10.0 if score_match else 0.5
                        score = max(0.0, min(score, 1.0))
                    except Exception:
                        score = 0.5
                    consistency_scores.append(score)

                results["metrics"]["causal_consistency"] = float(np.mean(consistency_scores))
                results["detailed"]["causal_consistency"] = consistency_scores

            if "causal_completeness" in self.metrics:
                completeness_scores = []
                for i, (question, answer, paths) in enumerate(
                    zip(questions, answers, causal_paths)
                ):
                    if not paths:
                        completeness_scores.append(1.0)
                        continue
                    all_factors = {factor for path in paths for factor in path}
                    factors_text = "\n".join(f"- {f}" for f in all_factors)
                    prompt = (
                        f"Evaluate if the answer addresses all important causal factors.\n\n"
                        f"Question: {question}\n\n"
                        f"Important causal factors that should be addressed:\n{factors_text}\n\n"
                        f"Answer to evaluate:\n{answer}\n\n"
                        "On a scale of 0-10, how completely does the answer address these important causal factors?\n"
                        "- Score 0: Answer misses all important causal factors\n"
                        "- Score 5: Answer addresses some factors but misses several important ones\n"
                        "- Score 10: Answer comprehensively addresses all important causal factors\n\n"
                        "Provide your rating as a number from 0-10, followed by a brief explanation of which factors were missed (if any).\nRating:"
                    )
                    response = self.llm_interface.generate(prompt, temperature=0.1)

                    try:
                        first_line = response.strip().splitlines()[0]
                        score_match = re.search(r"(\d+(?:\.\d+)?)", first_line)
                        score = float(score_match.group(1)) / 10.0 if score_match else 0.5
                        score = max(0.0, min(score, 1.0))
                    except Exception:
                        score = 0.5
                    completeness_scores.append(score)
                results["metrics"]["causal_completeness"] = float(np.mean(completeness_scores))
                results["detailed"]["causal_completeness"] = completeness_scores
            return results

        except Exception as e:
            logger.error(f"Error in causal evaluation: {e}")
            results["errors"]["causal_evaluation"] = str(e)
            return results

    def _run_llm_evaluations(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        causal_paths: Optional[List[List[List[str]]]],
        ground_truths: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Run general LLM-based evaluations"""
        if not self.llm_interface:
            return {}

        results: Dict[str, Any] = {
            "metrics": {},
            "detailed": {},
            "errors": {},
            "raw": {}
        }

        try:
            if "answer_quality" in self.metrics:
                quality_scores = []

                for i, (question, answer, context) in enumerate(
                    zip(questions, answers, contexts)
                ):
                    context_text = "\n".join(
                        f"[{j+1}] {c}" for j, c in enumerate(context)
                    )
                    gt_text = (
                        f"\nGround truth answer:\n{ground_truths[i]}"
                        if ground_truths and i < len(ground_truths)
                        else ""
                    )
                    prompt = (
                        f"Evaluate the quality of this answer based on the question and provided context.\n\n"
                        f"Question: {question}\n\n"
                        f"Context:\n{context_text}{gt_text}\n\n"
                        f"Answer to evaluate:\n{answer}\n\n"
                        "Rate the answer on a scale of 0-10 based on:\n"
                        "1. Accuracy - Does it correctly use information from the context?\n"
                        "2. Completeness - Does it address all aspects of the question?\n"
                        "3. Conciseness - Is it appropriately detailed without unnecessary information?\n"
                        "4. Coherence - Is it well-structured and logical?\n\n"
                        "Provide your overall rating as a number from 0-10.\nOverall rating:"
                    )
                    response = self.llm_interface.generate(prompt, temperature=0.1)

                    try:
                        first_line = response.strip().splitlines()[0]
                        score_match = re.search(r"(\d+(?:\.\d+)?)", first_line)
                        score = float(score_match.group(1)) / 10.0 if score_match else 0.7
                        score = max(0.0, min(score, 1.0))
                    except Exception:
                        score = 0.7
                    quality_scores.append(score)

                results["metrics"]["answer_quality"] = float(np.mean(quality_scores))
                results["detailed"]["answer_quality"] = quality_scores

            return results

        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}")
            results["errors"]["llm_evaluation"] = str(e)
            return results

    def _save_results(self, result: EvaluationResult) -> None:
        """Save evaluation results to disk"""
        if not self.results_dir:
            return

        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            metrics_file = self.results_dir / f"metrics_summary_{timestamp}.json"
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(result.metrics, f, ensure_ascii=False, indent=2)

            detailed_file = self.results_dir / f"detailed_scores_{timestamp}.json"
            with open(detailed_file, "w", encoding="utf-8") as f:
                json.dump(result.detailed_scores, f, ensure_ascii=False, indent=2)

            if result.error_analysis:
                errors_file = self.results_dir / f"error_analysis_{timestamp}.json"
                with open(errors_file, "w", encoding="utf-8") as f:
                    json.dump(result.error_analysis, f, ensure_ascii=False, indent=2)

            report_file = self.results_dir / f"evaluation_report_{timestamp}.md"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write("# CausalRAG Evaluation Report\n\n")
                f.write(f"**Date:** {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
                f.write("## Metrics Summary\n\n")
                for metric, score in result.metrics.items():
                    f.write(f"- **{metric}:** {score:.4f}\n")

                f.write("\n## Metric Details\n\n")
                for metric, scores in result.detailed_scores.items():
                    avg = np.mean(scores)
                    med = np.median(scores)
                    min_val = np.min(scores)
                    max_val = np.max(scores)
                    f.write(f"### {metric}\n")
                    f.write(f"- **Average:** {avg:.4f}\n")
                    f.write(f"- **Median:** {med:.4f}\n")
                    f.write(f"- **Min:** {min_val:.4f}\n")
                    f.write(f"- **Max:** {max_val:.4f}\n\n")

            logger.info(f"Evaluation results saved to {self.results_dir}")

        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")

    @classmethod
    def evaluate_pipeline(
        cls,
        pipeline,
        eval_data: List[Dict[str, str]],
        metrics: List[str] = None,
        llm_interface=None,
        results_dir: Optional[str] = None,
    ) -> EvaluationResult:
        evaluator = cls(
            llm_interface=llm_interface or getattr(pipeline, "llm", None),
            metrics=metrics,
            results_dir=results_dir,
        )

        questions = [item["question"] for item in eval_data]
        ground_truths = [item.get("ground_truth") for item in eval_data if "ground_truth" in item]

        if not ground_truths:
            ground_truths = None

        answers, contexts, causal_paths = [], [], []
        for question in questions:
            result = pipeline.run(question)
            answers.append(result["answer"])
            contexts.append(result.get("context", []))
            causal_paths.append(result.get("causal_paths", []))

        return evaluator.evaluate(
            questions=questions,
            answers=answers,
            contexts=contexts,
            causal_paths=causal_paths,
            ground_truths=ground_truths,
        )