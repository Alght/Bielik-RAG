"""Calculate classification metrics and paired McNemar tests for each prompt."""

import csv
import math
import random
import re
import zlib
from collections import Counter

from parameters import (
    ABLATION_SCENARIOS,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    PROMPT_NAMES,
    ROOT,
    SCENARIOS,
    SET_SIZE,
)


SOURCE = ROOT / "06.summarized_results"
OUTPUT = ROOT / "14.statistics"
ABLATION_OUTPUT = ROOT / "15.ablation_analysis"
PROMPT_FILE = re.compile(rf"^results-key-{SET_SIZE}-prompt-(\d+)\.csv$")
OUTCOMES = {"TP", "TN", "FP", "FN", "ERROR"}
CLASSIFIED_OUTCOMES = {"TP", "TN", "FP", "FN"}
Z_95 = 1.959963984540054


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter=";")
        required = {"ID", "Key", *SCENARIOS}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Missing required columns in {path.name}")
        rows = list(reader)
    ids = [row["ID"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path.name}")
    return rows


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target, fieldnames=columns, delimiter=";", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def percentage(numerator, denominator):
    return f"{100 * numerator / denominator:.4f}" if denominator else ""


def wilson_interval(successes, total):
    if not total:
        return "", ""
    estimate = successes / total
    z_squared = Z_95**2
    denominator = 1 + z_squared / total
    center = (estimate + z_squared / (2 * total)) / denominator
    margin = Z_95 * math.sqrt(
        estimate * (1 - estimate) / total + z_squared / (4 * total**2)
    ) / denominator
    return f"{100 * (center - margin):.4f}", f"{100 * (center + margin):.4f}"


def configuration_statistics(prompt_id, scenario, outcomes):
    counts = Counter(outcomes.values())
    total = len(outcomes)
    classified = sum(counts[outcome] for outcome in CLASSIFIED_OUTCOMES)
    correct = counts["TP"] + counts["TN"]
    precision_denominator = counts["TP"] + counts["FP"]
    recall_denominator = counts["TP"] + counts["FN"]
    specificity_denominator = counts["TN"] + counts["FP"]
    precision = counts["TP"] / precision_denominator if precision_denominator else None
    recall = counts["TP"] / recall_denominator if recall_denominator else None
    specificity = counts["TN"] / specificity_denominator if specificity_denominator else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    lower, upper = wilson_interval(correct, total)
    return {
        "PromptID": prompt_id,
        "Configuration": scenario,
        "N": total,
        "TP": counts["TP"],
        "TN": counts["TN"],
        "FP": counts["FP"],
        "FN": counts["FN"],
        "Errors": counts["ERROR"],
        "CoveragePercent": percentage(classified, total),
        "ClassifiedAccuracyPercent": percentage(correct, classified),
        "EffectiveAccuracyPercent": percentage(correct, total),
        "Wilson95LowerPercent": lower,
        "Wilson95UpperPercent": upper,
        "PrecisionPercent": percentage(precision, 1) if precision is not None else "",
        "RecallPercent": percentage(recall, 1) if recall is not None else "",
        "SpecificityPercent": percentage(specificity, 1) if specificity is not None else "",
        "F1Percent": percentage(f1, 1) if f1 is not None else "",
    }


def exact_mcnemar_p(b, c):
    discordant = b + c
    if not discordant:
        return 1.0
    lower_tail = sum(math.comb(discordant, index) for index in range(min(b, c) + 1))
    return min(1.0, 2 * lower_tail / (2**discordant))


def holm_adjust(p_values):
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    family_size = len(p_values)
    for rank, index in enumerate(order):
        running_max = max(running_max, min(1.0, (family_size - rank) * p_values[index]))
        adjusted[index] = running_max
    return adjusted


def mcnemar_comparison(prompt_id, scenario, baseline, treatment):
    shared_ids = [
        input_id for input_id in baseline.keys() & treatment.keys()
        if baseline[input_id] and treatment[input_id]
    ]
    cells = Counter()
    for input_id in shared_ids:
        baseline_correct = baseline[input_id] in {"TP", "TN"}
        treatment_correct = treatment[input_id] in {"TP", "TN"}
        if baseline_correct and treatment_correct:
            cells["a"] += 1
        elif baseline_correct:
            cells["b"] += 1
        elif treatment_correct:
            cells["c"] += 1
        else:
            cells["d"] += 1
    b, c = cells["b"], cells["c"]
    return {
        "PromptID": prompt_id,
        "Reference": "norag",
        "Configuration": scenario,
        "N": len(shared_ids),
        "a_BothCorrect": cells["a"],
        "b_ReferenceCorrectTreatmentIncorrect": b,
        "c_ReferenceIncorrectTreatmentCorrect": c,
        "d_BothIncorrect": cells["d"],
        "DiscordantPairs": b + c,
        "ExactTwoSidedP": exact_mcnemar_p(b, c),
    }


def paired_bootstrap_interval(first_values, second_values, seed_key):
    if not first_values:
        return "", ""
    differences = [int(second) - int(first)
                   for first, second in zip(first_values, second_values)]
    rng = random.Random(BOOTSTRAP_SEED + zlib.crc32(seed_key.encode("utf-8")))
    estimates = sorted(
        100 * sum(rng.choices(differences, k=len(differences))) / len(differences)
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    lower_index = int(0.025 * (BOOTSTRAP_RESAMPLES - 1))
    upper_index = int(0.975 * (BOOTSTRAP_RESAMPLES - 1))
    return f"{estimates[lower_index]:.4f}", f"{estimates[upper_index]:.4f}"


def prompt_pair_comparison(scenario, prompt_a, prompt_b, endpoint, first_values, second_values):
    if len(first_values) != len(second_values):
        raise ValueError("Paired prompt outcomes must have matching lengths")
    both_positive = sum(first and second for first, second in zip(first_values, second_values))
    prompt_a_only = sum(first and not second for first, second in zip(first_values, second_values))
    prompt_b_only = sum(not first and second for first, second in zip(first_values, second_values))
    both_negative = sum(not first and not second for first, second in zip(first_values, second_values))
    sample_count = len(first_values)
    prompt_a_positive = sum(first_values)
    prompt_b_positive = sum(second_values)
    difference = (
        100 * (prompt_b_positive - prompt_a_positive) / sample_count
        if sample_count else None
    )
    interval_lower, interval_upper = paired_bootstrap_interval(
        first_values, second_values,
        f"{endpoint}|{scenario}|{prompt_a}|{prompt_b}",
    )
    return {
        "Scenario": scenario,
        "PromptA": prompt_a,
        "PromptAName": PROMPT_NAMES[prompt_a],
        "PromptB": prompt_b,
        "PromptBName": PROMPT_NAMES[prompt_b],
        "Endpoint": endpoint,
        "N": sample_count,
        "PromptAValuePercent": percentage(prompt_a_positive, sample_count),
        "PromptBValuePercent": percentage(prompt_b_positive, sample_count),
        "DifferenceBMinusAPercentagePoints": f"{difference:.4f}" if difference is not None else "",
        "PairedBootstrap95LowerPP": interval_lower,
        "PairedBootstrap95UpperPP": interval_upper,
        "a_BothPositive": both_positive,
        "b_PromptAPositivePromptBNegative": prompt_a_only,
        "c_PromptANegativePromptBPositive": prompt_b_only,
        "d_BothNegative": both_negative,
        "DiscordantPairs": prompt_a_only + prompt_b_only,
        "ExactTwoSidedP": exact_mcnemar_p(prompt_a_only, prompt_b_only),
    }


def cochran_q_paired(outcome_vectors):
    prompt_count = len(outcome_vectors)
    sample_count = len(outcome_vectors[0]) if outcome_vectors else 0
    if prompt_count < 2 or not sample_count:
        return 0.0, 1.0
    column_totals = [sum(values) for values in outcome_vectors]
    row_totals = [sum(values) for values in zip(*outcome_vectors)]
    total_positive = sum(column_totals)
    denominator = prompt_count * total_positive - sum(value**2 for value in row_totals)
    if denominator == 0:
        return 0.0, 1.0
    numerator = (prompt_count - 1) * (
        prompt_count * sum(value**2 for value in column_totals) - total_positive**2
    )
    statistic = numerator / denominator
    p_value = math.exp(-statistic / 2) if prompt_count == 3 else ""
    return statistic, p_value


def completed_ablation_scenarios():
    completed = set()
    for scenario in ABLATION_SCENARIOS:
        incomplete_runs = []
        for prompt_id in PROMPT_NAMES:
            run_name = "norag" if scenario == "norag" else f"rag-{int(scenario[2:]):02d}"
            path = ROOT / "02.inference_output" / (
                f"bielik11-{SET_SIZE}-cpu-{run_name}-prompt-{prompt_id}.csv"
            )
            if not path.exists():
                row_count = 0
            else:
                with path.open(encoding="utf-8-sig", newline="") as source:
                    reader = csv.DictReader(source, delimiter=";")
                    records = list(reader)
                ids = [record.get("ID", "") for record in records]
                row_count = len(ids) if len(ids) == len(set(ids)) else -1
            if row_count != SET_SIZE:
                incomplete_runs.append(f"Prompt {prompt_id}: {max(row_count, 0)}/{SET_SIZE}")
        if incomplete_runs:
            print(f"Skipping ablation {scenario}; incomplete source runs: {', '.join(incomplete_runs)}")
        else:
            completed.add(scenario)
    return completed


def run_prompt_ablation(outcomes_by_prompt, completed_scenarios):
    prompt_ids = tuple(sorted(outcomes_by_prompt, key=int))
    prompt_pairs = [
        (prompt_ids[first_index], prompt_ids[second_index])
        for first_index in range(len(prompt_ids))
        for second_index in range(first_index + 1, len(prompt_ids))
    ]
    summary_rows = []
    pairwise_rows = []
    omnibus_rows = []

    for scenario in ABLATION_SCENARIOS:
        if scenario not in completed_scenarios or len(prompt_ids) != 3 or any(
            not any(outcomes_by_prompt[prompt_id][scenario].values())
            for prompt_id in prompt_ids
        ):
            continue
        observed_ids = [
            {input_id for input_id, outcome in outcomes_by_prompt[prompt_id][scenario].items()
             if outcome}
            for prompt_id in prompt_ids
        ]
        shared_ids = sorted(set.intersection(*observed_ids))
        if not shared_ids:
            continue
        paired_outcomes = {
            prompt_id: [outcomes_by_prompt[prompt_id][scenario][input_id]
                        for input_id in shared_ids]
            for prompt_id in prompt_ids
        }

        for prompt_id in prompt_ids:
            row = configuration_statistics(
                prompt_id, scenario,
                dict(zip(shared_ids, paired_outcomes[prompt_id])),
            )
            row["PromptName"] = PROMPT_NAMES[prompt_id]
            summary_rows.append(row)

        outcome_vectors = {
            "effective_accuracy": {
                prompt_id: [outcome in {"TP", "TN"}
                            for outcome in paired_outcomes[prompt_id]]
                for prompt_id in prompt_ids
            },
            "coverage": {
                prompt_id: [outcome in CLASSIFIED_OUTCOMES
                            for outcome in paired_outcomes[prompt_id]]
                for prompt_id in prompt_ids
            },
        }
        jointly_classified_indices = [
            index for index in range(len(shared_ids))
            if all(paired_outcomes[prompt_id][index] in CLASSIFIED_OUTCOMES
                   for prompt_id in prompt_ids)
        ]
        outcome_vectors["classified_accuracy"] = {
            prompt_id: [
                paired_outcomes[prompt_id][index] in {"TP", "TN"}
                for index in jointly_classified_indices
            ]
            for prompt_id in prompt_ids
        }

        for endpoint, prompt_vectors in outcome_vectors.items():
            for prompt_a, prompt_b in prompt_pairs:
                pairwise_rows.append(prompt_pair_comparison(
                    scenario, prompt_a, prompt_b, endpoint,
                    prompt_vectors[prompt_a], prompt_vectors[prompt_b],
                ))
            statistic, p_value = cochran_q_paired(
                [prompt_vectors[prompt_id] for prompt_id in prompt_ids]
            )
            omnibus_rows.append({
                "Scenario": scenario,
                "Endpoint": endpoint,
                "N": len(next(iter(prompt_vectors.values()))),
                "CochranQ": f"{statistic:.6f}",
                "DegreesFreedom": len(prompt_ids) - 1,
                "AsymptoticP": f"{p_value:.8f}" if isinstance(p_value, float) else "",
            })

    for endpoint in ("effective_accuracy", "coverage", "classified_accuracy"):
        endpoint_rows = [row for row in pairwise_rows if row["Endpoint"] == endpoint]
        adjusted_p_values = holm_adjust([row["ExactTwoSidedP"] for row in endpoint_rows])
        for row, adjusted_p in zip(endpoint_rows, adjusted_p_values):
            row["HolmAdjustedP"] = adjusted_p
            row["HolmFamily"] = f"{endpoint}_all_pairs_at_shared_settings"
            row["HolmFamilySize"] = len(endpoint_rows)
            row["SignificantAt005"] = adjusted_p < 0.05

        omnibus_endpoint_rows = [
            row for row in omnibus_rows if row["Endpoint"] == endpoint
        ]
        omnibus_adjusted = holm_adjust([
            float(row["AsymptoticP"]) for row in omnibus_endpoint_rows
        ])
        for row, adjusted_p in zip(omnibus_endpoint_rows, omnibus_adjusted):
            row["HolmAdjustedP"] = adjusted_p
            row["HolmFamilySize"] = len(omnibus_endpoint_rows)
            row["SignificantAt005"] = adjusted_p < 0.05

    write_csv(
        ABLATION_OUTPUT / "prompt_ablation_summary.csv",
        (
            "PromptID", "PromptName", "Configuration", "N", "TP", "TN", "FP", "FN",
            "Errors", "CoveragePercent", "ClassifiedAccuracyPercent",
            "EffectiveAccuracyPercent", "Wilson95LowerPercent", "Wilson95UpperPercent",
            "PrecisionPercent", "RecallPercent", "SpecificityPercent", "F1Percent",
        ),
        summary_rows,
    )
    write_csv(
        ABLATION_OUTPUT / "prompt_ablation_pairwise_tests.csv",
        (
            "Scenario", "PromptA", "PromptAName", "PromptB", "PromptBName", "Endpoint",
            "N", "PromptAValuePercent", "PromptBValuePercent",
            "DifferenceBMinusAPercentagePoints", "PairedBootstrap95LowerPP",
            "PairedBootstrap95UpperPP", "a_BothPositive",
            "b_PromptAPositivePromptBNegative", "c_PromptANegativePromptBPositive",
            "d_BothNegative", "DiscordantPairs", "ExactTwoSidedP", "HolmAdjustedP",
            "HolmFamily", "HolmFamilySize", "SignificantAt005",
        ),
        pairwise_rows,
    )
    write_csv(
        ABLATION_OUTPUT / "prompt_ablation_omnibus_tests.csv",
        (
            "Scenario", "Endpoint", "N", "CochranQ", "DegreesFreedom", "AsymptoticP",
            "HolmAdjustedP", "HolmFamilySize", "SignificantAt005",
        ),
        omnibus_rows,
    )
    print(
        f"Wrote {len(summary_rows)} prompt-setting summaries, "
        f"{len(pairwise_rows)} paired comparisons, and "
        f"{len(omnibus_rows)} Cochran Q tests to {ABLATION_OUTPUT}"
    )


def main():
    prompt_files = []
    for path in sorted(SOURCE.glob(f"results-key-{SET_SIZE}-prompt-*.csv")):
        match = PROMPT_FILE.fullmatch(path.name)
        if match:
            prompt_files.append((int(match[1]), path))
    if not prompt_files:
        raise FileNotFoundError(f"No prompt summary CSVs found in {SOURCE}")

    metric_rows = []
    test_rows = []
    outcomes_by_prompt = {}
    for _, path in sorted(prompt_files):
        prompt_id = PROMPT_FILE.fullmatch(path.name)[1]
        rows = read_csv(path)
        outcomes_by_scenario = {
            scenario: {row["ID"]: row[scenario].strip().upper() for row in rows}
            for scenario in SCENARIOS
        }
        outcomes_by_prompt[prompt_id] = outcomes_by_scenario
        for scenario, outcomes in outcomes_by_scenario.items():
            for outcome in outcomes.values():
                if outcome and outcome not in OUTCOMES:
                    raise ValueError(f"Unsupported result {outcome!r} in {path.name}, {scenario}")
            observed = {input_id: outcome for input_id, outcome in outcomes.items() if outcome}
            if observed:
                metric_rows.append(configuration_statistics(prompt_id, scenario, observed))

        baseline = outcomes_by_scenario["norag"]
        comparisons = []
        for scenario in SCENARIOS[1:]:
            treatment = outcomes_by_scenario[scenario]
            if any(treatment.values()):
                comparisons.append(mcnemar_comparison(prompt_id, scenario, baseline, treatment))
        adjusted = holm_adjust([row["ExactTwoSidedP"] for row in comparisons])
        for row, adjusted_p in zip(comparisons, adjusted):
            row["HolmAdjustedP"] = adjusted_p
            row["HolmFamilySize"] = len(comparisons)
            row["SignificantAt005"] = adjusted_p < 0.05
            test_rows.append(row)

    write_csv(
        OUTPUT / "classification_statistics.csv",
        (
            "PromptID", "Configuration", "N", "TP", "TN", "FP", "FN", "Errors",
            "CoveragePercent", "ClassifiedAccuracyPercent", "EffectiveAccuracyPercent",
            "Wilson95LowerPercent", "Wilson95UpperPercent", "PrecisionPercent",
            "RecallPercent", "SpecificityPercent", "F1Percent",
        ),
        metric_rows,
    )
    write_csv(
        OUTPUT / "mcnemar_holm_results.csv",
        (
            "PromptID", "Reference", "Configuration", "N", "a_BothCorrect",
            "b_ReferenceCorrectTreatmentIncorrect", "c_ReferenceIncorrectTreatmentCorrect",
            "d_BothIncorrect", "DiscordantPairs", "ExactTwoSidedP", "HolmAdjustedP",
            "HolmFamilySize", "SignificantAt005",
        ),
        test_rows,
    )
    run_prompt_ablation(outcomes_by_prompt, completed_ablation_scenarios())
    print(
        f"Processed {len(prompt_files)} prompt summaries, {len(metric_rows)} configurations, "
        f"and {len(test_rows)} McNemar comparisons. Wrote results to {OUTPUT}"
    )


if __name__ == "__main__":
    main()
