"""Recalculate classification and timing artifacts from the final inference runs."""

import csv
import re
from collections import defaultdict
from statistics import median

from parameters import ROOT, SCENARIOS, SET_SIZE


SOURCE = ROOT / "02.inference_output"
NAME = re.compile(rf"^bielik11-{SET_SIZE}-cpu-(norag|rag-(\d+))-prompt-(\d+)$")
TIMING = re.compile(
    r"Inference timing: input_id=(\d+) step=([a-z_]+) elapsed_seconds=(\d+(?:\.\d+)?)"
)
TOKEN_COUNT = re.compile(r"Prompt length: characters=\d+, input_tokens=(\d+),")
MODEL_CONTEXT_LIMIT = re.compile(
    r"Prompt length: characters=\d+, input_tokens=\d+, model_context_limit=(\d+),"
)
MAX_NEW_TOKENS = re.compile(r"effective_generation_config: .*['\"]max_new_tokens['\"]:\s*(\d+)")
ANSWER = re.compile(r"^(?:Result:\s*|Odpowiedź:\s*)?(TRUE|FALSE|TAK|NIE)(?!\w)", re.I)
QUESTION = "Czy czyn opisany w tekście jest zabroniony przez polski Kodeks Karny?"
PROHIBITED = "Czyn opisany w tekście jest zabroniony przez polski Kodeks Karny."
PROVISION = re.compile(r"^(?:art\.?|artykuł|§)\s*\d", re.I)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source, delimiter=";"))


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target, fieldnames=columns, delimiter=";", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def scenario_and_prompt(stem):
    match = NAME.fullmatch(stem)
    if not match:
        raise ValueError(f"Unrecognized run name: {stem}")
    scenario = "norag" if match[1] == "norag" else f"k={int(match[2])}"
    return scenario, match[3]


def classify(response, prompt_id):
    text = (response or "").strip()
    if text.startswith(QUESTION):
        text = text[len(QUESTION):].lstrip(" |:\t\r\n-")
    match = ANSWER.match(text)
    if match:
        return "positive" if match[1].upper() in ("TAK", "TRUE") else "negative"
    if text.startswith(PROHIBITED):
        return "positive"
    if prompt_id == "2" and PROVISION.match(text):
        return "positive"
    return "ERROR"


def outcome(key, classification):
    if classification == "ERROR":
        return "ERROR"
    return {
        ("positive", "positive"): "TP",
        ("negative", "negative"): "TN",
        ("negative", "positive"): "FP",
        ("positive", "negative"): "FN",
    }[(key, classification)]


def has_well_formed_output(response):
    text = response or ""
    start = 0
    while start < len(text) and not text[start].isalnum():
        start += 1
    answer = text[start:start + 3].upper()
    end = start + 3
    return answer in ("TAK", "NIE") and (end == len(text) or not text[end].isalnum())


def format_duration(seconds):
    if seconds > 10:
        return str(int(seconds + 0.5))
    return f"{seconds:.6f}"


def read_timings(path, keys):
    observed = {}
    for match in TIMING.finditer(path.read_text(encoding="utf-8", errors="replace")):
        input_id, stage, duration = match.groups()
        if input_id not in keys:
            raise ValueError(f"Unknown timing ID {input_id} in {path.name}")
        stages = observed.setdefault(input_id, {})
        if stage in stages:
            raise ValueError(f"Duplicate timing for ID {input_id}, stage {stage} in {path.name}")
        stages[stage] = float(duration)
    return observed


def read_token_counts(path, keys):
    observed = {}
    pending_input_id = None
    context_limits = set()
    generation_limits = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        context_limit = MODEL_CONTEXT_LIMIT.search(line)
        if context_limit:
            context_limits.add(int(context_limit.group(1)))
        generation_limit = MAX_NEW_TOKENS.search(line)
        if generation_limit:
            generation_limits.add(int(generation_limit.group(1)))

        timing = TIMING.search(line)
        if timing:
            input_id, stage, _ = timing.groups()
            if stage == "tokenization":
                if pending_input_id is not None:
                    raise ValueError(
                        f"Missing token count for ID {pending_input_id} in {path.name}"
                    )
                pending_input_id = input_id
            elif pending_input_id is not None:
                raise ValueError(
                    f"Missing token count for ID {pending_input_id} in {path.name}"
                )

        token_count = TOKEN_COUNT.search(line)
        if token_count:
            if pending_input_id is None:
                raise ValueError(f"Unmatched token count in {path.name}")
            if pending_input_id not in keys:
                raise ValueError(
                    f"Unknown token-count ID {pending_input_id} in {path.name}"
                )
            if pending_input_id in observed:
                raise ValueError(
                    f"Duplicate token count for ID {pending_input_id} in {path.name}"
                )
            observed[pending_input_id] = int(token_count.group(1))
            pending_input_id = None

    if pending_input_id is not None:
        raise ValueError(f"Missing token count for ID {pending_input_id} in {path.name}")
    if set(observed) != set(keys):
        missing = sorted(set(keys) - set(observed), key=int)
        raise ValueError(
            f"Missing token counts for {len(missing)} IDs in {path.name}: {missing[:10]}"
        )
    if len(context_limits) != 1:
        raise ValueError(
            f"Expected one model context limit in {path.name}, found {context_limits}"
        )
    if len(generation_limits) != 1:
        raise ValueError(
            f"Expected one max_new_tokens limit in {path.name}, found {generation_limits}"
        )
    return observed, context_limits.pop(), generation_limits.pop()


def write_token_statistics(keys, logs):
    detail_rows = []
    summary_rows = []
    for path in logs:
        scenario, prompt_id = scenario_and_prompt(path.stem)
        token_counts, context_limit, generation_limit = read_token_counts(path, keys)
        detail_rows.extend(
            {
                "Run": path.stem,
                "PromptID": prompt_id,
                "Scenario": scenario,
                "ID": input_id,
                "InputTokens": token_counts[input_id],
            }
            for input_id in keys
        )
        values = list(token_counts.values())
        summary_rows.append(
            {
                "Run": path.stem,
                "PromptID": prompt_id,
                "Scenario": scenario,
                "CaseCount": len(values),
                "MinimumInputTokens": min(values),
                "MedianInputTokens": f"{median(values):g}",
                "MaximumInputTokens": max(values),
                "ModelContextLimit": context_limit,
                "MaxNewTokens": generation_limit,
            }
        )
    write_csv(
        ROOT / "16.token_statistics" / "token_statistics.csv",
        ("Run", "PromptID", "Scenario", "ID", "InputTokens"),
        detail_rows,
    )
    write_csv(
        ROOT / "16.token_statistics" / "tokens.csv",
        (
            "Run",
            "PromptID",
            "Scenario",
            "CaseCount",
            "MinimumInputTokens",
            "MedianInputTokens",
            "MaximumInputTokens",
            "ModelContextLimit",
            "MaxNewTokens",
        ),
        summary_rows,
    )


def main():
    reference = read_csv(ROOT / "01.key" / f"key-{SET_SIZE}.csv")
    keys = {row["ID"]: row["Key"].strip().lower() for row in reference}
    if len(keys) != len(reference) or set(keys.values()) != {"positive", "negative"}:
        raise ValueError("The reference key has duplicate IDs or unsupported labels")

    summaries = defaultdict(dict)
    errors = defaultdict(list)
    quality = []
    runs = set()
    for source in sorted(SOURCE.glob("*.csv")):
        scenario, prompt_id = scenario_and_prompt(source.stem)
        if scenario not in SCENARIOS or (prompt_id, scenario) in runs:
            raise ValueError(f"Duplicate or unsupported run: {source.name}")
        runs.add((prompt_id, scenario))
        records = read_csv(source)
        if any("ID" not in row or "Bielik(treść)" not in row or None in row for row in records):
            raise ValueError(f"Malformed CSV: {source.name}")
        by_id = {row["ID"]: row for row in records}
        if len(by_id) != len(records) or set(by_id) - keys.keys():
            raise ValueError(f"Duplicate or unknown IDs in {source.name}")

        clean_columns = [column for column in records[0] if column != "Notatka"] if records else ["ID", "Bielik(treść)"]
        cleared_output = ROOT / "03.cleared_inference_output" / source.name
        write_csv(
            cleared_output,
            clean_columns,
            ({column: row.get(column, "") for column in clean_columns} for row in records),
        )
        cleared_records = read_csv(cleared_output)
        well_formed_count = sum(
            has_well_formed_output(row.get("Bielik(treść)"))
            for row in cleared_records
        )
        total_count = len(cleared_records)
        quality.append(
            {
                "File": source.name,
                "TotalRecords": total_count,
                "WellFormedRecords": well_formed_count,
                "MalformedRecords": total_count - well_formed_count,
                "WellFormedPercent": f"{well_formed_count / total_count * 100:.2f}" if total_count else "0.00",
            }
        )
        classified = []
        run_errors = 0
        for input_id, key in keys.items():
            response = by_id.get(input_id, {}).get("Bielik(treść)")
            classification = classify(response, prompt_id)
            calc = outcome(key, classification)
            classified.append(
                {"PromptID": prompt_id, "ID": input_id, "Key": key,
                 "Classification": classification, "Calc": calc}
            )
            summaries[prompt_id].setdefault(input_id, {})[scenario] = calc
            if calc == "ERROR":
                run_errors += 1
                errors[prompt_id].append(
                    {"ID": input_id, "ClassificationType": scenario,
                     "Result": response if response else "MISSING" if input_id not in by_id else "EMPTY"}
                )
        write_csv(
            ROOT / "04.classification" / source.name.replace("bielik11-", "result-"),
            ("PromptID", "ID", "Key", "Classification", "Calc"),
            classified,
        )
        print(f"{source.name}: {len(records)} responses, {run_errors} classification errors")

    write_csv(
        ROOT / "13.output_quality" / "output_quality_summary.csv",
        ("File", "TotalRecords", "WellFormedRecords", "MalformedRecords", "WellFormedPercent"),
        quality,
    )

    for prompt_id in sorted(summaries):
        write_csv(
            ROOT / "06.summarized_results" / f"results-key-{SET_SIZE}-prompt-{prompt_id}.csv",
            ("ID", "Key", *SCENARIOS),
            ({"ID": input_id, "Key": key,
              **{scenario: summaries[prompt_id].get(input_id, {}).get(scenario, "")
                 for scenario in SCENARIOS}} for input_id, key in keys.items()),
        )
        write_csv(
            ROOT / "05.classification_errors" / f"prompt-{prompt_id}-errors.csv",
            ("ID", "ClassificationType", "Result"),
            errors[prompt_id],
        )

    stats = defaultdict(list)
    logs = sorted((ROOT / "10.inference_logs").glob("*.log"))
    timings = {path: read_timings(path, keys) for path in logs}
    write_token_statistics(keys, logs)
    stage_names = list(dict.fromkeys(
        stage for observed in timings.values() for stages in observed.values()
        for stage in stages if stage != "total"
    ))
    stage_columns = {stage: "".join(word.title() for word in stage.split("_")) + "Seconds"
                     for stage in stage_names}
    for path in logs:
        scenario, prompt_id = scenario_and_prompt(path.stem)
        observed = timings[path]
        write_csv(
            ROOT / "11.inference_stats" / f"{path.stem}_metrics.csv",
            ("ID", *stage_columns.values(), "InferenceSeconds"),
            ({"ID": input_id,
                            **{column: format_duration(stages[stage]) if stage in stages else ""
                 for stage, column in stage_columns.items()},
                            "InferenceSeconds": format_duration(stages["total"]) if "total" in stages else ""}
             for input_id, stages in observed.items()),
        )
        for stage in (*stage_names, "total"):
            durations = [stages[stage] for stages in observed.values() if stage in stages]
            if durations:
                stats[prompt_id].append(
                    {"Scenario": scenario, "Stage": stage, "Count": len(durations),
                     "MinimumSeconds": format_duration(min(durations)),
                     "MaximumSeconds": format_duration(max(durations)),
                     "AverageSeconds": format_duration(sum(durations) / len(durations))}
                )
    for prompt_id in sorted(summaries):
        write_csv(
            ROOT / "11.inference_stats" / f"prompt-{prompt_id}-statistics.csv",
            ("Scenario", "Stage", "Count", "MinimumSeconds", "MaximumSeconds", "AverageSeconds"),
            sorted(stats[prompt_id], key=lambda row: (SCENARIOS.index(row["Scenario"]),
                                                      (*stage_names, "total").index(row["Stage"]))),
        )


if __name__ == "__main__":
    main()