import argparse
import csv
import json
import re
from pathlib import Path


DEFAULT_ALGORITHMS = ["ERM", "CORAL", "GroupDRO", "IRM", "Mixup"]

DOMAIN_NAMES = {
    "PACS": ["Art", "Cartoon", "Photo", "Sketch"],
    # 如果后面跑 VLCS，可以先用通用名字；确认 DomainBed 顺序后再改
    "VLCS": ["Env0", "Env1", "Env2", "Env3"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=str, default="./outputs")
    parser.add_argument("--dataset", type=str, default="PACS")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=DEFAULT_ALGORITHMS,
        help="Algorithms to collect, e.g. ERM CORAL GroupDRO IRM Mixup",
    )
    parser.add_argument(
        "--select",
        choices=["final", "best_source_val"],
        default="final",
        help=(
            "final: use the last checkpoint record. "
            "best_source_val: select checkpoint with best average source-domain out_acc."
        ),
    )
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--md", type=str, default=None)
    return parser.parse_args()


def step_from_dirname(name):
    m = re.search(r"_steps(\d+)$", name)
    return int(m.group(1)) if m else -1


def find_result_file(outputs, dataset, alg, env_id, steps):
    outputs = Path(outputs)

    exact = outputs / f"{dataset}_{alg}_env{env_id}_steps{steps}" / "results.jsonl"
    if exact.exists():
        return exact

    # 如果 exact 找不到，则尝试找同算法同 env 的任意 steps 结果，选 steps 最大的那个
    pattern = f"{dataset}_{alg}_env{env_id}_steps*/results.jsonl"
    candidates = list(outputs.glob(pattern))
    if not candidates:
        return None

    candidates.sort(key=lambda p: step_from_dirname(p.parent.name), reverse=True)
    return candidates[0]


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def select_record(records, test_env, num_envs, mode):
    if not records:
        return None

    if mode == "final":
        return records[-1]

    if mode == "best_source_val":
        def source_val(record):
            vals = []
            for env_id in range(num_envs):
                if env_id == test_env:
                    continue
                key = f"env{env_id}_out_acc"
                if key in record and record[key] is not None:
                    vals.append(record[key])
            if not vals:
                return -1
            return sum(vals) / len(vals)

        return max(records, key=source_val)

    raise ValueError(f"Unknown selection mode: {mode}")


def fmt_percent(x):
    if x is None:
        return "-"
    return f"{100 * x:.2f}"


def print_markdown_table(rows, domain_names):
    headers = ["Method"] + domain_names + ["Avg", "Worst"]

    print("\nMarkdown table:")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |")

    for row in rows:
        values = [row["Method"]]
        for d in domain_names:
            values.append(fmt_percent(row.get(d)))
        values.append(fmt_percent(row.get("Avg")))
        values.append(fmt_percent(row.get("Worst")))
        print("| " + " | ".join(values) + " |")


def save_csv(rows, domain_names, path):
    headers = ["Method"] + domain_names + ["Avg", "Worst"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            out = {"Method": row["Method"]}
            for h in headers[1:]:
                out[h] = "" if row.get(h) is None else 100 * row[h]
            writer.writerow(out)


def save_markdown(rows, domain_names, path):
    headers = ["Method"] + domain_names + ["Avg", "Worst"]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |")

    for row in rows:
        values = [row["Method"]]
        for d in domain_names:
            values.append(fmt_percent(row.get(d)))
        values.append(fmt_percent(row.get("Avg")))
        values.append(fmt_percent(row.get("Worst")))
        lines.append("| " + " | ".join(values) + " |")

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()

    domain_names = DOMAIN_NAMES.get(args.dataset)
    if domain_names is None:
        raise ValueError(
            f"Unknown dataset {args.dataset}. Please add its domain names to DOMAIN_NAMES."
        )

    num_envs = len(domain_names)
    rows = []
    missing = []

    print(f"Collecting results for dataset={args.dataset}, steps={args.steps}, select={args.select}")
    print(f"Algorithms: {args.algorithms}")

    for alg in args.algorithms:
        row = {"Method": alg}
        accs = []

        for env_id, domain_name in enumerate(domain_names):
            result_file = find_result_file(
                args.outputs,
                args.dataset,
                alg,
                env_id,
                args.steps,
            )

            if result_file is None:
                row[domain_name] = None
                missing.append(f"{args.dataset}_{alg}_env{env_id}_steps{args.steps}")
                continue

            records = load_jsonl(result_file)
            record = select_record(records, env_id, num_envs, args.select)

            if record is None:
                row[domain_name] = None
                missing.append(str(result_file))
                continue

            key = f"env{env_id}_out_acc"
            acc = record.get(key)

            row[domain_name] = acc
            if acc is not None:
                accs.append(acc)

            selected_step = record.get("step", "unknown")
            print(
                f"{alg:10s} env{env_id} ({domain_name:8s}) "
                f"acc={fmt_percent(acc)} step={selected_step} file={result_file}"
            )

        if accs:
            row["Avg"] = sum(accs) / len(accs)
            row["Worst"] = min(accs)
        else:
            row["Avg"] = None
            row["Worst"] = None

        rows.append(row)

    print_markdown_table(rows, domain_names)

    csv_path = args.csv or f"{args.dataset.lower()}_results_{args.select}_steps{args.steps}.csv"
    md_path = args.md or f"{args.dataset.lower()}_results_{args.select}_steps{args.steps}.md"

    save_csv(rows, domain_names, csv_path)
    save_markdown(rows, domain_names, md_path)

    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved Markdown table to: {md_path}")

    if missing:
        print("\nWarning: Missing or unreadable runs:")
        for item in missing:
            print("  -", item)


if __name__ == "__main__":
    main()
