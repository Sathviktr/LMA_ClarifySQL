"""Run an agent system on the episodes of a split and score it.

Systems (proposal, Section 11):
  direct     B1 (first answer) and B2 (after up to 2 repairs): the request goes straight to the SQL writer;
             writes two result files, <split>_b1 and <split>_b2
  oracle     the original, fully specified question goes to the same SQL writer (upper bound)
  b3         the prompted agent: base model (adapter off) as the policy, instructions + worked examples
  b3zero     the same without the worked examples (a prompt variant, chosen between on validation only)
  sft        our system: the trained adapter as the policy               (needs --adapter)
  a2         the no-slot adapter (ablation A2)                           (needs --adapter)
  a3         the trained adapter with the full schema and glossary in the prompt (ablation A3)
  reference  follows each episode's reference path (sanity check of the controller and the scoring)

Usage (GPU):  python scripts/run_agent.py --system b3 --split validation
Laptop check: python -I scripts/run_agent.py --system reference --split validation --backend mock
Writes results/agent/<split>_<system>.jsonl (one trace per episode) and ..._summary.json.
"""

from __future__ import annotations

import argparse
import json
import time

from clarifysql.agent.controller import Controller, trace
from clarifysql.agent.prompts import b3_system_prompt
from clarifysql.config import results_dir
from clarifysql.episodes.schema import load_episodes
from clarifysql.eval.metrics import score_trace, summarize
from clarifysql.simuser.simulated_user import SimulatedUser


def direct_trace(rec: dict, ep, first: bool) -> dict:
    """A direct-SQL result in the same trace format as the agents (decision is always 'execute')."""
    sql = rec["first_pred_sql"] if first else rec["pred_sql"]
    status = rec["first_status"] if first else rec["status"]
    return {"episode": ep.id, "kind": ep.kind, "db_id": ep.db_id, "request": rec["request"], "decision": "execute",
            "gold_decision": ep.first_decision, "outcome": "executed", "steps": [], "rewritten_request": rec["request"],
            "sql": sql, "sql_status": status, "sql_error": None, "abstain_text": None,
            "calls": 1 if first else rec["model_calls"], "inspections": 0, "questions": 0,
            "repairs": 0 if first else rec["repairs"], "invalid_outputs": 0, "format_retries": 0,
            "blocked_attempts": int(status == "blocked"), "tokens_in": rec["prompt_tokens"],
            "tokens_out": rec["new_tokens"], "seconds": rec["seconds"], "events": []}


def save(scored: list[dict], split: str, name: str, meta: dict) -> dict:
    out = results_dir() / "agent"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{split}_{name}.jsonl", "w", encoding="utf-8") as f:
        for t in scored:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    summary = {**meta, "name": name, **summarize(scored)}
    (out / f"{split}_{name}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    keys = ("episodes", "execution_accuracy", "decision_macro_f1", "unnecessary_question_rate_clear",
            "correct_abstention_rate", "avg_clarification_turns", "invalid_output_rate", "limit_rate")
    print(name, json.dumps({k: summary[k] for k in keys}))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=["direct", "oracle", "b3", "b3zero", "sft", "a2", "a3", "reference"],
                    required=True)
    ap.add_argument("--split", choices=["validation", "test"], required=True)
    ap.add_argument("--backend", choices=["hf", "mock"], default="hf")
    ap.add_argument("--adapter", default=None, help="folder of a trained LoRA adapter (sft, a2, a3)")
    ap.add_argument("--name", default=None, help="name of the output files (default: the system)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--include-unwritten", action="store_true",
                    help="also run episodes whose request the team has not written yet")
    args = ap.parse_args()

    episodes = [ep for ep in load_episodes(args.split) if args.include_unwritten or not ep.draft.get("needs_writing")]
    episodes = episodes[: args.limit]
    by_id = {ep.id: ep for ep in episodes}
    suffix = "" if args.backend == "hf" else "_mock"
    meta = {"system": args.system, "split": args.split, "adapter": args.adapter, "backend": args.backend}
    if args.system in ("sft", "a2", "a3") and args.backend == "hf" and not args.adapter:
        ap.error(f"--adapter is required for {args.system}")

    if args.backend == "mock":
        from clarifysql.agent.mocks import GoldWriter, KeywordUser, ReferencePolicy

        with_slots = args.system != "a2"
        policy, writer, user_backend = ReferencePolicy(episodes, with_slots), GoldWriter(episodes), KeywordUser()
    else:
        from clarifysql.llm.backend import HFBackend

        policy = writer = user_backend = HFBackend(adapter_path=args.adapter)  # one model, three roles

    t0 = time.perf_counter()
    if args.system in ("direct", "oracle"):
        from clarifysql.data import spider
        from clarifysql.systems.direct import Item, run_direct

        items = [Item(ep.id, ep.db_id, (ep.source_question or ep.request) if args.system == "oracle" else ep.request,
                      ep.gold_sql) for ep in episodes]
        records = run_direct(items, writer, spider.db_path, max_repairs=2, batch_size=args.batch_size)
        meta["wall_seconds"] = round(time.perf_counter() - t0, 1)
        names = (("b1", True), ("b2", False)) if args.system == "direct" else (("oracle", True), ("oracle_repair", False))
        for name, first in names:
            scored = [score_trace(direct_trace(rec, by_id[rec["id"]], first), by_id[rec["id"]]) for rec in records]
            save(scored, args.split, (args.name or name) + suffix, meta)
        return

    with_slots = args.system != "a2"
    system_prompt = {"b3": b3_system_prompt(with_slots)}.get(args.system)
    controller = Controller(policy, writer, SimulatedUser(user_backend), with_slots=with_slots,
                            policy_adapter=args.system in ("sft", "a2", "a3") and args.backend == "hf",
                            policy_system=system_prompt, full_context=args.system == "a3",
                            batch_size=args.batch_size)
    runs = controller.run(episodes)
    meta["wall_seconds"] = round(time.perf_counter() - t0, 1)
    scored = [score_trace(trace(r), by_id[r.ep.id]) for r in runs]
    save(scored, args.split, (args.name or args.system) + suffix, meta)
    log = results_dir() / "agent" / f"{args.split}_{(args.name or args.system) + suffix}_user_log.jsonl"
    log.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in controller.user.log), encoding="utf-8")


if __name__ == "__main__":
    main()
