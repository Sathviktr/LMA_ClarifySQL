"""Frozen-condition evaluation in small durable batches, separate from development results."""

from __future__ import annotations

import json
from pathlib import Path

from clarifysql.config import REPO_ROOT, MODEL_ID
from clarifysql.eval.artifacts import RunStore, digest, write_json
from clarifysql.episodes.schema import load_episodes
from clarifysql.eval.metrics import summarize, score_trace

SYSTEMS = ("b1", "b2", "oracle", "b3", "sft", "a2", "a3")


def identity(system: str, split: str, backend: str, adapter: Path | None, ids: list[str], batch_size: int,
             db_ids: list[str]) -> dict:
    paths = [*sorted((REPO_ROOT / "src" / "clarifysql").rglob("*.py")),
             REPO_ROOT / "data/episodes/slot_terms.json", REPO_ROOT / "data/splits/splits.json"]
    paths.append(REPO_ROOT / ("annotation/cosql_dev_slice.csv" if split == "cosql"
                             else f"data/episodes/{split}.jsonl"))
    weights = {} if adapter is None else {p.name: digest(p) for p in sorted(adapter.iterdir()) if p.is_file()}
    if adapter is not None and not {"adapter_config.json", "adapter_model.safetensors"} <= weights.keys():
        raise ValueError(f"{adapter}: expected adapter_config.json and adapter_model.safetensors")
    from clarifysql.data import spider
    return {"system": system, "split": split, "backend": backend, "model": MODEL_ID, "ids": ids,
            "database_files": {db_id: digest(spider.db_path(db_id)) for db_id in sorted(set(db_ids))},
            "batch_size": batch_size, "files": {str(p.relative_to(REPO_ROOT)).replace('\\', '/'): digest(p)
                                                for p in paths}, "adapter_files": weights,
            "protocol": "checkpoint7-v1", "decoding": "greedy"}


def direct_rows(episodes, backend, system, batch_size):
    from clarifysql.data import spider
    from clarifysql.systems.direct import Item, run_direct

    items = [Item(ep.id, ep.db_id, (ep.source_question or ep.request) if system == "oracle" else ep.request,
                  ep.gold_sql) for ep in episodes]
    records = run_direct(items, backend, spider.db_path, max_repairs=2 if system == "b2" else 0,
                         batch_size=batch_size)
    out = []
    for ep, rec in zip(episodes, records):
        out.append({"episode": ep.id, "db_id": ep.db_id, "kind": ep.kind, "request": ep.request,
                    "decision": "execute", "gold_decision": ep.first_decision, "outcome": "executed",
                    "steps": [], "sql": rec["pred_sql"], "sql_status": rec["status"],
                    "sql_error": rec["error"], "abstain_text": None, "calls": rec["model_calls"],
                    "questions": 0, "inspections": 0, "repairs": rec["repairs"], "invalid_outputs": 0,
                    "format_retries": 0, "blocked_attempts": int(rec["status"] == "blocked"),
                    "tokens_in": rec["prompt_tokens"], "tokens_out": rec["new_tokens"], "seconds": rec["seconds"],
                    "events": [{"role": "writer", "sql": rec["pred_sql"], "raw": rec["raw"]}]})
    return out


def run_condition(system: str, split: str, output: Path, *, adapter: Path | None = None, name: str | None = None,
                  backend_name: str = "hf", backend=None, batch_size: int = 16, chunk_size: int = 16,
                  limit: int | None = None, use_testsuite: bool = False) -> dict:
    if batch_size < 1 or chunk_size < 1 or (limit is not None and limit < 1):
        raise ValueError("Batch, chunk and limit sizes must be positive")
    if split == "cosql":
        from clarifysql.eval.cosql_slice import load_slice
        episodes = load_slice()
        ids = [r["episode"] for r in episodes]
    else:
        episodes = load_episodes(split)
        ids = [ep.id for ep in episodes]
    episodes, ids = episodes[:limit], ids[:limit]
    if split == "cosql" and system not in ("b3", "sft"):
        raise ValueError("CoSQL: only b3 and sft are in the evaluation protocol")
    if system not in SYSTEMS:
        raise ValueError(system)
    if backend_name == "hf" and system in ("sft", "a2", "a3") and adapter is None:
        raise ValueError("A trained adapter is required")
    folder = Path(output) / ("mock" if backend_name == "mock" else "") / f"{split}_{name or system}"
    db_ids = [ep["db_id"] if split == "cosql" else ep.db_id for ep in episodes]
    config = identity(system, split, backend_name, adapter, ids, batch_size, db_ids)
    config["testsuite"] = use_testsuite
    config["chunk_size"] = chunk_size
    store = RunStore(folder, config)
    if not store.completed <= set(ids):
        raise ValueError("Saved traces contain unexpected episodes")
    pending = [ep for ep, id_ in zip(episodes, ids) if id_ not in store.completed]
    environment_path = folder / "environment.json"
    if pending and backend is None:
        if backend_name == "mock":
            if split == "cosql":
                raise ValueError("Use a scripted backend in tests; no research CoSQL mock is provided")
            from clarifysql.agent.mocks import ReferencePolicy, GoldWriter, KeywordUser
            policy, writer, user = ReferencePolicy(episodes, system != "a2"), GoldWriter(episodes), KeywordUser()
        else:
            from clarifysql.llm.backend import HFBackend
            policy = writer = user = HFBackend(adapter_path=str(adapter) if adapter else None)
            from clarifysql.pilot import env_report
            write_json(environment_path, {**env_report(), "model_revision": getattr(policy.model.config, "_commit_hash", None),
                                          "load_seconds": policy.load_seconds})
    else:
        policy = writer = user = backend
    for start in range(0, len(pending), chunk_size):
        chunk = pending[start:start + chunk_size]
        if split == "cosql":
            from clarifysql.eval.cosql_slice import run_slice
            scored = run_slice(chunk, policy, system, batch_size)
        else:
            if system in ("b1", "b2", "oracle"):
                traces = direct_rows(chunk, writer, system, batch_size)
            else:
                from clarifysql.agent.controller import Controller, trace
                from clarifysql.agent.prompts import b3_system_prompt
                from clarifysql.simuser.simulated_user import SimulatedUser
                controller = Controller(policy, writer, SimulatedUser(user), with_slots=system != "a2",
                                        policy_adapter=system != "b3" and backend_name != "mock",
                                        policy_system=b3_system_prompt() if system == "b3" else None,
                                        full_context=system == "a3", batch_size=batch_size)
                traces = [trace(r) for r in controller.run(chunk)]
            scored = [score_trace(t, ep) for t, ep in zip(traces, chunk)]
            if use_testsuite:
                from clarifysql.data import spider
                from clarifysql.eval.testsuite import testsuite_match
                for t, ep in zip(scored, chunk):
                    t["testsuite_correct"] = (bool(t["sql"]) and t["outcome"] == "executed" and
                        testsuite_match(ep.db_id, t["sql"], ep.gold_sql, spider.db_path(ep.db_id))) if ep.gold_sql else None
        store.append(scored)
        print(f"{folder.name}: {len(store.rows)}/{len(ids)} saved", flush=True)
    ordered = sorted(store.rows, key=lambda r: ids.index(r["episode"]))
    if split == "cosql":
        from clarifysql.eval.cosql_slice import summarize_slice
        stats = summarize_slice(ordered)
    else:
        stats = summarize(ordered)
        if use_testsuite:
            ans = [r for r in ordered if r["kind"] != "unanswerable"]
            stats["testsuite_accuracy"] = sum(r["testsuite_correct"] for r in ans) / len(ans) if ans else None
    summary = {"name": name or system, "system": system, "split": split, "backend": backend_name,
               "complete": len(ordered) == len(ids), "expected": len(ids), **stats}
    if split != "cosql":
        inspections = [step for r in ordered for step in r["steps"] if step["kind"] == "inspect"]
        vague = [(r, ep) for r, ep in zip(ordered, episodes) if ep.kind in ("ambiguous", "two_slot")]
        summary["capabilities"] = {
            "inspections": len(inspections),
            "valid_tool_rate": sum(bool(s.get("tool_ok")) for s in inspections) / len(inspections) if inspections else None,
            "premature_execution_rate_vague": sum(r["outcome"] == "executed" and
                len(r.get("answered_slots", [])) < len(ep.slots) for r, ep in vague) / len(vague) if vague else None,
            "repairs": sum(r["repairs"] for r in ordered),
            "note": "Premature execution proxy: executes before the simulated user supplied every hidden detail."
        }
    write_json(folder / "summary.json", summary)
    return summary
