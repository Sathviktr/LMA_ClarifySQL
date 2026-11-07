"""Checkpoint 7: final inference with the three saved adapters, no retraining."""

from nbtools import code, md, write

CELLS = [
    md("""
    # ClarifySQL · Notebook 07 — Final evaluation

    This runs the fixed comparison on 188 test episodes and the 59 reviewed CoSQL turns.
    **No training is needed.** Use Kaggle GPU T4 x2 and enable Internet.

    The setup cell clones or updates `Sathviktr/LMA_ClarifySQL` from GitHub. Push the new evaluation
    files before running. The repository is public, so this does not require a token.

    Attach one input before starting: the saved **Notebook Output** of `sathviktr/phase6`, run **363373340**.
       It must contain `final_adapters/{sft_s1,sft_s2,a2_s1}` and `choice.json`.
       The 220 KB `checkpoint6_results.zip` is only metrics/logs and cannot replace these weights.

    Optional fallback: set `SOURCE = 'archive'` in the setup cell and attach the checkpoint-7 source ZIP.

    Save and run a version with **Save & Run All**. Keep the resulting version output.
    At the end download `checkpoint7_results.zip` and return it for analysis and the human review step.
    Existing final traces are resumed only when data, code, weights and settings match exactly.
    """),
    code("""
    import os, sys, json, zipfile, pathlib, subprocess, time, hashlib, shutil
    from pathlib import Path
    WORK = Path('/kaggle/working')
    INPUT = Path('/kaggle/input')
    assert WORK.exists(), 'Use Kaggle with GPU T4 x2 for this notebook.'
    REPO = WORK / 'LMA_ClarifySQL'
    SOURCE = 'github'  # 'archive' is an optional source-ZIP fallback
    REPO_URL = 'https://github.com/Sathviktr/LMA_ClarifySQL.git'
    BRANCH = 'main'
    REVISION = ''  # optional: exact commit printed by an earlier run, for reproduction
    if SOURCE == 'github':
        if (REPO / '.git').exists():
            subprocess.run(['git', '-C', str(REPO), 'fetch', 'origin', BRANCH], check=True)
            if not REVISION:
                subprocess.run(['git', '-C', str(REPO), 'merge', '--ff-only', f'origin/{BRANCH}'], check=True)
        else:
            assert not REPO.exists() or not any(REPO.iterdir()), 'Use a fresh session or SOURCE=archive for a ZIP checkout.'
            subprocess.run(['git', 'clone', '--branch', BRANCH, REPO_URL, str(REPO)], check=True)
        if REVISION:
            subprocess.run(['git', '-C', str(REPO), 'fetch', 'origin', REVISION], check=True)
            subprocess.run(['git', '-C', str(REPO), 'checkout', '--detach', REVISION], check=True)
        commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
        source_info = {'mode': 'github', 'commit': commit, 'branch': BRANCH}
        print('GitHub source commit:', commit)
    elif SOURCE == 'archive':
        sources = list(INPUT.rglob('clarifysql_checkpoint7_source.zip'))
        REPO.mkdir(exist_ok=True)
        if len(sources) == 1:
            with zipfile.ZipFile(sources[0]) as z:
                for member in z.infolist():
                    dest = (REPO / member.filename).resolve()
                    assert dest.is_relative_to(REPO.resolve()), 'Unsafe source archive path'
                z.extractall(REPO)
        else:
            # Kaggle may automatically unpack an uploaded source ZIP into the dataset.
            manifests = list(INPUT.rglob('CHECKPOINT7_SOURCE.json'))
            assert not sources and len(manifests) == 1, 'Attach exactly one checkpoint-7 source dataset.'
            source_root = manifests[0].parent.resolve()
            source_files = json.loads(manifests[0].read_text())['files']
            for relative in [*source_files, 'CHECKPOINT7_SOURCE.json']:
                src, dest = (source_root / relative).resolve(), (REPO / relative).resolve()
                assert src.is_relative_to(source_root) and dest.is_relative_to(REPO.resolve()), 'Unsafe source path'
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
        source_manifest = json.loads((REPO / 'CHECKPOINT7_SOURCE.json').read_text())
        for relative, expected in source_manifest['files'].items():
            assert hashlib.sha256((REPO / relative).read_bytes()).hexdigest() == expected, relative
        source_info = {'mode': 'archive', 'manifest_sha256': hashlib.sha256(
            (REPO / 'CHECKPOINT7_SOURCE.json').read_bytes()).hexdigest()}
        print('Source files verified:', len(source_manifest['files']))
    else:
        raise ValueError('SOURCE must be github or archive')
    required = ['scripts/run_final.py', 'src/clarifysql/eval/final_run.py',
                'requirements-eval.txt', 'data/episodes/slot_terms.json']
    missing = [p for p in required if not (REPO / p).is_file()]
    assert not missing, f'Evaluation code missing: {missing}. Push the new files to GitHub before running.'
    sys.path.insert(0, str(REPO / 'src'))
    os.environ['CLARIFYSQL_DATA'] = '/tmp/clarifysql_data'
    def checked(args, **kwargs):
        return subprocess.run(args, cwd=REPO, check=True, **kwargs)
    checked([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-eval.txt'])
    checked([sys.executable, '-m', 'pip', 'install', '-q', 'pytest', 'nbformat', 'numpy'])
    checked([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.', '--no-deps'])
    # The old preinstalled torchao can prevent PEFT from loading these adapters.
    checked([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'torchao'])
    """),
    md("""
    ## 1. Check adapters and prepare the shared inputs
    `ADAPTER_ROOT` can be set manually if several saved outputs are attached. The automatic search
    selects a folder only if it contains all three complete adapters and the saved choice file.
    """),
    code("""
    import torch
    assert torch.cuda.device_count() >= 2, 'Select GPU T4 x2 in the notebook settings.'
    print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    ADAPTER_ROOT = ''  # optional: exact attached final_adapters folder
    if not ADAPTER_ROOT:
        roots = [p.parent for p in INPUT.rglob('choice.json')
                 if all((p.parent / name / 'adapter_model.safetensors').is_file()
                        and (p.parent / name / 'adapter_config.json').is_file()
                        for name in ('sft_s1', 'sft_s2', 'a2_s1'))]
        assert len(roots) == 1, f'Attach the saved phase6 output, or set ADAPTER_ROOT. Candidates: {roots}'
        ADAPTER_ROOT = str(roots[0])
    ADAPTERS = Path(ADAPTER_ROOT)
    choice = json.loads((ADAPTERS / 'choice.json').read_text())
    assert choice['lr'] == '2e-4' and choice['epoch'] == 3, 'Unexpected adapter selection'
    print('Using:', ADAPTERS, 'chosen:', choice['chosen'])
    checked([sys.executable, 'scripts/download_data.py', '--only', 'spider'])
    from clarifysql.config import MODEL_ID
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID)  # download once, before starting two independent workers
    checked([sys.executable, '-m', 'pytest', 'tests/test_final_eval.py', 'tests/test_notebooks.py',
             '-p', 'no:cacheprovider', '-q'])
    """),
    md("""
    ## 2. Run the fixed conditions, one worker per GPU
    Completed batches are saved after every 16 episodes. Rerunning resumes a matching run;
    it refuses a run created with different inputs. Mock checks go in a separate directory and are never results.
    The no-repair oracle is a reference condition; it is not a guaranteed upper bound.
    """),
    code("""
    from concurrent.futures import ThreadPoolExecutor
    RESULT = REPO / 'results' / 'final'
    LOGS = WORK / 'checkpoint7_logs'
    LOGS.mkdir(exist_ok=True)
    # Optional: attach the previous result ZIP to resume a saved partial run.
    previous = list(INPUT.rglob('checkpoint7_results.zip'))
    assert len(previous) <= 1, 'Attach at most one earlier result archive for resume.'
    if previous and not RESULT.exists():
        with zipfile.ZipFile(previous[0]) as z:
            for member in z.infolist():
                if member.filename.startswith('results/final/'):
                    dest = (REPO / member.filename).resolve()
                    assert dest.is_relative_to(REPO.resolve()), 'Unsafe result archive path'
                    z.extract(member, REPO)
    jobs = [
        [('b1', 'b1', None, 'test'), ('b2', 'b2', None, 'test'),
         ('oracle', 'oracle', None, 'test'), ('b3', 'b3', None, 'test'),
         ('b3', 'b3', None, 'cosql')],
        [('sft', 'sft_s1', 'sft_s1', 'test'), ('sft', 'sft_s2', 'sft_s2', 'test'),
         ('a2', 'a2', 'a2_s1', 'test'), ('a3', 'a3', 'sft_s1', 'test'),
         ('sft', 'sft_s1', 'sft_s1', 'cosql')],
    ]
    def worker(gpu, conditions):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        times = []
        for system, name, adapter, split in conditions:
            args = [sys.executable, 'scripts/run_final.py', '--system', system, '--name', name,
                    '--split', split, '--output', str(RESULT)]
            if adapter:
                args += ['--adapter', str(ADAPTERS / adapter)]
            begin = time.time()
            with (LOGS / f'{split}_{name}.log').open('a') as log:
                checked(args, env=env, stdout=log, stderr=subprocess.STDOUT)
            times.append({'condition': f'{split}_{name}', 'wall_seconds': round(time.time()-begin, 1)})
            print('Finished', split, name, 'GPU', gpu, flush=True)
        return times
    begin = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, gpu, conditions) for gpu, conditions in enumerate(jobs)]
        timing = [item for future in futures for item in future.result()]
    (RESULT / 'compute.json').write_text(json.dumps({'conditions': timing,
        'wall_seconds': time.time()-begin, 'source': source_info,
        'adapter_source': 'sathviktr/phase6 run 363373340'}, indent=2))
    """),
    md("""
    ## 3. Score database variants and estimate uncertainty (CPU)
    The test-suite databases are larger than the main benchmark. This cell downloads them once.
    The evaluator receives predicted values unchanged. Whole databases are resampled for the intervals.
    """),
    code("""
    checked([sys.executable, 'scripts/download_data.py', '--only', 'testsuite'])
    checked([sys.executable, 'scripts/score_final_testsuite.py', '--root', str(RESULT)])
    checked([sys.executable, 'scripts/analyze_final.py', '--root', str(RESULT)])
    print((RESULT / 'analysis' / 'results.md').read_text())
    # These sheets contain blank human judgments. Do not open the key while rating questions.
    audit = REPO / 'annotation' / 'evaluation'
    if not audit.exists() or not any(audit.iterdir()):
        checked([sys.executable, 'scripts/make_eval_audits.py', '--root', str(RESULT)])
    """),
    md("## 4. Pack small outputs and return them"),
    code("""
    archive = WORK / 'checkpoint7_results.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for root in [RESULT, REPO / 'annotation' / 'evaluation']:
            if root.exists():
                for path in sorted(root.rglob('*')):
                    if path.is_file():
                        z.write(path, path.relative_to(REPO))
        for path in LOGS.glob('*.log'):
            z.write(path, 'logs/checkpoint7/' + path.name)
    print('Download:', archive, 'size:', archive.stat().st_size)
    """),
    md("""
    Download **checkpoint7_results.zip** from this saved version's Output and share it.
    Keep the final adapters attached/saved for the later live demo. Do not retrain to improve test scores.
    If a cell fails, share its error and the matching checkpoint7 log; completed trace batches can be reused.
    Human question ratings and simulator checks are a later team task, not something this notebook fills in.
    """),
]

if __name__ == "__main__":
    print(write(CELLS, "07_final_evaluation.ipynb"))
