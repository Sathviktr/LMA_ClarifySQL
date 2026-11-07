"""Seven saved demo checks, followed by an optional interactive session; one GPU is enough."""

from nbtools import code, md, write
from build_07_final_evaluation import CELLS as EVAL_CELLS

CELLS = [
    md("""
    # ClarifySQL · Notebook 08 — Fictional university demo
    Use this after the final evaluation. The setup cell obtains the updated code from GitHub.
    Attach the saved output of `sathviktr/phase6` run 363373340. One T4 is enough. Enable Internet.
    With the default settings, **Save & Run All** finishes without asking for input and saves
    `/kaggle/working/checkpoint8_results.zip`. Return that ZIP for analysis. No training is needed.

    The seven automatic checks use fixed replies and made-up records. They test model behavior,
    including changed ranking preferences and age cutoffs; they are separate from the benchmark.
    Every record is made up. The model's question, explanation and SQL are displayed as plain text.
    The demo has no file/database upload and does not expose the sample-values tool.
    """),
    code(EVAL_CELLS[1].source.replace("Use Kaggle with GPU T4 x2 for this notebook.",
                                     "Use Kaggle with a GPU for this notebook.")),
    md("""
    ## 1. Load the selected adapter
    Keep `RUN_INTERACTIVE = False` for the first saved run. If Kaggle offers T4 x2, that also works;
    the model uses the first GPU. The attached output must contain the weights, not just the result ZIP.
    """),
    code("""
    RUN_INTERACTIVE = False  # enable later in an interactive session to try the web interface
    assert (REPO / 'src/clarifysql/demo/checks.py').is_file(), 'Update the repository before running notebook 08.'
    checked([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-demo.txt'])
    checked([sys.executable, '-m', 'pytest', 'tests/test_demo.py', 'tests/test_demo_checks.py',
             '-p', 'no:cacheprovider', '-q'])
    import torch
    assert torch.cuda.is_available(), 'Enable a T4 GPU.'
    roots = [p.parent for p in INPUT.rglob('choice.json')
             if (p.parent / 'sft_s1' / 'adapter_model.safetensors').is_file()
             and (p.parent / 'sft_s1' / 'adapter_config.json').is_file()]
    assert len(roots) == 1, 'Attach exactly one completed phase6 notebook output.'
    ADAPTER = roots[0] / 'sft_s1'
    choice = json.loads((roots[0] / 'choice.json').read_text())
    assert choice['lr'] == '2e-4' and choice['epoch'] == 3, 'Unexpected adapter selection.'
    from clarifysql.demo.university import create_database
    from clarifysql.llm.backend import HFBackend
    from clarifysql.eval.artifacts import write_json
    DEMO = WORK / 'checkpoint8_demo'
    DEMO.mkdir(exist_ok=True)
    db = create_database(DEMO / 'university.sqlite')
    model = HFBackend(adapter_path=str(ADAPTER))
    write_json(DEMO / 'source.json', {'source': source_info, 'choice': choice,
        'adapter_source': 'sathviktr/phase6 run 363373340', 'interactive_requested': RUN_INTERACTIVE})
    print('Loaded sft_s1 on:', torch.cuda.get_device_name(0))
    """),
    md("""
    ## 2. Run seven checks and save every outcome
    Clear requests should execute, the glossary should handle high marks, and missing email data should
    cause abstention. The ambiguous examples use predefined replies. The age and ranking variants check
    that a reply can change the answer. No simulated-user model is called here.

    A failed check is saved and does not prevent packaging. The rows, SQL, questions, replies, errors,
    model revision and source hashes are kept. A matching rerun resumes; changed inputs require a new
    `DEMO` directory. Extra result columns count as a mismatch under the strict reference check.
    """),
    code("""
    from clarifysql.demo.checks import run_demo_checks
    summary = run_demo_checks(model, db, DEMO / 'checks', adapter_path=ADAPTER)
    assert summary['backend'] == 'hf' and summary['complete'], 'Incomplete or mock results are not a completed GPU demo run.'
    print(f"Completed {summary['cases']} cases: {summary['passed']} passed.")
    print('Failed cases:', summary['failed'])
    """),
    md("""
    ## 3. Check interface construction and package the output
    Building the Gradio interface is recorded separately from a real browser interaction. If construction
    fails, its error is included in the ZIP. The automatic check does not claim a live user session occurred.
    Download **checkpoint8_results.zip** from this saved version's output. Model weights are not included.
    """),
    code("""
    from clarifysql.demo.app import create_app, launch
    app = None
    try:
        app = create_app(model, db)
        interface = {'constructed': True, 'browser_interaction_checked': False, 'error': None}
    except Exception as exc:
        interface = {'constructed': False, 'browser_interaction_checked': False,
                     'error': f'{type(exc).__name__}: {exc}'}
    write_json(DEMO / 'interface.json', interface)
    ZIP = WORK / 'checkpoint8_results.zip'
    with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(DEMO.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(DEMO).as_posix())
    print('Download:', ZIP, 'bytes:', ZIP.stat().st_size)
    print('Interface:', interface)
    from IPython.display import FileLink, display
    display(FileLink(str(ZIP)))
    """),
    md("""
    ## 4. Optional: try your own replies in the web interface
    For the first saved run this cell skips interaction, so the ZIP can be downloaded normally.
    To try the interface later, set `RUN_INTERACTIVE = True` and run this cell interactively.
    Choose a demo username and password, then confirm the password. These are the credentials for
    the demo login page. The cell closes the previous app and creates a fresh one with the already
    loaded model. Open the newly printed share link in a separate browser tab.
    These credentials are not saved in the result ZIP. Stop the notebook when finished.
    """),
    code("""
    if RUN_INTERACTIVE:
        assert 'model' in globals() and 'db' in globals(), 'Run the model-loading cell in this interactive session first.'
        import os
        from getpass import getpass
        from clarifysql.demo.app import create_app, launch
        demo_user = input('Choose demo username: ').strip()
        demo_password = getpass('Choose demo password: ')
        confirmation = getpass('Confirm demo password: ')
        if not demo_user or not demo_password:
            raise ValueError('Choose a nonempty demo username and password.')
        if demo_password != confirmation:
            raise ValueError('The passwords did not match. Run this cell again.')
        previous_app = globals().get('app')
        if previous_app is not None:
            previous_app.close()
        app = create_app(model, db)
        os.environ['CLARIFYSQL_DEMO_USER'] = demo_user
        os.environ['CLARIFYSQL_DEMO_PASSWORD'] = demo_password
        launch(app, share=True)
    else:
        print('Automatic run complete. Return checkpoint8_results.zip; no interactive input is needed.')
    """),
]

if __name__ == "__main__":
    print(write(CELLS, "08_demo.ipynb"))
