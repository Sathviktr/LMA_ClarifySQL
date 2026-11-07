"""Interactive, real-adapter demo after checkpoint 7; one GPU is enough."""

from nbtools import code, md, write
from build_07_final_evaluation import CELLS as EVAL_CELLS

CELLS = [
    md("""
    # ClarifySQL · Notebook 08 — Fictional university demo
    Use this after the final evaluation. The setup cell obtains the updated code from GitHub.
    Attach the saved output of `sathviktr/phase6` run 363373340. One T4 is enough. Enable Internet.
    Every record is made up. The model's question, explanation and SQL are displayed as plain text.
    The demo has no file/database upload and does not expose the sample-values tool.
    """),
    EVAL_CELLS[1],
    code("""
    checked([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-demo.txt'])
    import torch
    assert torch.cuda.is_available(), 'Enable a T4 GPU.'
    roots = [p.parent for p in INPUT.rglob('choice.json')
             if (p.parent / 'sft_s1' / 'adapter_model.safetensors').exists()]
    assert len(roots) == 1, 'Attach exactly one completed phase6 notebook output.'
    ADAPTER = roots[0] / 'sft_s1'
    from clarifysql.config import outputs_dir
    from clarifysql.demo.university import create_database
    from clarifysql.demo.app import create_app, launch
    from clarifysql.llm.backend import HFBackend
    db = create_database(outputs_dir() / 'demo/university.sqlite')
    model = HFBackend(adapter_path=str(ADAPTER))
    app = create_app(model, db)
    """),
    md("""
    ## Launch the demo
    Set a username and a password when prompted. The temporary share link requires these credentials.
    The password is not printed or stored in notebook source. Stop the notebook when finished.
    """),
    code("""
    from getpass import getpass
    os.environ['CLARIFYSQL_DEMO_USER'] = input('Demo username: ').strip()
    os.environ['CLARIFYSQL_DEMO_PASSWORD'] = getpass('Demo password: ')
    launch(app, share=True)
    """),
]

if __name__ == "__main__":
    print(write(CELLS, "08_demo.ipynb"))
