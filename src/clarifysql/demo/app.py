"""Plain-text Gradio interface. Public sharing requires explicit credentials."""

import os
import threading
import uuid
from clarifysql.demo.session import LiveSession


def create_app(backend, db_path):
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    import gradio as gr
    sessions = {}
    lock = threading.Lock()

    def render(view):
        import pandas as pd
        return view["message"], view["history"], view["sql"], pd.DataFrame(view["rows"], columns=view["columns"])

    def start(request, key):
        with lock:
            if not request or not request.strip():
                return render({"message": "Enter a database question.", "history": "", "sql": "", "rows": [], "columns": []})
            if len(request) > 2000:
                raise gr.Error("Please keep requests under 2,000 characters.")
            # Bound memory even when browser sessions are abandoned.
            if len(sessions) >= 100 and key not in sessions:
                sessions.pop(next(iter(sessions)))
            sessions[key] = LiveSession(backend, db_path, request.strip())
            return render(sessions[key].advance())

    def reply(answer, key):
        with lock:
            if key not in sessions:
                raise gr.Error("Start a request first.")
            if len(answer or "") > 2000:
                raise gr.Error("Please keep replies under 2,000 characters.")
            return render(sessions[key].advance(answer))

    with gr.Blocks(analytics_enabled=False) as app:
        gr.Markdown("# ClarifySQL\nAsk about a small fictional university database. All records are made up.")
        key = gr.State(value=lambda: uuid.uuid4().hex)
        request = gr.Textbox(label="Your database question", lines=2)
        start_button = gr.Button("Start request")
        message = gr.Textbox(label="Assistant", interactive=False, lines=3)
        answer = gr.Textbox(label="Your clarification reply")
        reply_button = gr.Button("Send reply")
        history = gr.Textbox(label="Clarification history", interactive=False, lines=5)
        sql = gr.Textbox(label="Read-only SQL", interactive=False, lines=4)
        result = gr.Dataframe(label="Database results", interactive=False)
        gr.Examples([["List the names of all departments."], ["List students with high marks."],
                     ["Which is the best department?"], ["List older students with strong marks."],
                     ["List students' email addresses."]], request)
        outputs = [message, history, sql, result]
        start_button.click(start, [request, key], outputs, concurrency_id="model", concurrency_limit=1)
        reply_button.click(reply, [answer, key], outputs, concurrency_id="model", concurrency_limit=1)
    return app.queue(default_concurrency_limit=1)


def launch(app, share=False):
    auth = None
    if share:
        user, password = os.environ.get("CLARIFYSQL_DEMO_USER"), os.environ.get("CLARIFYSQL_DEMO_PASSWORD")
        if not user or not password:
            raise ValueError("Set CLARIFYSQL_DEMO_USER and CLARIFYSQL_DEMO_PASSWORD before enabling sharing")
        auth = (user, password)
    return app.launch(share=share, auth=auth, inbrowser=False)
