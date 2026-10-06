import gradio as gr
import pandas as pd
from typing import Tuple, List, Dict, Any
from pipeline_orchestrator import process_user_text_to_tasks, PipelineExecutionError
from database import init_db, insert_tasks, get_all_tasks, get_connection


init_db()



def fetch_tasks_dataframe(status_filter: str = "All", category_filter: str = "All") -> pd.DataFrame:

    tasks = get_all_tasks()
    if not tasks:
        return pd.DataFrame(columns=["ID", "Title", "Description", "Priority", "Deadline", "Category", "Status", "Created At"])

    df = pd.DataFrame(tasks)
    

    if status_filter != "All":
        df = df[df["status"] == status_filter]
    if category_filter != "All":
        df = df[df["category"] == category_filter]


    df = df.rename(columns={
        "id": "ID",
        "title": "Title",
        "description": "Description",
        "priority": "Priority",
        "deadline": "Deadline",
        "category": "Category",
        "status": "Status",
        "created_at": "Created At"
    })
    

    df["Description"] = df["Description"].fillna("—")
    df["Deadline"] = df["Deadline"].fillna("—")

    return df[["ID", "Status", "Priority", "Category", "Deadline", "Title", "Description"]]


def update_task_status_db(task_id: int, new_status: str) -> str:

    if not task_id:
        return "Please enter a valid Task ID."
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE tasks SET status = ? WHERE id = ?", (new_status, int(task_id)))
            conn.commit()
            if cursor.rowcount == 0:
                return f"Task ID #{task_id} not found."
        return f"Task #{task_id} marked as {new_status}."
    except Exception as e:
        return f"Error: {e}"


def delete_task_db(task_id: int) -> str:

    if not task_id:
        return "Please enter a valid Task ID."
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM tasks WHERE id = ?", (int(task_id),))
            conn.commit()
            if cursor.rowcount == 0:
                return f"Task ID #{task_id} not found."
        return f"Task #{task_id} deleted."
    except Exception as e:
        return f"Error: {e}"



def run_pipeline_ui(raw_text: str) -> Tuple[str, pd.DataFrame, List[Dict[str, Any]], str]:

    if not raw_text or not raw_text.strip():
        return (
            "⚠️ Please enter some task notes to process.",
            pd.DataFrame(columns=["Title", "Priority", "Category", "Deadline", "Status", "Description"]),
            [],
            "No input provided."
        )

    try:
        tasks = process_user_text_to_tasks(raw_text)
        if not tasks:
            return (
                "ℹ️️ No actionable tasks found in text.",
                pd.DataFrame(),
                [],
                "0 tasks extracted."
            )

        df = pd.DataFrame(tasks)
        df["description"] = df["description"].fillna("—")
        df["deadline"] = df["deadline"].fillna("—")
        df = df.rename(columns={
            "title": "Title",
            "description": "Description",
            "priority": "Priority",
            "deadline": "Deadline",
            "category": "Category",
            "status": "Status"
        })

        status_msg = f"✅ Extracted, normalized, and audited {len(tasks)} tasks successfully!"
        metrics_msg = f"Status: Ready to Save | Tasks Count: {len(tasks)}"
        return status_msg, df[["Title", "Priority", "Category", "Deadline", "Status", "Description"]], tasks, metrics_msg

    except PipelineExecutionError as pe:
        return f"❌ Pipeline Failed: {pe}", pd.DataFrame(), [], "Execution aborted."
    except Exception as e:
        return f"❌ Unexpected Error: {e}", pd.DataFrame(), [], "Unexpected fault."


def commit_tasks_to_db(staged_tasks: List[Dict[str, Any]]) -> Tuple[str, pd.DataFrame]:

    if not staged_tasks:
        return "⚠️ No tasks to save. Run the pipeline first.", fetch_tasks_dataframe()

    inserted_count = insert_tasks(staged_tasks)
    refreshed_df = fetch_tasks_dataframe()
    return f"💾 Successfully committed {inserted_count} tasks to database!", refreshed_df



custom_css = """
.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.header-box {
    text-align: center;
    margin-bottom: 25px;
    padding: 20px;
    background: linear-gradient(135deg, rgba(20, 24, 33, 0.04) 0%, rgba(20, 24, 33, 0.08) 100%);
    border-radius: 12px;
}
.header-box h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 5px;
}
.header-box p {
    font-size: 1.05rem;
    color: #64748b;
    margin: 0;
}
.action-btn-primary {
    background: #2563eb !important;
    color: white !important;
    font-weight: 600 !important;
}
.action-btn-success {
    background: #059669 !important;
    color: white !important;
    font-weight: 600 !important;
}
"""

with gr.Blocks(title="AI Smart To-Do Organizer", css=custom_css, theme=gr.themes.Soft(primary_hue="blue")) as demo:
    
    staged_tasks_state = gr.State([])

    with gr.Column(elem_classes=["header-box"]):
        gr.Markdown(
            """
            # 🎯 AI Smart To-Do Organizer
            ### High-Precision Multi-Model Extraction, Normalization & Verification Pipeline
            """
        )

    with gr.Tabs() as tabs:
        with gr.Tab("📥 Ingest & Process Tasks", id="tab_ingest"):
            with gr.Row():
                with gr.Column(scale=5):
                    gr.Markdown("#### 📝 Paste Messy Text / Voice Transcripts")
                    raw_input_box = gr.Textbox(
                        label="Unstructured Task Text",
                        placeholder="e.g., Pay electricity bill right now or power cuts in 10 minutes! Also finish history essay by Friday and buy groceries tomorrow afternoon.",
                        lines=10,
                        max_lines=15
                    )
                    
                    with gr.Row():
                        process_btn = gr.Button("⚡ Run 3-Stage Pipeline", variant="primary", scale=2)
                        clear_input_btn = gr.Button("Clear Input", scale=1)

                    sample_btn = gr.Button("Load 20-Task Benchmark Prompt", variant="secondary", size="sm")

                with gr.Column(scale=7):
                    gr.Markdown("#### 🔍 Pipeline Audit Preview")
                    pipeline_status_box = gr.Markdown("Ready for input.")
                    
                    preview_dataframe = gr.Dataframe(
                        label="Audited & Normalized Candidates",
                        headers=["Title", "Priority", "Category", "Deadline", "Status", "Description"],
                        datatype=["str", "str", "str", "str", "str", "str"],
                        interactive=False,
                        wrap=True
                    )
                    
                    save_db_btn = gr.Button("💾 Commit Tasks to Database", variant="primary", size="lg")
                    save_status_box = gr.Markdown("")


        with gr.Tab("📋 Task Dashboard", id="tab_dashboard"):
            with gr.Row():
                with gr.Column(scale=4):
                    status_filter = gr.Dropdown(
                        label="Filter by Status",
                        choices=["All", "Pending", "In Progress", "Completed"],
                        value="All"
                    )
                with gr.Column(scale=4):
                    category_filter = gr.Dropdown(
                        label="Filter by Category",
                        choices=["All", "Work", "Study", "Personal", "Shopping", "Health", "Finance", "General"],
                        value="All"
                    )
                with gr.Column(scale=4):
                    refresh_btn = gr.Button("🔄 Refresh Database", variant="secondary")

            dashboard_dataframe = gr.Dataframe(
                value=fetch_tasks_dataframe,
                label="Stored Tasks in SQLite (tasks.db)",
                headers=["ID", "Status", "Priority", "Category", "Deadline", "Title", "Description"],
                datatype=["number", "str", "str", "str", "str", "str", "str"],
                interactive=False,
                wrap=True
            )

            gr.Markdown("---")
            gr.Markdown("#### ⚙️ Task Actions")
            with gr.Row():
                task_id_input = gr.Number(label="Target Task ID", precision=0, scale=2)
                mark_complete_btn = gr.Button("✅ Mark Completed", scale=3)
                mark_pending_btn = gr.Button("⏳ Mark Pending", scale=3)
                delete_btn = gr.Button("🗑️ Delete Task", variant="stop", scale=2)

            action_result_box = gr.Markdown("")


    benchmark_text = (
        "Buy milk from the store today.\n"
        "I need to finish my history essay by Friday and also pick up my dry cleaning tomorrow.\n"
        "Submit the quarterly report tomorrow afternoon.\n"
        "Sometime soon I should probably organize my garage.\n"
        "Take the car to the mechanic for an oil change.\n"
        "Pay the electricity bill right now or they will shut off power in 10 minutes!\n"
        "I already finished filing my taxes yesterday.\n"
        "I've started working on the slide deck for the client meeting.\n"
        "Man, what a crazy weekend. The weather was amazing.\n"
        "Oh wait, remind me to cancel that subscription whenever I get a chance, but actually do it later this month maybe, well whatever.\n"
        "Call dentist today.\n"
        "Study for exam sometime next month.\n"
        "Submit assignment next Friday.\n"
        "Finish presentation for meeting.\n"
        "Finish presentation for CEO meeting tomorrow.\n"
        "Buy textbook tomorrow.\n"
        "Buy groceries tomorrow.\n"
        "Already finished an important client report.\n"
        "Started cleaning my room.\n"
        "Pay parking ticket in 15 minutes or receive a penalty."
    )
    sample_btn.click(lambda: benchmark_text, outputs=[raw_input_box])
    clear_input_btn.click(lambda: "", outputs=[raw_input_box])


    process_btn.click(
        fn=run_pipeline_ui,
        inputs=[raw_input_box],
        outputs=[pipeline_status_box, preview_dataframe, staged_tasks_state, save_status_box]
    )


    save_db_btn.click(
        fn=commit_tasks_to_db,
        inputs=[staged_tasks_state],
        outputs=[save_status_box, dashboard_dataframe]
    )


    filter_inputs = [status_filter, category_filter]
    status_filter.change(fn=fetch_tasks_dataframe, inputs=filter_inputs, outputs=[dashboard_dataframe])
    category_filter.change(fn=fetch_tasks_dataframe, inputs=filter_inputs, outputs=[dashboard_dataframe])
    refresh_btn.click(fn=fetch_tasks_dataframe, inputs=filter_inputs, outputs=[dashboard_dataframe])


    mark_complete_btn.click(
        fn=lambda tid: update_task_status_db(tid, "Completed"),
        inputs=[task_id_input],
        outputs=[action_result_box]
    ).then(
        fn=fetch_tasks_dataframe,
        inputs=filter_inputs,
        outputs=[dashboard_dataframe]
    )

    mark_pending_btn.click(
        fn=lambda tid: update_task_status_db(tid, "Pending"),
        inputs=[task_id_input],
        outputs=[action_result_box]
    ).then(
        fn=fetch_tasks_dataframe,
        inputs=filter_inputs,
        outputs=[dashboard_dataframe]
    )

    delete_btn.click(
        fn=delete_task_db,
        inputs=[task_id_input],
        outputs=[action_result_box]
    ).then(
        fn=fetch_tasks_dataframe,
        inputs=filter_inputs,
        outputs=[dashboard_dataframe]
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)