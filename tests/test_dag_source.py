from pathlib import Path


def test_airflow_dag_uses_train_model_task_name():
    dag_source = Path("dags/improved_automl_pipeline.py").read_text(encoding="utf-8")
    assert 'task_id="train_model"' in dag_source
    assert 'task_id="train_autopilot"' not in dag_source
