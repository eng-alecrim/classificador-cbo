import mlflow
from common.utils import get_project_root


def set_local_mlflow_tracking_uri(project_name: "str") -> "str":
    project_root_dir = get_project_root(project_name=project_name)
    local_mlruns_path = project_root_dir / "mlruns"
    absolute_mlruns_path = local_mlruns_path.resolve()
    local_tracking_uri = absolute_mlruns_path.as_uri()
    mlflow.set_tracking_uri(local_tracking_uri)

    return str(local_tracking_uri)
