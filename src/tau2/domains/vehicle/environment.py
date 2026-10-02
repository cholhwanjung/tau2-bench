from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.vehicle.data_model import VehicleDB
from tau2.domains.vehicle.tools import VehicleTools
from tau2.domains.vehicle.utils import (
    VEHICLE_DB_PATH,
    VEHICLE_POLICY_PATH,
    VEHICLE_TASK_SET_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


def get_environment(
    db: Optional[VehicleDB] = None,
    solo_mode: bool = False,
) -> Environment:
    if solo_mode:
        raise ValueError("Solo mode not supported for vehicle")
    if db is None:
        db = VehicleDB.load(VEHICLE_DB_PATH)
    tools = VehicleTools(db)
    with open(VEHICLE_POLICY_PATH, "r") as fp:
        policy = fp.read()
    return Environment(
        domain_name="vehicle",
        policy=policy,
        tools=tools,
        user_tools=None,
    )


def get_tasks(task_split_name: Optional[str] = None) -> list[Task]:
    tasks = load_file(VEHICLE_TASK_SET_PATH)
    tasks = [Task.model_validate(task) for task in tasks]
    if task_split_name is None:
        return tasks
    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. Valid splits are: {task_splits.keys()}"
        )
    return [task for task in tasks if task.id in task_splits[task_split_name]]


def get_tasks_split() -> dict[str, list[str]]:
    split_file = (
        Path(VEHICLE_TASK_SET_PATH).parent
        / f"split_{Path(VEHICLE_TASK_SET_PATH).stem}.json"
    )
    return load_file(split_file)
