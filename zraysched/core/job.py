import ray
from loguru import logger
from abc import ABC, abstractmethod


class JobActor(ABC):
    def __init__(self, job_name: str, application_handle, device):
        self.job_name = job_name
        self.application_handle = application_handle
        self.device = device
        self.metrics = {}

    def get_metrics(self):
        return self.metrics

    @abstractmethod
    async def run_for(self, current_time: int, duration: float, ensure_no_side_effects=False, hyps=None):
        pass


class TrainingJobActor(JobActor):
    pass


class InferenceJobActor(JobActor):
    pass
