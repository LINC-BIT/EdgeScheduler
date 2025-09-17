import ray
from loguru import logger
from enum import Enum
from dataclasses import dataclass


class AppEventType(Enum):
    TRAINING_START = "training_start"
    TRAINING_FINISH = "training_finish"
    INFERENCE_START = "inference_start"
    INFERENCE_FINISH = "inference_finish"


@dataclass
class AppEvent:
    app_id: str
    timestamp: int
    event_type: AppEventType


class ApplicationActor:
    def __init__(self, app_name: str, training_job_actor_class, inference_job_actor_class, device: str = "cuda"):
        self.app_name = app_name
        self.device = device
        self.training_job_actor_class = ray.remote(num_gpus=0.01)(training_job_actor_class)
        self.inference_job_actor_class = ray.remote(num_gpus=0.01)(inference_job_actor_class)
        
        self.model_ref = ray.put(self.init_model().cpu())
        self.training_job = None
        self.inference_job = None
        self.training_job_id = None
        self.inference_job_id = None
        self.distribution_index = 0

        logger.info(f"✅ [App: {self.app_name}] Initialized.")

    def init_model(self):
        raise NotImplementedError

    def get_dataloader_func(self):
        raise NotImplementedError

    def get_model_ref(self):
        return self.model_ref

    def update_model(self, new_model):
        logger.debug(f"🔄 [App: {self.app_name}] Model updated by the training worker.")
        self.model_ref = ray.put(new_model)

    def launch_training_job(self):
        app_handle = ray.get_runtime_context().current_actor
        self.training_job_id = f'{self.app_name}-training'
        self.training_job = self.training_job_actor_class.remote(
            self.training_job_id, app_handle, self.device
        )
        logger.debug(f"▶️ [App: {self.app_name}] Training job launched.")
        return self.training_job, self.training_job_id

    def launch_inference_job(self):
        app_handle = ray.get_runtime_context().current_actor
        self.inference_job_id = f'{self.app_name}-inference'
        self.inference_job = self.inference_job_actor_class.remote(
            self.inference_job_id, app_handle, self.device
        )
        logger.debug(f"▶️ [App: {self.app_name}] Inference job launched.")
        return self.inference_job, self.inference_job_id
    
    def stop_training_job(self):
        if self.training_job:
            ray.kill(self.training_job)
            self.training_job = None
            self.distribution_index += 1
            logger.debug(f"🛑 [App: {self.app_name}] Training job stopped.")
            return self.training_job_id

    def stop_inference_job(self):
        if self.inference_job:
            ray.kill(self.inference_job)
            self.inference_job = None
            logger.debug(f"🛑 [App: {self.app_name}] Inference job stopped.")
            return self.inference_job_id
