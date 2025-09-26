import ray
from typing import List
from loguru import logger
from zraysched import Scheduler, AppEventType, SchedulingTiming


class AdaInfScheduler(Scheduler):
    def reacted_events_type(self):
        return [
            AppEventType.INFERENCE_START,
            AppEventType.INFERENCE_FINISH,
            AppEventType.TRAINING_START,
            AppEventType.TRAINING_FINISH
        ]
    
    async def run(self, jobs):
        # TODO:
        # The core idea of AdaInf is to allocate resources according to the inference jobs' latency constraints.
        # But in current implementation, the latency constraints of all inference jobs are not considered and implemented correctly yet
        # (I just set inference_max_gpu_utilization to a constant value 0.1, which represents that all inference jobs have a consistent inference latency constraint).
        # So currently, AdaInf is the same as Uniform.
        num_training_jobs = len([job_id for job_id, _ in jobs.items() if 'train' in job_id])
        num_inference_jobs = len([job_id for job_id, _ in jobs.items() if 'inference' in job_id])

        res = {}
        inference_max_gpu_utilization = 0.1

        for job_id, job in jobs.items():
            res[job_id] = {
                'max_gpu_utilization': (1. - inference_max_gpu_utilization * num_inference_jobs) / num_training_jobs - 0.01 if 'train' in job_id else inference_max_gpu_utilization
            }
        
        return res