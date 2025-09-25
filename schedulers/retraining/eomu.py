import ray
import random
from typing import List
from loguru import logger
from zraysched import Scheduler, AppEventType, SchedulingTiming
    

class EOMUScheduler(Scheduler):
    def __init__(self, training_configs, teacher_models_latency, low_confidence_threshold):
        super().__init__()
        self.training_configs = training_configs
        self.low_confidence_threshold = low_confidence_threshold
        self.teacher_models_latency = teacher_models_latency

    def reacted_events_type(self):
        return [
            AppEventType.INFERENCE_START,
            AppEventType.INFERENCE_FINISH,
            AppEventType.TRAINING_START,
            AppEventType.TRAINING_FINISH
        ]
    
    async def find_best_config_for_a_job(self, job):
        best_config, best_utility = None, -1e8
        for config_i, config in enumerate(self.training_configs):
            t = await self.calculate_utility_of_a_config_of_a_job(job, config)
            if t > best_utility:
                best_config, best_utility = config, t
        return best_config, best_utility
    
    async def calculate_utility_of_a_config_of_a_job(self, job, config):
        metrics = await job.get_metrics.remote()

        # TODO: I don't see how to obtain the accuracy (A_k) in the paper.
        accuracy = random.random() * 0.5 + 0.5
        
        n_samples = len(metrics['confidences_of_samples'])
        if n_samples == 0:
            urgency = 0
        else:
            num_low_confidence_samples, num_samples = 0, 0
            for _, c in metrics['confidences_of_samples']:
                num_low_confidence_samples += sum([1 for c1 in c if c1 < self.low_confidence_threshold])
                num_samples += len(c)
            urgency = num_low_confidence_samples / num_samples

        latency = self.teacher_models_latency[config['teacher_model']]

        return accuracy - urgency * latency
    
    async def run(self, jobs):
        num_training_jobs = len([job_id for job_id, _ in jobs.items() if 'train' in job_id])
        num_inference_jobs = len([job_id for job_id, _ in jobs.items() if 'inference' in job_id])

        res = {}
        inference_max_gpu_utilization = 0.1

        utility_of_different_configs = {}
        sum_utility = 0
        for job_id, job in jobs.items():
            if 'train' not in job_id:
                continue
            utility_of_different_configs[job_id] = await self.find_best_config_for_a_job(job)
            sum_utility += utility_of_different_configs[job_id][1]
        
        for job_id, job in jobs.items():
            res[job_id] = {
                'max_gpu_utilization': max(
                    (1. - inference_max_gpu_utilization * num_inference_jobs) * utility_of_different_configs[job_id][1] / sum_utility - 0.01, 0.01) \
                    if 'train' in job_id else inference_max_gpu_utilization,
                'hyps': utility_of_different_configs[job_id][0] if 'train' in job_id else None
            }
        
        return res
    