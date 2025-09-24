import ray
from typing import List
from loguru import logger
from zraysched import Scheduler, AppEventType, SchedulingTiming
    

class RECLScheduler(Scheduler):
    def reacted_events_type(self):
        return [
            AppEventType.INFERENCE_START,
            AppEventType.INFERENCE_FINISH,
            AppEventType.TRAINING_START,
            AppEventType.TRAINING_FINISH,
            SchedulingTiming.EACH_WINDOW
        ]
    
    def calculate_acc_improvement(self, accuracies: List[float]):
        if len(accuracies) <= 3:
            return 10
        else:
            last_accs = accuracies[-3: -1]
            avg_last_acc = sum(last_accs) / len(last_accs)
            return accuracies[-1] - avg_last_acc
    
    async def run(self, jobs):
        num_training_jobs = len([job_id for job_id, _ in jobs.items() if 'train' in job_id])
        num_inference_jobs = len([job_id for job_id, _ in jobs.items() if 'inference' in job_id])

        acc_improvements = {}
        res = {}
        inference_max_gpu_utilization = 0.1
        
        sum_acc_improvements = 0.
        for job_id, job in jobs.items():
            if 'train' not in job_id:
                continue
            accuracies = await job.get_metrics.remote()
            accuracies = accuracies['accuracies']
            accuracies = [v[1] for v in accuracies]
            acc_improvement = self.calculate_acc_improvement(accuracies)
            acc_improvements[job_id] = max(0.01, acc_improvement)
            sum_acc_improvements += max(0.01, acc_improvement)
        
        for job_id, job in jobs.items():
            res[job_id] = {
                'max_gpu_utilization': max(
                    (1. - inference_max_gpu_utilization * num_inference_jobs) * acc_improvements[job_id] / sum_acc_improvements - 0.01, 0.01) \
                    if 'train' in job_id else inference_max_gpu_utilization
            }
        
        return res
    