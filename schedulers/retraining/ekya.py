import ray
from typing import List
from loguru import logger
from copy import deepcopy
import asyncio
from zraysched import PeriodicScheduler, AppEventType, SchedulingTiming
from ..retraining.uniform import UniformScheduler
    

class EkyaScheduler(PeriodicScheduler):
    def __init__(self, react_interval, training_trial_duration, candidate_hyps):
        super().__init__(react_interval)

        self.candidate_hyps = candidate_hyps
        self.training_trial_duration = training_trial_duration

        self._cur_schedule = None
        self._solving = False
        self._uniform_scheduler = UniformScheduler()
    
    async def try_a_training(self, job, hyps):
        acc_improvement, _, _ = await job.run_for.remote(None, self.training_trial_duration, 
                                                         ensure_no_side_effects=True, hyps=hyps)
        return acc_improvement
    
    async def get_uniform_schedule(self, jobs):
        res = await self._uniform_scheduler.run(jobs)
        for job_id, schedule in res.items():
            if 'train' in job_id:
                schedule['hyps'] = self.candidate_hyps[0]
        return res
    
    async def run(self, jobs):
        if not self._solving:
            asyncio.create_task(self.start_solving(jobs))

        if self._cur_schedule is None or self._cur_schedule == {}:
            res = await self.get_uniform_schedule(jobs)
        else:
            res = self._cur_schedule
        return res
    
    async def start_solving(self, jobs):
        self._solving = True

        num_training_jobs = len([job_id for job_id, _ in jobs.items() if 'train' in job_id])
        num_inference_jobs = len([job_id for job_id, _ in jobs.items() if 'inference' in job_id])
        
        best_hyps = {}
        acc_improvements = {}
        for job_id, job in jobs.items():
            if 'train' not in job_id:
                continue

            try:
                await job.get_metrics.remote()
            except ray.exceptions.RayActorError as e:
                continue

            best_acc, cur_best_hyps = 0, None
            for hyps in self.candidate_hyps:
                acc = await self.try_a_training(job, hyps)
                if acc >= best_acc:
                    best_acc = acc
                    cur_best_hyps = hyps
            best_hyps[job_id] = cur_best_hyps
            acc_improvements[job_id] = best_acc

        res = {}
        inference_max_gpu_utilization = 0.1
        
        sum_acc_improvements = 0.
        for job_id, job in jobs.items():
            if 'train' not in job_id or job_id not in acc_improvements:
                continue
            acc_improvements[job_id] = max(0.01, acc_improvements[job_id])
            sum_acc_improvements += max(0.01, acc_improvements[job_id])
        
        for job_id, job in jobs.items():
            if 'train' in job_id and job_id not in acc_improvements:
                continue

            res[job_id] = {
                'max_gpu_utilization': max(
                    (1. - inference_max_gpu_utilization * num_inference_jobs) * acc_improvements[job_id] / sum_acc_improvements - 0.01, 0.01) \
                    if 'train' in job_id else inference_max_gpu_utilization,
                'hyps': best_hyps[job_id] if 'train' in job_id else None,
            }

        self._cur_schedule = res

        self._solving = False
    