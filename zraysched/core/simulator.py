import ray
import time
import os
import asyncio
from collections import defaultdict
from typing import List, Dict
from loguru import logger
from copy import deepcopy
import json

from .app import AppEvent, ApplicationActor, AppEventType
from .reporter import Reporter
from .scheduler import Scheduler, SchedulingTiming


@ray.remote
class SimulatorActor:
    def __init__(self, apps: Dict[str, ApplicationActor], apps_events: List[AppEvent], scheduler: Scheduler, reporter: Reporter, res_save_dir: str, window_size: int = 10):
        self.apps_events = apps_events
        self.window_size = window_size
        self.reporter = reporter
        self.scheduler = scheduler
        self.res_save_dir = res_save_dir
        self.current_time = 0
        self.apps = apps
        self.running_jobs = {}
        self.is_running = False
        self.schedule = None
        self.cur_window_scheduled = False

        os.makedirs(self.res_save_dir, exist_ok=True)

        logger.info("✅ [Simulator] Initialized.")

    async def simulate_event(self, event: AppEvent):
        app = self.apps.get(event.app_id, None)
        if not app:
            logger.warning(f"⚠️ [Simulator] Event for unknown app '{event.app_id}' at T={event.timestamp}s.")
            return
        
        if event.event_type == AppEventType.TRAINING_START:
            new_job, new_job_id = await app.launch_training_job.remote()
            self.running_jobs[new_job_id] = new_job
        elif event.event_type == AppEventType.TRAINING_FINISH:
            job_id = await app.stop_training_job.remote()
            if job_id in self.running_jobs:
                self.running_jobs.pop(job_id)
        elif event.event_type == AppEventType.INFERENCE_START:
            new_job, new_job_id = await app.launch_inference_job.remote()
            self.running_jobs[new_job_id] = new_job
        elif event.event_type == AppEventType.INFERENCE_FINISH:
            job_id = await app.stop_inference_job.remote()
            if job_id in self.running_jobs:
                self.running_jobs.pop(job_id)
        else:
            logger.warning(f"⚠️ [Simulator] Unknown event type '{event.event_type}' for app '{event.app_id}'.")
            
    async def reschedule(self):
        new_schedule = self.scheduler.run(self.running_jobs)
        logger.debug(f"🔄 [Simulator] Schedule updated: {new_schedule}")
        self.schedule = new_schedule

    async def run(self):
        if self.is_running:
            return
        self.is_running = True
        
        logger.info("🚀 [Simulator] Starting control loop...")

        while True:
            tick_start_time = time.time()
            self.cur_window_scheduled = False
            
            cur_app_events = self._check_app_event_at_current_time()
            if len(cur_app_events) > 0:
                for cur_app_event in cur_app_events:
                    await self.simulate_event(cur_app_event)

                if any(cur_app_event.event_type in self.scheduler.reacted_events_type() for cur_app_event in cur_app_events):
                    await self.reschedule()
                    self.cur_window_scheduled = True
            
            if SchedulingTiming.EACH_WINDOW in self.scheduler.reacted_events_type() and not self.cur_window_scheduled:
                await self.reschedule()
                self.cur_window_scheduled = True

            self._log_simulation_status()

            if len(self.running_jobs) > 0:
                for job_id, job in self.running_jobs.items():
                    duration = self.schedule[job_id]['max_gpu_utilization'] * self.window_size
                    logger.info(f"⏱️ [Simulator] Running job {job_id} for {duration:.2f}s.")
                    if duration > 0:
                        await job.run_for.remote(self.current_time, duration)

                    self.reporter.accumulate_metrics(job_id, await job.get_metrics.remote())

            self._report_jobs_metrics()
            
            elapsed = time.time() - tick_start_time
            sleep_time = self.window_size - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

            self.current_time += self.window_size

            if self._need_stop():
                break

    def _check_app_event_at_current_time(self):
        config_specificed_events = [event for event in self.apps_events if event.timestamp == self.current_time]
        return config_specificed_events
    
    def _log_simulation_status(self):
        logger.info(f"[Simulator] -------- Current Time: {self.current_time}s --------")
        logger.info(f"[Simulator] Running Jobs: {list(self.running_jobs.keys())}")
        if self.schedule:
            logger.info(f"[Simulator] Current Schedule{' (Updated)' if self.cur_window_scheduled else ''}: {self.schedule}")
        else:
            logger.info("[Simulator] No schedule set yet.")

        self.reporter.accumulate_schedule(self.current_time, self.schedule)

    def _report_jobs_metrics(self):
        self.reporter.report_by_text()
        self.reporter.report_by_plot(os.path.join(self.res_save_dir, 'plots'))

    def _need_stop(self):
        if len(self.running_jobs) == 0 and self.current_time > max(event.timestamp for event in self.apps_events):
            return True
        return False
    