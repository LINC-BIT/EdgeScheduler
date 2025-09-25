import torch
import time
import os
import ray
import argparse
from loguru import logger

logger.remove()
logger.add(lambda msg: print(msg, end=''), level='INFO')


from zraysched import ApplicationActor, AppEvent, AppEventType, SimulatorActor, TrainingJobActor, InferenceJobActor, Reporter
from zraysched.utils.time import get_cur_time_str

from examples.two_classification_apps.app_impl import DemoApplication_ResNet18, DemoApplication_MobileNet
from examples.two_classification_apps.job_impl import DemoTrainingJob, DemoInferenceJob
from examples.two_classification_apps.metrics_reporter import DemoReporter
from schedulers.retraining import UniformScheduler, RECLScheduler, EkyaScheduler, EOMUScheduler

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=str, required=True, choices=["uniform", "recl", "ekya", "eomu"])
    args = parser.parse_args()

    if ray.is_initialized(): ray.shutdown()
    ray.init(_temp_dir="/data/zql/.ray_temp")

    apps = dict(
        resnet18=ray.remote(DemoApplication_ResNet18).remote('resnet18', DemoTrainingJob, DemoInferenceJob, device="cuda"),
        mobilenet=ray.remote(DemoApplication_MobileNet).remote('mobilenet', DemoTrainingJob, DemoInferenceJob, device="cuda")
    )
    ray.get([app.get_model_ref.remote() for app in apps.values()])
    
    apps_events=[
        AppEvent(app_id="resnet18", timestamp=10, event_type=AppEventType.INFERENCE_START),
        AppEvent(app_id="mobilenet", timestamp=20, event_type=AppEventType.INFERENCE_START),

        AppEvent(app_id="mobilenet", timestamp=50, event_type=AppEventType.TRAINING_START),
        AppEvent(app_id="resnet18", timestamp=60, event_type=AppEventType.TRAINING_START),

        AppEvent(app_id="mobilenet", timestamp=150, event_type=AppEventType.TRAINING_FINISH),
        AppEvent(app_id="resnet18", timestamp=150, event_type=AppEventType.TRAINING_FINISH),

        AppEvent(app_id="resnet18", timestamp=150, event_type=AppEventType.INFERENCE_FINISH),
        AppEvent(app_id="mobilenet", timestamp=150, event_type=AppEventType.INFERENCE_FINISH),


        AppEvent(app_id="resnet18", timestamp=160, event_type=AppEventType.INFERENCE_START),
        AppEvent(app_id="mobilenet", timestamp=170, event_type=AppEventType.INFERENCE_START),

        AppEvent(app_id="mobilenet", timestamp=180, event_type=AppEventType.TRAINING_START),
        AppEvent(app_id="resnet18", timestamp=190, event_type=AppEventType.TRAINING_START),

        AppEvent(app_id="mobilenet", timestamp=300, event_type=AppEventType.TRAINING_FINISH),
        AppEvent(app_id="resnet18", timestamp=300, event_type=AppEventType.TRAINING_FINISH),

        AppEvent(app_id="resnet18", timestamp=300, event_type=AppEventType.INFERENCE_FINISH),
        AppEvent(app_id="mobilenet", timestamp=300, event_type=AppEventType.INFERENCE_FINISH),
    ]

    # choose a scheduler
    if args.scheduler == "uniform":
        scheduler = ray.remote(UniformScheduler).remote()
    elif args.scheduler == "recl":
        scheduler = ray.remote(RECLScheduler).remote()
    elif args.scheduler == "ekya":
        scheduler = ray.remote(EkyaScheduler).remote(
            react_interval=40, 
            training_trial_duration=5, 
            candidate_hyps=[
                dict(lr=1e-3, batch_size=64),
                dict(lr=2e-3, batch_size=128)
            ]
        )
    elif args.scheduler == "eomu":
        scheduler = ray.remote(EOMUScheduler).remote(
            training_configs=[
                dict(teacher_model=0),
                dict(teacher_model=1)
            ],
            teacher_models_latency=[
                0.5,
                1.0
            ],
            low_confidence_threshold=0.2
        )

    reporter = DemoReporter()

    simulator = SimulatorActor.remote(
        apps, apps_events, scheduler, reporter, 
        res_save_dir=f"examples/two_classification_apps/results/{args.scheduler}/{get_cur_time_str()}"
    )

    await simulator.run.remote()

    ray.shutdown()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
