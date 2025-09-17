# EdgeScheduler

## Running Example

运行如下命令：
```bash
python examples/two_classification_apps/main.py
```

其将按照以下事件顺序运行包括一个ResNet18应用的训练和推理作业和一个MobileNetV2应用的训练和推理作业：
```python
# examples/two_classification_apps/main.py

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
```

目前支持Uniform和RECL两种训练调度器。其它调度器逐渐添加中。
