import torch
import time
import os
import ray
from torchvision.models import resnet18, vgg16
from loguru import logger

from zraysched import TrainingJobActor, InferenceJobActor
from examples.two_classification_apps.data import get_caltech256_dataloader, get_domainnet_dataloader


class DemoTrainingJob(TrainingJobActor):
    def __init__(self, job_name: str, application_handle, device):
        super().__init__(job_name, application_handle, device)

        self.hyps = {
            'batch_size': 128,
        }
        self.dataloader = ray.get(self.application_handle.get_dataloader_func.remote())('train', self.hyps['batch_size'])

        self.metrics = {
            'accuracies': [],
            'losses': []
        }
    async def run_for(self, current_time: int, duration: float):
        await super().run_for(current_time, duration)

        latest_model_ref = await self.application_handle.get_model_ref.remote()
        model = await latest_model_ref
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        avg_acc = 0.
        avg_loss = 0.
        num_iters = 0

        start_time = time.time()
        while time.time() - start_time < duration:
            optimizer.zero_grad()
            x, y = next(iter(self.dataloader))
            x, y = x.to(self.device), y.to(self.device)
            output = model(x)
            loss = torch.nn.functional.cross_entropy(output, y)
            loss.backward()
            optimizer.step()

            avg_acc += (output.argmax(dim=1) == y).float().mean().item()
            avg_loss += loss.item()
            num_iters += 1

        await self.application_handle.update_model.remote(model.cpu())

        avg_acc /= num_iters
        avg_loss /= num_iters
        self.metrics['accuracies'].append((current_time, avg_acc))
        self.metrics['losses'].append((current_time, loss.item()))
        

class DemoInferenceJob(InferenceJobActor):
    def __init__(self, job_name: str, application_handle, device):
        super().__init__(job_name, application_handle, device)

        self.hyps = {
            'batch_size': 16,
        }
        self.dataloader = ray.get(self.application_handle.get_dataloader_func.remote())('val', self.hyps['batch_size'])

        self.metrics = {
            'accuracies': []
        }
        
    async def run_for(self, current_time: int, duration: float):
        await super().run_for(current_time, duration)

        latest_model_ref = await self.application_handle.get_model_ref.remote()
        model = await latest_model_ref
        model = model.to(self.device)

        avg_acc = 0.
        num_iters = 0

        start_time = time.time()
        while time.time() - start_time < duration:
            with torch.no_grad():
                x, y = next(iter(self.dataloader))
                x, y = x.to(self.device), y.to(self.device)
                output = model(x)
                avg_acc += (output.argmax(dim=1) == y).float().mean().item()
                num_iters += 1
        
        avg_acc /= num_iters
        self.metrics['accuracies'].append((current_time, avg_acc))
