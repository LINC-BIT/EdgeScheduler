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
            'lr': 3e-4
        }
        self.dataloader = ray.get(self.application_handle.get_dataloader_func.remote())('train', self.hyps['batch_size'])

        self.metrics = {
            'accuracies': [],
            'losses': []
        }
    async def run_for(self, current_time: int, duration: float, ensure_no_side_effects=False, hyps=None):
        await super().run_for(current_time, duration)

        latest_model_ref = await self.application_handle.get_model_ref.remote()
        model = await latest_model_ref
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.hyps['lr'] if hyps is None or 'lr' not in hyps else hyps['lr'])
        
        def get_a_batch():
            batch_size = self.hyps['batch_size'] if hyps is None or 'batch_size' not in hyps else hyps['batch_size']
            res_x, res_y = None, None
            while True:
                x, y = next(iter(self.dataloader))
                x, y = x.to(self.device), y.to(self.device)

                if res_x is None:
                    res_x = x
                    res_y = y
                else:
                    res_x = torch.cat((res_x, x), dim=0)
                    res_y = torch.cat((res_y, y), dim=0)

                if res_x.shape[0] >= batch_size:
                    res_x = res_x[: batch_size]
                    res_y = res_y[: batch_size]
                    break
            return res_x, res_y

        accs = []
        avg_loss = 0.
        num_iters = 0

        start_time = time.time()
        while time.time() - start_time < duration:
            optimizer.zero_grad()
            x, y = get_a_batch()
            output = model(x)
            loss = torch.nn.functional.cross_entropy(output, y)
            loss.backward()
            optimizer.step()

            accs += [(output.argmax(dim=1) == y).float().mean().item()]
            avg_loss += loss.item()
            num_iters += 1

        avg_acc = sum(accs) / num_iters
        avg_loss /= num_iters
        
        if not ensure_no_side_effects:
            await self.application_handle.update_model.remote(model.cpu())
            self.metrics['accuracies'].append((current_time, avg_acc))
            self.metrics['losses'].append((current_time, loss.item()))
        else:
            acc_improvement = accs[-1] - accs[0]
            avg_acc /= num_iters
            avg_loss /= num_iters
            return acc_improvement, avg_acc, avg_loss
        

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
        
    async def run_for(self, current_time: int, duration: float, ensure_no_side_effects=False, hyps=None):
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

        if not ensure_no_side_effects:
            self.metrics['accuracies'].append((current_time, avg_acc))
        else:
            return avg_acc
        