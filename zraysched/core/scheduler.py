from abc import ABC, abstractmethod
from typing import List, Dict, Union
from dataclasses import dataclass
from enum import Enum

from .app import AppEventType


class SchedulingTiming(Enum):
    EACH_WINDOW = 5
    PERIODIC = 6


class Scheduler(ABC):
    @abstractmethod
    def reacted_events_type(self) -> List[Union[AppEventType, SchedulingTiming]]:
        pass
    
    @abstractmethod
    async def run(self, jobs):
        pass


class PeriodicScheduler(Scheduler):
    def __init__(self, react_interval: int):
        super().__init__()
        self.react_interval = react_interval
    
    def reacted_events_type(self):
        return [
            SchedulingTiming.PERIODIC
        ]
    
    def get_react_interval(self):
        return self.react_interval
    