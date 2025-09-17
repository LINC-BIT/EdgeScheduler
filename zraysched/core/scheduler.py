from abc import ABC, abstractmethod
from typing import List, Dict, Union
from dataclasses import dataclass
from enum import Enum

from .app import AppEventType


class SchedulingTiming(Enum):
    EACH_WINDOW = 5


class Scheduler(ABC):
    @abstractmethod
    def reacted_events_type(self) -> List[Union[AppEventType, SchedulingTiming]]:
        pass
    
    @abstractmethod
    def run(self, jobs):
        pass
