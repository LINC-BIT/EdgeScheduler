from abc import ABC, abstractmethod


class Reporter(ABC):
    @abstractmethod
    def accumulate_metrics(self, job_id: str, metrics: dict):
        pass

    @abstractmethod
    def accumulate_schedule(self, time, schedule):
        pass
    
    @abstractmethod
    def report_by_text(self):
        pass

    @abstractmethod
    def report_by_plot(self, save_dir: str):
        pass
    