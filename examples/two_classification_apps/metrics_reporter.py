import os
from loguru import logger
import json
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from zraysched import Reporter


class DemoReporter(Reporter):
    def __init__(self):
        self.jobs_metrics = {}
        self.jobs_metrics_by_time = {}
        self.schedules = []

    def accumulate_metrics(self, job_id: str, metrics: dict):
        for k, v in metrics.items():
            if job_id not in self.jobs_metrics:
                self.jobs_metrics[job_id] = {}
                self.jobs_metrics_by_time[job_id] = {}
            if k not in self.jobs_metrics[job_id]:
                self.jobs_metrics[job_id][k] = []
                self.jobs_metrics_by_time[job_id][k] = {}

            for k1, v1 in v:
                self.jobs_metrics_by_time[job_id][k][k1] = v1

            self.jobs_metrics[job_id][k] = [(k1, v1) for k1, v1 in self.jobs_metrics_by_time[job_id][k].items()]

    def accumulate_schedule(self, time, schedule):
        self.schedules += [{'schedule': schedule, 'time': time}]

    def report_by_text(self):
        jobs_avg_metrics = {}

        for job_id, metrics in self.jobs_metrics.items():
            jobs_avg_metrics[job_id] = {k: f'{v[-1][1]:.4f}' if len(v) > 0 else '(None)' for k, v in metrics.items()}
            logger.info(f"📊 [Report] Job {job_id} Metrics: {jobs_avg_metrics[job_id]}")

    def report_by_plot(self, save_dir):
        
        os.makedirs(save_dir, exist_ok=True)

        plt.figure()
        for job_id, metrics in self.jobs_metrics.items():
            metric_values = metrics['accuracies']
            times = [t for t, _ in metric_values]
            values = [v for _, v in metric_values]
            plt.plot(times, values, label=job_id)

        plt.xlabel('Time')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'accuracy.png'))
        plt.clf()
        plt.close()

        with open(os.path.join(save_dir, f'accuracy.png.json'), 'w') as f:
            f.write(json.dumps(self.jobs_metrics, indent=2))

        self.draw_schedules(os.path.join(save_dir, f'schedules.png'))
        
    def draw_schedules(self, fig_save_path, gpu_key: str = "max_gpu_utilization"):
        """
        根据输入的JSON数据绘制GPU分配的堆叠面积图。

        Args:
            json_data_string (str): 一个包含调度信息的JSON格式字符串。
                示例: '[{"time": 0, "schedule": {"Job A": {"max_gpu_utilization": 1.5}}}, ...]'
            gpu_key (str): 在每个任务的字典中，代表GPU分配数量的键。
                        默认为 "max_gpu_utilization"。
        """
        data = self.schedules

        with open(fig_save_path + '.json', 'w') as f:
            f.write(json.dumps(data, indent=2))

        if not isinstance(data, list) or not data:
            print("警告：JSON数据为空或格式不正确，无法绘制图表。")
            return

        # 提取所有唯一的时间点并排序
        times = sorted(list({item['time'] for item in data if 'time' in item}))
        if not times:
            print("警告：数据中未找到有效的时间点。")
            return

        # 提取所有唯一的任务名称并排序，以保证堆叠顺序一致
        all_jobs = sorted(list(set(
            job_name
            for item in data if 'schedule' in item
            for job_name in item['schedule']
        )))
        
        if not all_jobs:
            print("警告：数据中未找到任何任务。")
            return

        # 创建一个从时间点到调度详情的映射，方便快速查找
        schedule_map = {item['time']: item.get('schedule', {}) for item in data}

        # 构建一个 numpy 数组来存储每个任务在每个时间点的GPU分配情况
        # 形状为 (任务数量, 时间点数量)
        allocations = np.zeros((len(all_jobs), len(times)))

        for i, job in enumerate(all_jobs):
            for j, time in enumerate(times):
                # 获取当前时间的调度，如果任务存在，则记录其GPU分配数量
                current_schedule = schedule_map.get(time, {})
                gpu_val = current_schedule.get(job, {}).get(gpu_key, 0)
                allocations[i, j] = gpu_val
                
        # --- 开始绘图 ---
        fig, ax = plt.subplots(figsize=(17, 6))

        # 定义与示例相似的颜色方案
        # 顺序: 蓝色, 橙色, 紫色, 浅黄色
        colors = ['#6699EE', '#FF7F50', '#9370DB', '#FFDEAD']
        
        # 如果任务数多于颜色数，则循环使用颜色
        final_colors = [colors[i % len(colors)] for i in range(len(all_jobs))]

        # 为了让最后一段分配情况能够完整显示，在末尾增加一个额外的时间点
        plot_times = list(times)
        if len(plot_times) > 1:
            # 增加一段持续时间，大约是总时长的10%
            extended_time = plot_times[-1] + (plot_times[-1] - plot_times[0]) * 0.1
            plot_times.append(extended_time)
            # 扩展分配矩阵，复制最后一列的数据
            allocations = np.hstack([allocations, allocations[:, -1:]])
        elif len(plot_times) == 1:
            # 如果只有一个时间点，则默认延长50个时间单位
            plot_times.append(plot_times[0] + 50)
            allocations = np.hstack([allocations, allocations[:, -1:]])


        # 使用 fill_between 逐层绘制阶梯状的堆叠面积图
        y_bottom = np.zeros(len(plot_times))
        for i, job in enumerate(all_jobs):
            y_top = y_bottom + allocations[i]
            ax.fill_between(plot_times, y_bottom, y_top, step='post',
                            label=job, color=final_colors[i], linewidth=0)
            y_bottom = y_top

        # --- 美化图表 ---
        ax.set_xlabel("Time (s)", fontsize=18)
        ax.set_ylabel("Resources Allocated", fontsize=18)
        
        # 设置Y轴的范围和刻度
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=4))

        # 设置X轴的范围
        ax.set_xlim(left=0, right=plot_times[-1] if plot_times else 1)

        # 设置刻度字体大小
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # 设置图例，并将其放置在图表上方，类似示例
        ax.legend(loc='center right', bbox_to_anchor=(1.3, 0.5),
                ncol=1, fancybox=True, frameon=False, fontsize=16)

        # 调整布局以确保图例不会被裁剪
        plt.tight_layout()
        plt.savefig(fig_save_path, dpi=300)
        plt.clf()