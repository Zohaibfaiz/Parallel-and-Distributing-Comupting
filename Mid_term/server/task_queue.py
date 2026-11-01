"""
================================================================================
CSC-334: Parallel and Distributed Computing
Lab 4: Custom Distributed Task Offloading & Remote GPU Rendering System
Module: server/task_queue.py
Description: Thread-safe Task Queue Manager for concurrent workload scheduling.
================================================================================
"""

import time
import queue
import threading
from typing import Dict, Any, Optional, Callable


class TaskStatus:
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TaskItem:
    """Represents a discrete offloaded task within the distributed worker node."""
    def __init__(self, task_id: str, task_type: str, config: Dict[str, Any], input_path: str, output_path: str):
        self.task_id = task_id
        self.task_type = task_type  # 'video_transcode' or 'cuda_compute'
        self.config = config
        self.input_path = input_path
        self.output_path = output_path
        self.status = TaskStatus.QUEUED
        self.submit_time = time.time()
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.progress_percent: float = 0.0
        self.current_fps: float = 0.0
        self.speed_multiplier: str = "1.0x"
        self.error_message: Optional[str] = None
        self.execution_duration: float = 0.0
        self.is_cancelled = False
        self.client_socket = None  # Bound for real-time progress push


class TaskQueueManager:
    """
    Central thread-safe Task Queue Manager on the remote worker node.
    Schedules and tracks execution of compute tasks offloaded by clients.
    """
    def __init__(self, max_concurrent_workers: int = 1):
        self._queue = queue.Queue()
        self._tasks: Dict[str, TaskItem] = {}
        self._lock = threading.Lock()
        self._max_workers = max_concurrent_workers
        self._workers = []
        self._shutdown_flag = threading.Event()

    def submit_task(self, task: TaskItem) -> None:
        with self._lock:
            self._tasks[task.task_id] = task
        self._queue.put(task)

    def get_task(self, timeout: float = 1.0) -> Optional[TaskItem]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def task_done(self) -> None:
        self._queue.task_done()

    def get_task_by_id(self, task_id: str) -> Optional[TaskItem]:
        with self._lock:
            return self._tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status in [TaskStatus.QUEUED, TaskStatus.RUNNING]:
                task.is_cancelled = True
                task.status = TaskStatus.CANCELLED
                return True
        return False

    def get_active_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING)

    def get_queue_size(self) -> int:
        return self._queue.qsize()

    def list_tasks(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                tid: {
                    "task_type": t.task_type,
                    "status": t.status,
                    "progress": t.progress_percent,
                    "submit_time": t.submit_time,
                    "duration": t.execution_duration
                }
                for tid, t in self._tasks.items()
            }
