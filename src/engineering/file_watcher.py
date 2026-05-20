"""文件监听模块 — 基于watchdog自动触发增量索引。"""
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer


class DocChangeHandler(FileSystemEventHandler):
    """文档变更事件处理器 — 带防抖。"""

    def __init__(self, callback: Callable, debounce_seconds: float = 2.0):
        super().__init__()
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self._pending: set = set()
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def on_created(self, event: FileSystemEvent):
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_deleted(self, event: FileSystemEvent):
        if not event.is_directory:
            self._schedule(event.src_path)

    def _schedule(self, path: str):
        with self._lock:
            self._pending.add(path)
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_seconds, self._flush)
            self._timer.start()

    def _flush(self):
        with self._lock:
            paths = list(self._pending)
            self._pending.clear()
        if paths:
            print(f"[FileWatcher] 检测到 {len(paths)} 个文件变更，触发增量索引...")
            self.callback(paths)


class FileWatcher:
    """文件系统监听器 — 监控docs目录自动触发索引更新。"""

    def __init__(self, watch_dir: str, on_change_callback: Callable):
        self.watch_dir = str(watch_dir)
        self.callback = on_change_callback
        self._observer: Optional[Observer] = None

    def start(self):
        """启动文件监听（后台线程）。"""
        handler = DocChangeHandler(self.callback)
        self._observer = Observer()
        self._observer.schedule(handler, self.watch_dir, recursive=True)
        self._observer.start()
        print(f"[FileWatcher] 开始监听: {self.watch_dir}")

    def stop(self):
        """停止文件监听。"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            print("[FileWatcher] 监听已停止")

    def run_forever(self):
        """阻塞运行文件监听。"""
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
