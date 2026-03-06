# CALISTO — Copyright (C) 2026 Alptuğ Ulugöl and Stefanie Pritzl
# Licensed under the GNU General Public License v3.0 — see LICENSE
"""
Threading infrastructure for running computations off the GUI thread.

This module provides Worker classes and utilities to execute long-running
computations asynchronously, preventing the GUI from freezing.
"""

from PySide6.QtCore import QObject, QThread, Signal, Slot
import traceback
from typing import Callable, Any, Optional


class Worker(QObject):
    """
    Worker object that runs a function in a separate thread.
    
    Signals:
        finished: Emitted when the task completes successfully
        error: Emitted when an exception occurs (str: error message, str: traceback)
        result: Emitted with the result of the computation
        progress: Emitted with progress updates (int: percentage, str: message)
    """
    finished = Signal()
    error = Signal(str, str)
    result = Signal(object)
    progress = Signal(int, str)
    
    def __init__(self, func: Callable, *args, **kwargs):
        """
        Initialize the worker.
        
        Args:
            func: The function to execute in the background
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func
        """
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._is_running = False
        
    @Slot()
    def run(self):
        """
        Execute the function and emit appropriate signals.
        This method is called when the thread starts.
        """
        self._is_running = True
        try:
            result = self.func(*self.args, **self.kwargs)
            self.result.emit(result)
        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(str(e), tb)
        finally:
            self._is_running = False
            self.finished.emit()
    
    def is_running(self) -> bool:
        """Check if the worker is currently running."""
        return self._is_running


class ProgressWorker(Worker):
    """
    Worker with built-in progress tracking.
    
    The function should accept a 'progress_callback' keyword argument
    that can be called with (percentage, message) to report progress.
    """
    
    def __init__(self, func: Callable, *args, **kwargs):
        super().__init__(func, *args, **kwargs)
        # Inject progress callback into kwargs
        self.kwargs['progress_callback'] = self._emit_progress
    
    def _emit_progress(self, percentage: int, message: str = ""):
        """Internal method to emit progress updates."""
        self.progress.emit(percentage, message)


class WorkerManager:
    """
    Manages worker threads and provides convenience methods.
    
    This class helps manage the lifecycle of worker threads and
    provides a simple interface for running async tasks.
    """
    
    def __init__(self, parent_widget):
        """
        Initialize the worker manager.
        
        Args:
            parent_widget: The parent QWidget (usually the GUI window)
        """
        self.parent_widget = parent_widget
        self.active_threads = []
        self.active_workers = []
    
    def run_async(
        self,
        func: Callable,
        *args,
        on_result: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[str, str], None]] = None,
        on_finished: Optional[Callable[[], None]] = None,
        on_progress: Optional[Callable[[int, str], None]] = None,
        use_progress: bool = False,
        **kwargs
    ) -> tuple[QThread, Worker]:
        """
        Run a function asynchronously in a separate thread.
        
        Args:
            func: The function to run
            *args: Positional arguments for func
            on_result: Callback when result is ready (receives result)
            on_error: Callback on error (receives error_msg, traceback)
            on_finished: Callback when task finishes (success or error)
            on_progress: Callback for progress updates (receives percentage, message)
            use_progress: Whether to use ProgressWorker (func must accept progress_callback)
            **kwargs: Keyword arguments for func
            
        Returns:
            tuple: (thread, worker) - can be used to track or cancel the operation
            
        Example:
            def long_computation(data, progress_callback=None):
                # ... do work ...
                if progress_callback:
                    progress_callback(50, "Halfway done")
                # ... more work ...
                return result
            
            manager.run_async(
                long_computation,
                my_data,
                on_result=self.handle_result,
                on_error=self.handle_error,
                use_progress=True,
                on_progress=self.update_progress_bar
            )
        """
        # Create worker and thread
        WorkerClass = ProgressWorker if use_progress else Worker
        worker = WorkerClass(func, *args, **kwargs)
        thread = QThread(self.parent_widget)  # Set parent to keep thread alive
        
        # Move worker to thread
        worker.moveToThread(thread)
        
        # Connect signals
        if on_result:
            worker.result.connect(on_result)
        if on_error:
            worker.error.connect(on_error)
        if on_finished:
            worker.finished.connect(on_finished)
        if on_progress and use_progress:
            worker.progress.connect(on_progress)
        
        # Connect thread start to worker run
        thread.started.connect(worker.run)
        
        # Clean up thread when done
        # Important: quit the thread when worker finishes
        worker.finished.connect(thread.quit)
        
        # Only delete objects when thread has fully finished
        # This prevents "Destroyed while thread is still running" error
        thread.finished.connect(lambda: self._cleanup_thread(thread, worker))
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(worker.deleteLater)
        
        # Track active threads/workers (keeps them alive)
        self.active_threads.append(thread)
        self.active_workers.append(worker)
        
        # Start the thread
        thread.start()
        
        return thread, worker
    
    def _cleanup_thread(self, thread: QThread, worker: Worker):
        """Remove finished threads and workers from tracking."""
        if thread in self.active_threads:
            self.active_threads.remove(thread)
        if worker in self.active_workers:
            self.active_workers.remove(worker)
    
    def wait_all(self, timeout_ms: int = -1):
        """
        Wait for all active threads to finish.
        
        Args:
            timeout_ms: Timeout in milliseconds (-1 = wait forever)
        """
        for thread in self.active_threads[:]:  # Copy list to avoid modification during iteration
            thread.quit()
            thread.wait(timeout_ms)
    
    def has_active_workers(self) -> bool:
        """Check if there are any active workers."""
        return len(self.active_workers) > 0
    
    def cleanup(self):
        """Clean up all threads (call this when closing the window)."""
        self.wait_all(timeout_ms=3000)


def run_in_thread(
    parent_widget,
    func: Callable,
    *args,
    on_result: Optional[Callable] = None,
    on_error: Optional[Callable] = None,
    on_finished: Optional[Callable] = None,
    **kwargs
) -> tuple[QThread, Worker]:
    """
    Convenience function to run a function in a separate thread.
    
    This is a simpler alternative to using WorkerManager for one-off tasks.
    
    Args:
        parent_widget: Parent QWidget
        func: Function to execute
        *args: Positional arguments for func
        on_result: Callback when result is ready
        on_error: Callback on error
        on_finished: Callback when finished
        **kwargs: Keyword arguments for func
        
    Returns:
        tuple: (thread, worker)
    """
    manager = WorkerManager(parent_widget)
    return manager.run_async(
        func, *args,
        on_result=on_result,
        on_error=on_error,
        on_finished=on_finished,
        **kwargs
    )
