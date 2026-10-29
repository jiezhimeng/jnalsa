"""Current execution status shared with the GUI, independent of log queues."""
from contextlib import contextmanager
from datetime import datetime


_sink = None


def set_task_status_sink(sink):
    global _sink
    _sink = sink


def report_task(command=None):
    if _sink is not None:
        # One assignment keeps the command and start time in the same snapshot.
        _sink['current'] = (command, datetime.now().replace(microsecond=0)) if command else None


@contextmanager
def running_task(command):
    report_task(command)
    try:
        yield
    finally:
        report_task()
