def validate_job(name: str, pages: int, priority: int) -> None:
    raise NotImplementedError()

class PrintSpooler:
    """A stable priority queue with one non-preemptive printer."""

    def __init__(self):
        raise NotImplementedError()

    def submit(self, name: str, pages: int, priority: int=0) -> int:
        raise NotImplementedError()

    def cancel(self, job_id: int) -> bool:
        raise NotImplementedError()

    def reprioritize(self, job_id: int, priority: int) -> bool:
        raise NotImplementedError()

    def tick(self) -> dict | None:
        raise NotImplementedError()

    def snapshot(self) -> dict:
        raise NotImplementedError()
