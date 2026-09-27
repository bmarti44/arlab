def validate_job(name: str, pages: int, priority: int) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("invalid job name")
    if type(pages) is not int or pages <= 0:
        raise ValueError("invalid page count")
    if type(priority) is not int:
        raise ValueError("invalid priority")


class PrintSpooler:
    """A stable priority queue with one non-preemptive printer."""

    def __init__(self):
        self._next_id = 1
        self._queued = {}
        self._active = None

    def submit(self, name: str, pages: int, priority: int = 0) -> int:
        validate_job(name, pages, priority)
        job_id = self._next_id
        record = {
            "id": job_id,
            "name": name,
            "pages": pages,
            "priority": priority,
            "printed": 0,
        }
        self._queued[job_id] = record
        self._next_id += 1
        return job_id

    def cancel(self, job_id: int) -> bool:
        if self._active is not None and self._active["id"] == job_id:
            self._active = None
            return True
        if job_id in self._queued:
            del self._queued[job_id]
            return True
        return False

    def reprioritize(self, job_id: int, priority: int) -> bool:
        validate_job("priority update", 1, priority)
        if job_id not in self._queued:
            return False
        self._queued[job_id]["priority"] = priority
        return True

    def tick(self) -> dict | None:
        if self._active is None:
            if not self._queued:
                return None
            best = min(self._queued.values(), key=lambda job: (-job["priority"], job["id"]))
            self._active = self._queued.pop(best["id"])
        job = self._active
        job["printed"] += 1
        done = job["printed"] == job["pages"]
        event = {
            "id": job["id"],
            "name": job["name"],
            "page": job["printed"],
            "total": job["pages"],
            "done": done,
        }
        if done:
            self._active = None
        return event

    def snapshot(self) -> dict:
        queued = sorted(self._queued.values(), key=lambda job: (-job["priority"], job["id"]))
        active = None if self._active is None else dict(self._active)
        return {
            "active": active,
            "queued": [dict(job) for job in queued],
        }
