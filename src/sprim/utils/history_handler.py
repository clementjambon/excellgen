from typing import Any

HISTORY_SIZE = 10


class HistoryHandler:

    def __init__(self, history_size: int = HISTORY_SIZE) -> None:
        self.history_size = history_size
        self.history = []
        self.current_index = -1

    def record_new(self, new_result) -> None:
        # TODO: invalidate everyone after this one
        self.history = self.history[: self.current_index + 1] + [new_result]
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size :]
        self.current_index = len(self.history) - 1

    def current(self) -> Any | None:
        if len(self.history) > 0 and self.current_index >= 0:
            return self.history[self.current_index]
        return None

    def next(self) -> Any | None:
        if self.current_index + 1 >= len(self.history):
            return None
        self.current_index += 1
        return self.history[self.current_index]

    def previous(self) -> Any | None:
        if self.current_index - 1 < 0:
            return None
        self.current_index -= 1
        return self.history[self.current_index]
