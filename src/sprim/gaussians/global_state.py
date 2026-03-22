from dataclasses import dataclass, field
from typing import List


@dataclass(kw_only=True)
class GlobalState:

    use_depth: bool = True
    selection_color: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    move_sensitivity: float = 0.02

    # TODO: add persistence


GLOBAL_STATE = GlobalState()
