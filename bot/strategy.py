from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    name = "base"

    @abstractmethod
    def should_enter(self, df: pd.DataFrame) -> bool:
        raise NotImplementedError

    @abstractmethod
    def should_exit(self, df: pd.DataFrame) -> bool:
        raise NotImplementedError
