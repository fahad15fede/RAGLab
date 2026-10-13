
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalResult:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int = 5
    ) -> list[RetrievalResult]:
        pass
