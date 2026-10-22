"""Project-defined row boundaries for validation and final fitting."""

from collections.abc import Iterator

from ml_toolkit.data import Pool
from ml_toolkit.training import Split, Splitter


class ProjectSplitter(Splitter[Pool]):
    def __init__(self, data: Pool, *, validation_start: int, test_start: int):
        super().__init__(data)
        if not 0 < validation_start < test_start < len(data):
            raise ValueError(
                "Boundaries must leave nonempty train, validation and test"
            )
        self.validation_start = validation_start
        self.test_start = test_start

    def get_validation(self) -> Iterator[Split[Pool]]:
        yield Split(
            self.data.take(range(self.validation_start)),
            self.data.take(range(self.validation_start, self.test_start)),
            index=1,
        )

    def get_final(self) -> Split[Pool]:
        return Split(
            self.data.take(range(self.test_start)),
            self.data.take(range(self.test_start, len(self.data))),
        )

    def parameters(self) -> dict[str, object]:
        return {
            "validation_start": self.validation_start,
            "test_start": self.test_start,
        }
