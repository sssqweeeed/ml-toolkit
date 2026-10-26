from collections.abc import Sequence


class csr_matrix:
    def __init__(self, arg1: object) -> None: ...


def hstack(blocks: Sequence[csr_matrix], format: str = ...) -> csr_matrix: ...
