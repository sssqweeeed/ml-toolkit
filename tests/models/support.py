"""Small native-model training settings and a serializable custom objective."""

PARAMS = dict(
    iterations=5, depth=2, learning_rate=0.1, thread_count=1,
    verbose=False, allow_writing_files=False,
)


class SquaredObjective:
    def calc_ders_range(self, approxes, targets, weights):
        return [
            (
                (target - approx) * (1.0 if weights is None else weights[i]),
                -(1.0 if weights is None else weights[i]),
            )
            for i, (approx, target) in enumerate(zip(approxes, targets))
        ]
