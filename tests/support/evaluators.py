from ml_toolkit.training import PredictionEvaluator


class CallbackEvaluator(PredictionEvaluator):
    def __init__(self, *, name, evaluate, write_artifacts=None):
        super().__init__(name=name)
        self.callback = evaluate
        self.writer = write_artifacts

    def evaluate(self, **inputs):
        return self.callback(**inputs)

    def write_artifacts(self, **inputs):
        return [] if self.writer is None else self.writer(**inputs)
