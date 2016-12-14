from logging import getLogger


class Pipeline(object):
    def __init__(self, classifier, reducers=None):
        if reducers is None:
            reducers = []
        self._reducers = reducers
        self._classifier = classifier

    def train(self, train_data, labels):
        step_data = train_data
        for r in self._reducers:
            r.train(step_data, labels)
            step_data = r.reduce(step_data)
        start_dim = train_data.shape[-1]
        end_dim = step_data.shape[-1]
        getLogger('assignment.pipeline')\
            .info("Trained pipeline reducers ({} -> {} dimensions)"
                  .format(start_dim, end_dim))
        self._classifier.train(step_data, labels)
        getLogger('assignment.pipeline')\
            .info("Trained pipeline classifier")

    def process(self, test_data):
        reduced_data = self._reduce(test_data)
        return self._classifier.classify(reduced_data)

    def _reduce(self, test_data):
        step_data = test_data
        for r in self._reducers:
            step_data = r.reduce(step_data)
        return step_data

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)
