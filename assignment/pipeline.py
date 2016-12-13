class Pipeline(object):
    def __init__(self, classifier, reducers=None):
        if reducers is None:
            reducers = []
        self._reducers = reducers
        self._classifier = classifier

    def train(self, train_data):
        step_data = train_data
        for r in self._reducers:
            r.train(step_data)
            step_data = r.reduce(step_data)
        self._classifier.train(step_data)

    def process(self, test_data):
        step_data = test_data
        for reducer in self._reducers:
            step_data = reducer(step_data)
        return self._classifier(step_data)

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)
