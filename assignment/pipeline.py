"""Code for building reducer-classifier pipelines."""

from logging import getLogger


class Pipeline(object):
    """A pipeline of selectors, reducers and a classifier.

    :param classifier: The classifier to push the reduced data to
    :param reducers: A list of reducer objects to sequentially process the data
    """

    def __init__(self, classifier, reducers=None):
        """See class docstring."""
        if reducers is None:
            reducers = []
        self._reducers = reducers
        self._classifier = classifier

    def train(self, train_data, labels):
        """Train the whole pipeline with the provided data and labels.

        Sequentially trains each reducer and then the classifier on the
        provided data and labels.

        :param train_data: The n training vectors
        :param labels: The n labels corresponding to the training vectors
        """
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
        """Reduce and classify the input vector(s)."""
        reduced_data = self._reduce(test_data)
        return self._classifier.classify(reduced_data)

    def _reduce(self, test_data):
        step_data = test_data
        for r in self._reducers:
            step_data = r.reduce(step_data)
        return step_data

    def __call__(self, *args, **kwargs):
        """Magic method for passing the pipeline as a function."""
        return self.process(*args, **kwargs)
