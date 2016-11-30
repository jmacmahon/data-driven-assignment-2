from .display import show_letter

class Data(object):
    def __init__(self, raw_data):
        self._raw_data = raw_data

    def get_train_letter(self, index):
        return Letter(self._raw_data['train_data'][index, :])

class Letter(object):
    def __init__(self, raw_data):
        self._raw_data = raw_data

    def show(self):
        return show_letter(self._raw_data)
