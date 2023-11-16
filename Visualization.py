class Visualization:
    def __init__(self):
        self.id = IdManager.get_new_id()

    def plot_data(self, data, title=None, xlabel=None, ylabel=None):
        plt.plot(data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()