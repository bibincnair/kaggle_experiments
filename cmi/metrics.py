from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from clearml import Task

class Metrics:
    def __init__(self):
        self.task = Task.init(project_name="cmi", task_name="metrics")

    @staticmethod
    def cohen_kappa(y_true, y_pred):
        return cohen_kappa_score(y_true, y_pred)

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        return confusion_matrix(y_true, y_pred)