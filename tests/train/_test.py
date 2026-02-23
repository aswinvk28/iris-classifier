import unittest

class TestTrain(unittest.TestCase):

    def test_confusion_matrix_plot(self):
        from src.train import confusion_matrix_plot
        import numpy as np
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        cm_dataframe, ax = confusion_matrix_plot(y_true, y_pred)
        self.assertEqual(cm_dataframe.shape, (3, 3))

    def test_predict(self):
        from src.train import predict
        import numpy as np
        class DummyModel:
            def predict(self, X):
                return np.array([0, 1, 2])
        dtree = DummyModel()
        X_test = np.array([[1], [2], [3]])
        y_pred = predict(dtree, X_test)
        self.assertTrue(np.array_equal(y_pred, np.array([0, 1, 2])))

if __name__ == '__main__':
    unittest.main()

