class ModelEvaluator:

    @staticmethod
    def select_best(results):

        best_name = None
        best_rmse = float("inf")

        for name, result in results.items():

            if result is None:
                continue

            rmse = result["rmse"]

            if rmse < best_rmse:
                best_rmse = rmse
                best_name = name

        return best_name