class ModelSelector:

    @staticmethod
    def get_best_model(
        account_results,
        metric="rmse"
    ):
        candidates = []

        for model_name, res in account_results.items():

            if res is None:
                continue

            metrics = res.get("metrics")

            if metrics is None:
                continue

            score = metrics.get(metric)

            if score is None:
                continue

            if score <= 0:
                continue

            candidates.append(
                (
                    model_name,
                    score,
                    metrics.get("mae")
                )
            )

        if len(candidates) == 0:
            return None

        # ordenar por RMSE y desempatar por MAE
        candidates = sorted(
            candidates,
            key=lambda x: (
                x[1],   # RMSE
                x[2]    # MAE
            )
        )

        return candidates[0][0]