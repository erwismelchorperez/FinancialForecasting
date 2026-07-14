class ModelSelector:

    def __init__(self, metric="mae"):
        """
        metric: métrica para seleccionar el mejor modelo
        opciones: 'rmse' (default), 'mae'
        """
        self.metric = metric

    def select_best(self, results):

        best_model = None
        best_score = float("inf")
        best_name = None
        if not results:
            raise ValueError("No hay resultados de modelos")

        for name, res in results.items():

            # 🔴 Saltar modelos que fallaron
            if res is None:
                continue
            #print(res)
            # 🔹 Validar métrica
            if self.metric not in res['metrics']:
                raise ValueError(f"'{self.metric}' no encontrado en resultados de {name}")

            #score = res[self.metric]
            score = res['metrics'][self.metric]
            

            print(f"Modelo: {name} | {self.metric.upper()}: {score:.4f}")

            # 🔹 Minimizar error
            if score < best_score:
                best_score = score
                best_model = res["model"]
                best_name = name

        if best_model is None:
            raise ValueError("Ningún modelo válido fue entrenado")

        #print(f"\n🏆 Mejor modelo: {best_name} con {self.metric.upper()} = {best_score:.4f}")

        #return best_model
        return {"model":best_model}