class HierarchicalForecast:

    def __init__(self, hierarchy):
        self.hierarchy = hierarchy
    """
    def bottom_up(self, forecasts):

        results = forecasts.copy()

        for parent, children in self.hierarchy.items():

            # suma vectorial (mes a mes)
            results[parent] = [
                sum(child_vals[i] for child_vals in [results[c] for c in children])
                for i in range(len(results[children[0]]))
            ]

        return results
    """
    def bottom_up(self, forecasts):
        results = forecasts.copy()

        for parent, children in self.hierarchy.items():

            # detectar si es multi-modelo
            if isinstance(results[children[0]], dict):

                results[parent] = {}

                models = results[children[0]].keys()

                for model in models:
                    results[parent][model] = [
                        sum(results[c][model][i] for c in children)
                        for i in range(len(results[children[0]][model]))
                    ]

            else:
                # caso normal (listas)
                results[parent] = [
                    sum(results[c][i] for c in children)
                    for i in range(len(results[children[0]]))
                ]

        return results