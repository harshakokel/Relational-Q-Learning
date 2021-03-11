from srlearn.rdn import BoostedRDNRegressor


class RDNRegressor(BoostedRDNRegressor):

    def __init__(self,
                 background=None,
                 target="None",
                 n_estimators=10,
                 node_size=2,
                 max_tree_depth=3,
                 ):
        super().__init__(background,
                       target,
                       n_estimators,
                       node_size,
                       max_tree_depth)
        self.predict_cache = {}

    def fetch_cache(self, state):
        if state in self.predict_cache:
            return self.predict_cache[state]
        return None

    def add_cache(self, state, q_value):
        self.predict_cache[state] = q_value

    def set_cache(self, cache):
        self.predict_cache = cache


