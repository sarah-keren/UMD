class DataModel():


    """A data point representing the model being evaluated."""

    def __init__(self, model, feature_space, umd_problem):
        self.model = model
        self.feature_space = feature_space
        self.umd_problem = umd_problem

    def __repr__(self):
        node_string = "<DataModel{}>".format(self.transition_path())
        #try:
        #    node_string += 'hval:%.2f'%self.heuristic_value
        #except TypeError:
        #    if self.heuristic_value is not None:
        #        raise TypeError
        #    else:
        #        pass
        return node_string

    def __lt__(self, node):
        return self.get_value() < node.get_value()

    def modification_set(self):
        return None

    def get_X_vector(self):
        vector = []
        for feature in self.feature_space:
            cur_val = feature.get_value(self.model, self.umd_problem)
            vector.append(cur_val)
        return vector


    def get_Y_val(self, num_repeats):
        #val = self.umd_problem.evaluate(self.model)
        #self.model.run()
        self.model.allowed_rewards = 'none'
        val = self.model.get_utility()
        return val
