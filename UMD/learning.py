import rl_model
from rld import RLD
from sklearn import linear_model
import data_model
import constraint
import random
import search
import time
import copy
import numpy as np
from statistics import mean
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import os

# TODO - get test set to validate model on

def generate_models(budget_limit, n_episodes, mdp, n_instances=3, testing=False):
    out = []
    seen = set()

    # Generate n_instances problem instances, with random initial maps
    # and random design budgets
    rlm = rl_model.RL_model(mdp=mdp, discount=1, n_episodes=n_episodes, allowed_rewards='none', busy_wait=True)

    rld_problem = RLD(rlm,
                      constraints=[],
                      mdp=mdp,
                      design_problem_file_name=None
    )


    if testing:
        root_node = search.DesignNode(rld_problem.initial_model, None, None,0, rld_problem)
        possible_mods = rld_problem.get_possible_modifications(root_node)
        for mod in possible_mods:
            new_model = mod.apply(rlm)
            out.append(new_model)
        return out

    out.append((rlm, rld_problem))

    for i in range(n_instances):
        good_model = False

        root_node = search.DesignNode(rld_problem.initial_model, None, None,0, rld_problem)
        possible_mods = rld_problem.get_possible_modifications(root_node)

        while not good_model:
            new_model = copy.deepcopy(rlm)
            budget = random.choice(range(1, 1+budget_limit))
            mods = random.sample(possible_mods, budget)

            for mod in mods:
                new_model = mod.apply(new_model)

            if str(new_model.mdp) not in seen:
                seen.add(str(new_model.mdp))
                good_model = True
                out.append((new_model, rld_problem))
    return out

def create_combos(x_vector):
    out = copy.deepcopy(x_vector)
    for i in range(len(x_vector)-1):
        for j in range(i+1, len(x_vector)):
            out.append(x_vector[i]*x_vector[j])
    return out

def regression_umd(feature_space, data_set, f_out, budget, num_episodes, num_repeats=4, dir_name=None, hidden_layer_size=None):

    # Use the current time as directory name to avoid accidental overwrites
    if dir_name is None:
        dir_name = time.strftime("%Y%m%d-%H%M%S")
    directory = 'data/{}/'.format(dir_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Feel free to write any pertinent information to this note_file for debugging
    note_file = open(directory+'notes_{}.txt'.format(f_out), 'w')
    note_file.write('num_episodes is {}'.format(num_episodes))

    neur_x_vectors = []
    lin_x_vectors = []
    y_vals = []

    counter = 0
    cur_min = float('inf')
    cur_max = -float('inf')
    big_model = None
    small_model = None
    baseline = None # store the value of unmodified environment

    print('calculating data')
    for instance in data_set:
        (model, umd_problem) = instance
        dmodel = data_model.DataModel(model, feature_space, umd_problem)

        neur_x_vectors.append(dmodel.get_X_vector())
        lin_x_vectors.append(dmodel.get_X_vector())

        scores = []
        dmodel = data_model.DataModel(model, feature_space, umd_problem)

        avg_score = dmodel.get_Y_val(num_repeats)
        if baseline is None:
            baseline = avg_score
        else:
            avg_score -= baseline

        # Utility_{cur_env} = Score_{cur_env}-Score_{orig_env}
        y_vals.append(avg_score)

        if avg_score > cur_max:
            cur_max = avg_score
            big_model = dmodel.model
        if avg_score < cur_min:
            cur_min = avg_score
            small_model = dmodel.model
        counter += 1
        if counter % 100 == 0:
            print('.')
    print()

    # If no hidden layer is chosen, use a single hidden layer with n/2 nodes,
    # where n is the number of features
    if hidden_layer_size is None:
        hidden_layer_size = (int(len(neur_x_vectors[0])/2), )

    orig_y_val = y_vals[0]

    # Save data for experimentation
    np.save(directory+'{}_neur_x_vectors'.format(f_out), neur_x_vectors)
    np.save(directory+'{}_y_vals'.format(f_out), y_vals)
    np.save(directory+'{}_lin_x_vectors'.format(f_out), lin_x_vectors)

    # Get training data by selecting first 80%
    # TODO: Do this with random.choice in case there is some systematic bias
    # introduced by taking first 4/5
    train_x = lin_x_vectors[0:int(len(lin_x_vectors)*0.8)]
    test_x = lin_x_vectors[len(train_x):]

    neur_train_x = neur_x_vectors[0:int(len(neur_x_vectors)*0.8)]

    # Scaling because of this note: https://scikit-learn.org/stable/modules/neural_networks_supervised.html#tips-on-practical-use
    scaler.fit(neur_train_x)
    neural_train_x = scaler.transform(neur_train_x)
    neur_test_x = neur_x_vectors[len(neur_train_x):]
    neural_test_x = scaler.transform(neur_test_x)

    train_y = y_vals[0:int(len(y_vals)*0.8)]
    test_y = y_vals[len(train_y):]

    print('fitting model')

    clf = MLPRegressor(hidden_layer_sizes=hidden_layer_size, max_iter=1000)
    clf.fit(neural_train_x, train_y)

    reg = linear_model.LinearRegression()
    reg.fit(train_x, train_y)

    print('Linear model score on training data is {}'.format(reg.score(train_x, train_y)))

    # Test model fit
    lin_pred_y = reg.predict(test_x)
    neur_pred_y = clf.predict(neural_test_x)

    neur_explain_variance = explained_variance_score(y_true=test_y, y_pred=neur_pred_y)
    lin_explain_variance = explained_variance_score(y_true=test_y, y_pred=lin_pred_y)
    print('neural explained variance score {}, linear score {}'.format(neur_explain_variance, lin_explain_variance))

    scaler.fit(neur_x_vectors)
    neur_all_x = scaler.transform(neur_x_vectors)

    lin_pred_y = reg.predict(lin_x_vectors)
    neur_pred_y = clf.predict(neur_all_x)
    best_lin = np.argmax(lin_pred_y)
    best_neur = np.argmax(neur_pred_y)

    print('linear chooses best map {} with predicted score {}'.format(data_set[best_lin][0].mdp, lin_pred_y[best_lin]))
    print('neural chooses best map {} with predicted score {}'.format(data_set[best_neur][0].mdp, neur_pred_y[best_neur]))

    note_file.close()
    return reg.coef_, clf.coefs_


def linear_regression_test():
    reg = linear_model.LinearRegression()
    reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])


class Feature():
    def get_value(self, model, umd_problem):
        print('abstract class feature: get value not implemented')
        raise NotImplementedError
