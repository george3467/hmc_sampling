
import tensorflow_probability as tfp
import tensorflow as tf
layers = tf.keras.layers

tfd = tfp.distributions
tfb = tfp.bijectors

def get_likelihood_fn(input_data):
    numeric_data, categorical_data = input_data

    def likelihood_fn(prior_weights):
        W_sex, W_cp, W_fbs, W_restecg, W_exang, W_ca, W_thal, W_1, bias_1, W_2, bias_2, W_3, bias_3, W_4, bias_4, W_5, bias_5 = prior_weights

        categorical_Ws =    [W_sex, W_cp, W_fbs, W_restecg, W_exang, W_ca, W_thal]
        W_list =            [W_1, W_2, W_3, W_4, W_5]
        bias_list =         [bias_1, bias_2, bias_3, bias_4, bias_5]
        
        # categorical W[i] shape = (sample_size, categorical_num[i], categorical_unit)
        categorical_features = []
        for i, W in enumerate(categorical_Ws):

            # gather prior values from categorical data indices
            features = tf.gather(W, categorical_data[ :, i])
            categorical_features.append(features)

        categorical_features = tf.concat(categorical_features, axis=-1)
        
        # Applies a Linear layer: y = W * x + bias and concatenates the categorical features 
        x = tf.matmul(numeric_data, W_list[0])
        x = tf.concat([x, categorical_features], axis=-1)
        x += bias_list[0]

        # Applies a sequence of Linear layers
        num_layers = len(W_list) - 1
        for i in range(num_layers):
            x = tf.matmul(x, W_list[i+1])
            x += bias_list[i+1]
        
        output = tfd.Independent(tfd.Categorical(logits=x), reinterpreted_batch_ndims=1)
        return output
    
    return likelihood_fn


def get_model(categorical_num, train_x, train_y):
    """
    The model is designed specifically for this dataset.
    """
    num_numeric = 6
    num_categories = 7
    num_labels = 2

    scale = 1
    categorical_unit = 32
    numerical_units = 128
    combined_units = numerical_units + categorical_unit * num_categories
    units = [128, 64, 32, num_labels]

    prior = tfd.JointDistributionSequentialAutoBatched([
        
        tfd.MultivariateNormalDiag(loc = tf.zeros((categorical_num["sex"], categorical_unit)),
                                    scale_diag = tf.ones((categorical_num["sex"], categorical_unit)) * scale),              # W_sex

        tfd.MultivariateNormalDiag(loc = tf.zeros((categorical_num["cp"], categorical_unit)),
                                    scale_diag = tf.ones((categorical_num["cp"], categorical_unit)) * scale),               # W_cp

        tfd.MultivariateNormalDiag(loc = tf.zeros((categorical_num["fbs"], categorical_unit)),
                                    scale_diag = tf.ones((categorical_num["fbs"], categorical_unit)) * scale),              # W_fbs

        tfd.MultivariateNormalDiag(loc = tf.zeros((categorical_num["restecg"], categorical_unit)),
                                    scale_diag = tf.ones((categorical_num["restecg"], categorical_unit)) * scale),          # W_restecg

        tfd.MultivariateNormalDiag(loc = tf.zeros((categorical_num["exang"], categorical_unit)),
                                    scale_diag = tf.ones((categorical_num["exang"], categorical_unit)) * scale),            # W_exang

        tfd.MultivariateNormalDiag(loc = tf.zeros((categorical_num["ca"], categorical_unit)),
                                    scale_diag = tf.ones((categorical_num["ca"], categorical_unit)) * scale),               # W_ca

        tfd.MultivariateNormalDiag(loc = tf.zeros((categorical_num["thal"], categorical_unit)),
                                    scale_diag = tf.ones((categorical_num["thal"], categorical_unit)) * scale),             # W_thal

        tfd.MultivariateNormalDiag(loc = tf.zeros((num_numeric, numerical_units)), 
                                   scale_diag = tf.ones((num_numeric, numerical_units)) * scale),                           # W_1
        tfd.MultivariateNormalDiag(loc = tf.zeros((1, combined_units)), scale_diag = tf.ones((1, combined_units)) * scale), # bias_1

        tfd.MultivariateNormalDiag(loc = tf.zeros((combined_units, units[0])), 
                                   scale_diag = tf.ones((combined_units, units[0])) * scale),                               # W_2
        tfd.MultivariateNormalDiag(loc = tf.zeros((1, units[0])), scale_diag = tf.ones((1, units[0])) * scale),             # bias_2

        tfd.MultivariateNormalDiag(loc = tf.zeros((units[0], units[1])), 
                                   scale_diag = tf.ones((units[0], units[1])) * scale),                                     # W_3
        tfd.MultivariateNormalDiag(loc = tf.zeros((1, units[1])), scale_diag = tf.ones((1, units[1])) * scale),             # bias_3

        tfd.MultivariateNormalDiag(loc = tf.zeros((units[1], units[2])), 
                                   scale_diag = tf.ones((units[1], units[2])) * scale),                                     # W_4
        tfd.MultivariateNormalDiag(loc = tf.zeros((1, units[2])), scale_diag = tf.ones((1, units[2])) * scale),             # bias_4

        tfd.MultivariateNormalDiag(loc = tf.zeros((units[2], units[3])), 
                                   scale_diag = tf.ones((units[2], units[3])) * scale),                                     # W_5
        tfd.MultivariateNormalDiag(loc = tf.zeros((1, units[3])), scale_diag = tf.ones((1, units[3])) * scale),             # bias_5
    ])

    # gets the likelihood function using the training data
    likelihood_fn = get_likelihood_fn(train_x)

    # this function takes prior samples as input (tuple format for tfp.mcmc)
    log_prob_fn = lambda *x: 0.1 * prior.log_prob(*x) + likelihood_fn(x).log_prob(train_y,)
    print(log_prob_fn(*prior.sample()))

    return prior, log_prob_fn








