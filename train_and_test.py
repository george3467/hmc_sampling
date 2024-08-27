import tensorflow_probability as tfp
import tensorflow as tf
layers = tf.keras.layers
import pandas as pd
import pickle
import numpy as np

from model import get_model, get_likelihood_fn



def get_dataset(df, categorical_features, thal_lookup):
    numeric_dataset = df.loc[:, ["age", "trestbps", "chol", "thalach", "oldpeak", "slope"]]
    numeric_dataset = tf.cast(numeric_dataset, dtype=tf.float32)

    # normalize the numeric data
    numeric_data_norm = tf.linalg.norm(numeric_dataset, axis=0) + 1e-6
    numeric_dataset /= numeric_data_norm

    categorical_dataset = df.loc[:, categorical_features]
    categorical_dataset = tf.cast(categorical_dataset, dtype=tf.int32)

    # convert "thal" from text to tokens and concat with the other categories
    thal_tokens = thal_lookup(df.loc[:, "thal"])
    thal_tokens = tf.cast(thal_tokens, dtype=tf.int32)[:, tf.newaxis]
    categorical_dataset = tf.concat([categorical_dataset, thal_tokens], axis=-1)

    labels = tf.cast(df.loc[:, "target"], dtype=tf.int32)
    return [numeric_dataset, categorical_dataset], labels



def get_data(string_vocab=None):
    file_path = ("http://storage.googleapis.com/download.tensorflow.org/data/heart.csv")
    # file_path = ("heart.csv")
    df = pd.read_csv(file_path)

    # split into train and test dataframes
    test_df = df.sample(frac=0.15, random_state=1)
    train_df = df.drop(test_df.index)

    # count the number of unique values in each category
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "ca"]
    categorical_num = {}
    for feature in categorical_features:
        vocab = sorted(list(df.loc[:, feature].unique()))
        categorical_num[feature] = len(vocab) + 1

    # For training dataset, vocab is calculated. For test dataset, vocab is provided
    if string_vocab==None:

        # category "thal" is in text format, so it requires a StringLookup
        string_vocab = sorted(list(train_df.loc[:, "thal"].unique()))
        categorical_num["thal"] = len(string_vocab) + 1

    thal_lookup = layers.StringLookup(vocabulary=string_vocab)

    train_x, train_y = get_dataset(train_df, categorical_features, thal_lookup)
    test_x, test_y = get_dataset(test_df, categorical_features, thal_lookup)

    return train_x, train_y, test_x, test_y, categorical_num, string_vocab



def run_training():
    """
    This script trains the model.
    """
    train_x, train_y, test_x, test_y, categorical_num, string_vocab = get_data()
    prior, log_prob_fn = get_model(categorical_num, train_x, train_y)

    # Training the model
    num_samples = 300
    num_burnin_steps = 50

    hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=log_prob_fn,
            step_size=2.5e-6,
            num_leapfrog_steps=5)

    hmc_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                    hmc,
                    num_adaptation_steps=int(0.8 * num_burnin_steps),
                    target_accept_prob=0.65)

    hmc_samples, stats = tfp.mcmc.sample_chain(
                    num_results=num_samples,
                    num_burnin_steps=num_burnin_steps,
                    current_state=prior.sample(),
                    kernel=hmc_kernel,
                    trace_fn=lambda _, pkr:
                        [pkr.inner_results.is_accepted, 
                         pkr.inner_results.log_accept_ratio],
                    )
    
    print(
        np.mean(stats[0]), "\n",
        tf.math.exp(stats[1]),
      )

    # saving the weight samples
    weights_dict = {"hmc_samples": hmc_samples, "string_vocab": string_vocab}
    pickle.dump(weights_dict, open("tfp_hmc_weights_1.pkl", "wb"))



def test_step(hmc_samples, test_x, test_y):

    # get the likelihood function using the test data
    test_likelihood_fn = get_likelihood_fn(test_x)

    # take the mean of the samples for each weight
    mean_samples = list(map(lambda x: tf.math.reduce_mean(x, axis=0), hmc_samples))

    # apply the trained weights to the model and sample the output
    test_dist = test_likelihood_fn(mean_samples)
    test_pred = test_dist.sample()
    
    m = tf.keras.metrics.BinaryAccuracy()
    m.update_state(test_y, test_pred)
    return m.result()

    

def run_test():
    """
    This script tests the model using the trained weights.
    """
    weights_dict = pickle.load(open("tfp_hmc_weights.pkl", "rb"))
    hmc_samples = weights_dict["hmc_samples"]
    
    train_x, train_y, test_x, test_y, _, _ = get_data(weights_dict["string_vocab"])

    test_accuracy = test_step(hmc_samples, test_x, test_y)
    return test_accuracy


