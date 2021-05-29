def fit_predict_survival_model(self):
        #Init and fit RandomSurvivalForest with GridSearchCV
        self.rsf = RandomSurvivalForest(random_state=0) # Create the base model to tune
        self.rsf_estimators = GridSearchCV(
            estimator=self.rsf,
            param_grid=self.gridsearch_parameters,
            cv=2 ,
            verbose=2,
            n_jobs=-1,
            scoring=score_survival_model
        )

        self.rsf_estimators.fit(self.data_set["X_train"], self.data_set["Y_train"])

        #self.train_score = pd.DataFrame(
        #    index=self.X_train.index,
        #    data=pd.Series(self.rsf.predict(self.X_train)).values,
        #    columns=["score"]
        #)
        #print(self.train_score)

def get_evaluate_best_random_survival_forest_search(self):
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py
    logging.info("Tuning hyper-parameters for concordance_index_censored")
    logging.info("Best parameters set found on development set: {}".format(self.rsf_estimators.best_params_))
    means = self.rsf_estimators.cv_results_["mean_test_score"]
    stds = self.rsf_estimators.cv_results_["std_test_score"]

    logging.info("Parameters set found on development set: {}".format(self.rsf_estimators.best_params_))
    self.metrics["rsf_estimators_results"] = []
    for mean, std, params in zip(means, stds, self.rsf_estimators.cv_results_["params"]):
        logging.info("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        self.metrics["rsf_estimators_results"].append({"mean": mean, "std": std * 2, "params": params})

    self.metrics["rsf_best_params"] = self.rsf_estimators.best_params_
    self.model_parameters["rsf_best_estimator"] = self.rsf_estimators.best_estimator_

def get_scaled_probability(self):
    predictions = pd.DataFrame(
        data=self.model_parameters["rsf_best_estimator"].predict(self.data_set["X_train"]),
        columns=["predictions"]
    )
    scaler = MinMaxScaler()
    self.model_parameters["scaled_probability"] = scaler.fit(predictions)
    logging.info("Per feature minimum seen in the data: {}".format(scaler.data_min_))
    logging.info("Per feature maximum seen in the data: {}".format(scaler.data_max_))


def get_feature_importance(self):
    feature_names = list(self.data_set["X_test"].columns)
    perm = PermutationImportance(self.model_parameters["rsf_best_estimator"], n_iter=15, random_state=self.random_state)
    perm.fit(self.data_set["X_test"], self.data_set["Y_test"])
    self.metrics["permutation_importance"] = eli5.explain_weights_df(perm, feature_names=feature_names)
    self.metrics["permutation_importance"] = self.metrics["permutation_importance"].set_index("feature")
    #print(self.metrics["permutation_importance"])

    #self.metrics["permutation_importance"].sort_values(by=["weight"], ascending=False)

def score_test_survival_model(self):
    prediction = self.model_parameters["rsf_best_estimator"].predict(self.data_set["X_test"])
    result = concordance_index_censored(self.data_set["Y_test"]['sold'], self.data_set["Y_test"]['listing_days'], prediction)
    score_test_survival_model = result[0]
    score_cv_survival_model = max(self.metrics["rsf_estimators_results"], key=lambda x: x['mean'])['mean']
    self.metrics["score"] = {
        "train_date": datetime.today().strftime("%Y%m%d"),
        "score_test_survival_model": score_test_survival_model,
        "score_cv_survival_model": score_cv_survival_model
    }
    self.metrics["concordance_index_censored"] = result
    logging.info("Model Score: {}".format(self.metrics["score"]))

