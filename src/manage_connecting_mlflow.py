import os
import pickle
import seaborn as sn
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import dataframe_image as dfi

from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, precision_score, recall_score

import mlflow.h2o
import mlflow
import h2o

def get_mlflow_experiment(experiment_name, client):
    try:
        exp_id = mlflow.create_experiment(experiment_name)
        #exp_id = experiment
    except:
        experiment = client.get_experiment_by_name(experiment_name)
        mlflow.set_experiment(experiment_name)
        exp_id = experiment.experiment_id

    return exp_id

def get_confusion_matrix(y_test, y_pred, title):
    confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    svm = sn.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="YlGnBu")
    figure = svm.get_figure()
    return figure


def send_to_mlflow_metric(mlflow, model, test, y_test, y_pred, MODELS_PATH):
    # run = mlflow.start_run(experiment_id=exp_id, run_name = run_name)

    mlflow.h2o.log_model(model.leader, "model")




    mlflow.log_metric("f1", f1_score(y_test, y_pred))
    mlflow.log_metric("balanced_accuracy_score", balanced_accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision_score", precision_score(y_test, y_pred))
    mlflow.log_metric("recall_score", recall_score(y_test, y_pred))
    mlflow.log_metric("auc", model.leader.auc())
    mlflow.log_metric("aucpr", model.leader.aucpr())
    mlflow.log_metric("log_loss", model.leader.logloss())
    # mlflow.log_metric("mean_per_class_error", model.leader.mean_per_class_error())

    title = "Confusion_Matrix"
    figure = get_confusion_matrix(y_test, y_pred, title)
    figure.savefig(f'{title}.png', dpi=400)
    mlflow.log_artifact(f'{title}.png')

    perf = model.leader.model_performance(test)
    pickle_performance = os.path.join(
        os.path.join(MODELS_PATH), "model_performance.pickle"
    )
    pickle.dump(
        perf,
        open(pickle_performance, "wb")
    )  # save it into a file named save.p

    mlflow.log_artifact(os.path.join(MODELS_PATH, "data_preprocessing_steps.pickle"))

    dfi.export(
        model.leaderboard.as_data_frame(),
        os.path.join(MODELS_PATH, 'model_leaderboard.png')
    )
    mlflow.log_artifact(os.path.join(MODELS_PATH, 'model_leaderboard.png'))

    lb = model.leaderboard
    m = h2o.get_model(lb[1, "model_id"])
    # m.varimp_plot()
    df_feature_importance = m.varimp(use_pandas=True).sort_values(by="percentage")
    fig = go.Figure(go.Bar(
        x=df_feature_importance.percentage * 100,
        y=df_feature_importance.variable,
        orientation='h'))
    title = 'Feature Importance'
    fig.update_layout(title=title)
    fig.write_html(os.path.join(MODELS_PATH, f"{title}.html"))
    mlflow.log_artifact(os.path.join(MODELS_PATH, f"{title}.html"))