import pandas as pd

def get_extract_features(df):
    df = df.rename(columns={'first_price': 'price'})
    df['diff_price'] = df['average_price'] - df['price']
    df['diff_price_perc'] = (df['average_price'] - df['price']) / df['average_price']

    df[['car_brand', 'car_type', 'C']] = df['car_type'].str.split('_', 2, expand=True)

    current_year = datetime.datetime.now().year
    df['antiquity'] = current_year - df.year

    df = df.drop(columns=['C', 'year'])

    return df


def change_numeric_column_types(df):
    df[['average_price', 'price']] = df[['average_price', 'price']].astype('float32')
    df[['antiquity', 'km']] = df[['antiquity', 'km']].astype('int32')
    # df_detail_cars_inventory.index = pd.to_numeric(df_detail_cars_inventory.index, errors='coerce', downcast='integer').astype('int32')
    return df


def kaplan_meier_estimator_plot(df, action, time):
    time_sold, survival_prob_sold = kaplan_meier_estimator(df[action], df[time])
    fig = go.Figure(go.Scatter(x=time_sold, y=survival_prob_sold, mode='lines', name='lines'))
    fig.update_layout(
        title='The Survival Function',
        xaxis_title='DI - (Days od Inventory)',
        yaxis_title='est. probability of survival $\hat{S}(t)$',
        xaxis=dict(showline=True, linecolor='rgb(204, 204, 204)', linewidth=2),
        yaxis=dict(showline=True, linecolor='rgb(204, 204, 204)', linewidth=2),
        plot_bgcolor='white'
    )
    fig.show()


def kaplan_meier_estimator_by_statification_plot(df, action, time, statification):
    feature_values = df[statification].unique().tolist()
    fig = go.Figure()

    for value in feature_values:
        mask = df[feature] == value
        time_sold, survival_prob_sold = kaplan_meier_estimator(df[action][mask], df[time][mask])
        fig.add_trace(go.Scatter(x=time_sold, y=survival_prob_sold, mode='lines', name=value))

    fig.update_layout(
        title=f'The Survival Function by Statification - {statification}',
        xaxis_title='DI - (Days of Inventory)',
        yaxis_title='est. probability of survival $\hat{S}(t)$',
        xaxis=dict(showline=True, linecolor='rgb(204, 204, 204)', linewidth=2),
        yaxis=dict(showline=True, linecolor='rgb(204, 204, 204)', linewidth=2),
        plot_bgcolor='white'
    )
    fig.show()


def get_normalization_parameters(traindf, column):
    def _z_score_params(column):
        mean = traindf[column].mean()
        std = traindf[column].std()
        return {'mean': mean, 'std': std}

    normalization_parameters = {}
    normalization_parameters[column] = _z_score_params(column)
    return normalization_parameters


def make_zscaler(mean, std):
    def zscaler(col):
        return (col - mean) / std

    return zscaler


def create_feature_cols(column_name, use_normalization, normalization_parameters={}):
    normalizer_fn = None
    if use_normalization:
        column_params = normalization_parameters[column_name]
        mean = column_params['mean']
        std = column_params['std']
        normalizer_fn = make_zscaler(mean, std)

    return tf.feature_column.numeric_column(column_name, normalizer_fn=normalizer_fn)


def df_to_dataset(dataframe, target, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop(target)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)

    return ds


def get_autoencoder_features(example_batch):
    ds_autoencoder = pd.DataFrame(
        data=model.predict(example_batch),
        index=[-1 if b'NAN' == i else int(i) for i in tf.get_static_value(example_batch['car_id']).tolist()])
    ds_autoencoder.index.name = 'car_id'
    ds_autoencoder.columns = [f'encoder_{column}' for column in ds_autoencoder.columns]
    return ds_autoencoder
