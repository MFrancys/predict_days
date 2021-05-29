import plotly.express as px

def get_amount_censored_data(df):
    n_censored = len(df["sold"][df["sold"]==False])
    amount_censored_data = n_censored / len(df["sold"]) * 100
    print("%.1f%% of records are censored" % (n_censored / len(df["sold"]) * 100))
    return amount_censored_data

def graph_survival_censoring_times(df):
    fig = px.bar(df, x="listing_days", color='sold')
    return fig
