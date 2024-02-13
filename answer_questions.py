# Questions:

# (i) Does the distance to the nearest horn affect whether a person hears the siren or not?
# (ii) Are the people who hear the siren younger than the people who do not hear it?
# (iii) Does the direction towards the nearest horn affect whether a person hears the siren or not?

import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from evaluate_models import read_data, create_new_features


def plot_matrix(df):
    px.scatter_matrix(df).show()


def answer_distance(df):
    fig = px.histogram(
        df,
        x="distance",
        color="heard",
        histnorm="probability",
        barmode="overlay",
        title="Normalised histograms of distance for persons<br>who heard or did not hear the siren",
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="left", x=0)
    )
    fig.show()

    # => Those who heard the siren were generally closer to it, while those who did not hear it generally were further away.

    fig = px.histogram(
        df,
        x="distance_log",
        color="heard",
        histnorm="probability",
        barmode="overlay",
        title="Normalised histograms of logged distance for persons<br>who heard or did not hear the siren",
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="left", x=0)
    )
    fig.show()

    # Even more clear!

    distance_bins = pd.cut(df.distance_log, bins=np.arange(0, 13, 0.5))
    sum_per_bin = df.loc[:, "heard"].groupby(distance_bins, observed=False).sum()
    count_per_bin = df.loc[:, "heard"].groupby(distance_bins, observed=False).count()
    fraction_per_distance_bin = sum_per_bin / count_per_bin
    fraction_per_distance_bin.index.name = "Age"
    fraction_per_distance_bin.name = "Fraction heard"
    fraction_per_distance_bin.index = fraction_per_distance_bin.index.categories.mid
    count_per_bin.index = count_per_bin.index.categories.mid

    fig = make_subplots(
        rows=1,
        cols=1,
        shared_xaxes=True,
        specs=[[{"secondary_y": True}]],
        column_titles=["Fraction heard vs logged distance"],
    )
    fig1 = px.line(
        fraction_per_distance_bin,
        title="Fraction of people at various logged distances<br>that heard the siren",
        color_discrete_sequence=["black"],
        markers=True,
    )
    count_per_bin.name = "Count (secondary y-axis)"
    fig2 = px.bar(
        count_per_bin, title="Count of people at various distances that heard the siren"
    )
    fig.add_traces([fig1.data[0], fig2.data[0]], secondary_ys=[False, True])
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="left", x=0)
    )
    fig.show()
    print(
        "(i) Does the distance to the nearest horn affect whether a person hears the siren or not?"
    )
    print(
        "There is a clear trend towards lower fraction of people hearing the siren for larger distances up to around 20000 m (~e^10)"
    )
    print(
        "There is a cluster of data points above 20000 m. Considering the previous trend that cluster have a surprisinlgy high fraction of people that heard the siren."
    )
    print(
        "It is also interesting to see that several people at distances between 50 and 110 km from the siren claim to have heard it. These are likely to be errors in the data?"
    )


def answer_age(df):
    fig = px.histogram(
        df,
        x="age",
        color="heard",
        # marginal="rug",
        histnorm="probability",
        barmode="overlay",
        title="Normalised histograms of age for persons<br>who heard or did not hear the siren",
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="left", x=0)
    )
    fig.show()

    age_bins = pd.qcut(df.age, q=20)
    sum_per_bin = df.loc[:, "heard"].groupby(age_bins, observed=False).sum()
    count_per_bin = df.loc[:, "heard"].groupby(age_bins, observed=False).count()
    fraction_per_age_bin = sum_per_bin / count_per_bin
    fraction_per_age_bin.index = fraction_per_age_bin.index.categories.mid
    fraction_per_age_bin.index.name = "Age"
    fraction_per_age_bin.name = "Fraction heard"
    fig = px.bar(
        fraction_per_age_bin,
        title="Fraction of various age groups that heard the siren",
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="left", x=0)
    )
    fig.show()
    print(
        "(ii) Are the people who hear the siren younger than the people who do not hear it?"
    )
    print(
        "There is a clear tendency towards lower fraction of older people hearing the siren"
    )


def answer_direction(df):
    angle_bins = pd.cut(df.near_angle, np.arange(-180, 210, 5))
    sum_per_bin = df.loc[:, "heard"].groupby(angle_bins, observed=False).sum()
    count_per_bin = df.loc[:, "heard"].groupby(angle_bins, observed=False).count()
    fraction_per_angle_bin = sum_per_bin / count_per_bin
    fraction_per_angle_bin.index = fraction_per_angle_bin.index.categories.mid
    fraction_per_angle_bin.index.name = "Angle"
    fraction_per_angle_bin.name = "Fraction heard"
    fig = px.bar(
        fraction_per_angle_bin,
        title="Fraction of people that heard the siren,<br>grouped by their angle towards the closest siren",
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="left", x=0)
    )
    fig.show()
    print(
        "(iii) Does the direction towards the nearest horn affect whether a person hears the siren or not?"
    )
    print(
        "While it might be plausible that the direction towards the closest siren, combined with eg the wind direction, have an impact on the chance of hearing the siren, this cannot be observed in the data."
    )


def answer_questions():
    df = read_data()
    df = create_new_features(df)
    answer_distance(df)
    answer_age(df)
    answer_direction(df)


answer_questions()
