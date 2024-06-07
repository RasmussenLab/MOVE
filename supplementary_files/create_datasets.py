"""
describe what is in this file
"""

# %%
import numpy as np
import random as rnd
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.datasets import make_sparse_spd_matrix
from move.data.preprocessing import scale


########################### Hyperparameters ####################################
PROJECT_NAME = "ffa_sim"
MODE = "linear"  # "non-linear"
SEED_1 = 1234
np.random.seed(SEED_1)
rnd.seed(SEED_1)

COV_ALPHA = 0.99
N_SAMPLES = 500
# SETTINGS = {
#     "proteomics": {
#         "features": 3000,
#         "frequencies": [0.002, 0.01, 0.02],
#         "coefficients": [500, 100, 50],
#         "phase": 0,
#         "offset": 500,
#     },
#     "metagenomics": {
#         "features": 1000,
#         "frequencies": [0.001, 0.05, 0.08],
#         "coefficients": [80, 20, 10],
#         "phase": np.pi / 2,
#         "offset": 400,
#     },
# }

SETTINGS = {
    "LP": {
        "features": 900,
        "frequencies": [0.002, 0.01, 0.02],
        "coefficients": [500, 100, 50],
        "phase": 0,
        "offset": 500,
    },
    "transcriptomics": {
        "features": 1000,
        "frequencies": [0.001, 0.05, 0.08],
        "coefficients": [80, 20, 10],
        "phase": np.pi / 2,
        "offset": 400,
    },
    # "PRS": {
    #     "features": 15,
    #     "frequencies": [0.1, 0.5, 0.8],
    #     "coefficients": [.2, .1, .05],
    #     "phase": np.pi / 2,
    #     "offset": 10,
    # },
    # "age": {
    # "features": 1,
    # "frequencies": [0.1, 0.5, 0.8],
    # "coefficients": [.1, .05, 0.1],
    # "phase": np.pi / 2,
    # "offset": 50,
    # },
}

COR_THRES = 0.02
PAIRS_OF_INTEREST = [(1, 2), (3, 4)]  # ,(77,75),(99,70),(38,2),(67,62)]

# Path to store output files
outpath = Path("./") / f"dataset_creation_outputs_{PROJECT_NAME}"
outpath.mkdir(exist_ok=True, parents=True)

################################ Functions ####################################


def get_feature_names(settings):
    all_feature_names = [
        f"{key}_{i+1}"
        for key in settings.keys()
        for i in range(settings[key]["features"])
    ]
    return all_feature_names


def create_mean_profiles(settings):
    feature_means = []
    for key in settings.keys():
        mean = settings[key]["offset"]
        for freq, coef in zip(
            settings[key]["frequencies"], settings[key]["coefficients"]
        ):
            mean += coef * (
                np.sin(
                    freq * np.arange(settings[key]["features"]) + settings[key]["phase"]
                )
                + 1
            )
        feature_means.extend(list(mean))
    return feature_means


def create_ground_truth_correlations_file(correlations):
    sort_ids = np.argsort(abs(correlations), axis=None)[::-1]  # 1D: N x C
    corr = np.take(correlations, sort_ids)  # 1D: N x C
    sig_ids = sort_ids[abs(corr) > COR_THRES]
    sig_ids = np.vstack(
        (sig_ids // len(all_feature_names), sig_ids % len(all_feature_names))
    ).T
    associations = pd.DataFrame(sig_ids, columns=["feature_a_id", "feature_b_id"])
    a_df = pd.DataFrame(dict(feature_a_name=all_feature_names))
    a_df.index.name = "feature_a_id"
    a_df.reset_index(inplace=True)
    b_df = pd.DataFrame(dict(feature_b_name=all_feature_names))
    b_df.index.name = "feature_b_id"
    b_df.reset_index(inplace=True)
    associations = associations.merge(a_df, on="feature_a_id", how="left").merge(
        b_df, on="feature_b_id", how="left"
    )
    associations["Correlation"] = corr[abs(corr) > COR_THRES]
    associations = associations[
        associations.feature_a_id > associations.feature_b_id
    ]  # Only one half of the matrix
    return associations


def plot_score_matrix(
    array, feature_names, cmap="bwr", vmin=None, vmax=None, label_step=50
):
    if vmin is None:
        vmin = np.min(array)
    elif vmax is None:
        vmax = np.max(array)
    # if ax is None:
    fig = plt.figure(figsize=(15, 15))
    plt.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xticks(
        np.arange(0, len(feature_names), label_step),
        feature_names[::label_step],
        fontsize=8,
        rotation=90,
    )
    plt.yticks(
        np.arange(0, len(feature_names), label_step),
        feature_names[::label_step],
        fontsize=8,
    )
    plt.tight_layout()
    # ax
    return fig


def plot_feature_profiles(dataset, feature_means):
    ## Plot profiles
    fig = plt.figure(figsize=(15, 5))
    plt.plot(
        np.arange(len(feature_means)), feature_means, lw=1, marker=".", markersize=0
    )
    for sample in dataset:
        plt.plot(
            np.arange(len(feature_means)), sample, lw=0.1, marker=".", markersize=0
        )
    plt.xlabel("Feature number")
    plt.ylabel("Count number")
    plt.title("Patient specific profiles")
    plt.tight_layout()

    return fig


def plot_feature_correlations(dataset, pairs_2_plot):
    fig = plt.figure()
    for f1, f2 in pairs_2_plot:
        plt.plot(
            dataset[:, f1],
            dataset[:, f2],
            lw=0,
            marker=".",
            markersize=1,
            label=f"{correlations[f1,f2]:.2f}",
        )
    plt.xlabel("Feature A")
    plt.ylabel("Feature B")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        fancybox=True,
        shadow=True,
        ncol=5,
    )
    plt.title("Feature correlations")
    plt.tight_layout()

    return fig


def save_splitted_datasets(
    settings: dict, PROJECT_NAME, dataset, all_feature_names, n_samples, outpath
):
    # Save index file
    index = pd.DataFrame({"ID": list(np.arange(1, n_samples + 1))})
    index.to_csv(outpath / f"random.{PROJECT_NAME}.ids.txt", index=False)
    # Save continuous files
    df = pd.DataFrame(
        dataset, columns=all_feature_names, index=list(np.arange(1, n_samples + 1))
    )
    cum_feat = 0
    for key in settings.keys():
        df_feat = settings[key]["features"]
        df_cont = df.iloc[:, cum_feat : cum_feat + df_feat]
        df_cont.insert(0, "ID", np.arange(1, n_samples + 1))
        df_cont.to_csv(
            outpath / f"random.{PROJECT_NAME}.{key}.tsv", sep="\t", index=False
        )
        cum_feat += df_feat


################################## Main script ##################################
if __name__ == "__main__":
    # %%
    # Add all datasets in a single matrix:
    all_feature_names = get_feature_names(SETTINGS)
    feat_means = create_mean_profiles(SETTINGS)

    # %%
    ###### Covariance matrix definition ######
    if MODE == "linear":
        covariance_matrix = make_sparse_spd_matrix(
            dim=len(all_feature_names),
            alpha=COV_ALPHA,
            smallest_coef=-30,
            largest_coef=30,
            norm_diag=False,
            random_state=SEED_1,
        )
    elif MODE == "non-linear":
        covariance_matrix = np.identity(len(all_feature_names))

    ABS_MAX = np.max(abs(covariance_matrix))
    fig = plot_score_matrix(
        covariance_matrix, all_feature_names, vmin=-ABS_MAX, vmax=ABS_MAX
    )
    fig.savefig(outpath / f"Covariance_matrix_{PROJECT_NAME}.png")

    #    dataset = np.array(
    #        [
    #            list(np.random.multivariate_normal(feat_means, covariance_matrix))
    #            for _ in range(N_SAMPLES)
    #        ]
    #    )

    dataset = np.random.multivariate_normal(feat_means, covariance_matrix, N_SAMPLES)
    print(dataset.shape)

    # Add non-linearities
    if MODE == "non-linear":
        for i, j in PAIRS_OF_INTEREST:
            freq = np.random.choice([4, 5, 6])
            dataset[:, i] += np.sin(freq * dataset[:, j])

    scaled_dataset, _ = scale(dataset)

    # Actual correlations:
    # correlations = np.empty(np.shape(covariance_matrix))
    # for ifeat in range(len(covariance_matrix)):
    #     for jfeat in range(len(covariance_matrix)):
    #         correlations[ifeat, jfeat] = pearsonr(dataset[:, ifeat], dataset[:, jfeat])[
    #             0
    #         ]

    correlations = np.corrcoef(dataset, rowvar=False)
    fig = plot_score_matrix(correlations, all_feature_names, vmin=-1, vmax=1)
    fig.savefig(outpath / f"Correlations_{PROJECT_NAME}.png", dpi=200)

    # Sort correlations by absolute value
    associations = create_ground_truth_correlations_file(correlations)
    associations.to_csv(outpath / f"changes.{PROJECT_NAME}.txt", sep="\t", index=False)

    # Plot feature profiles per sample
    fig = plot_feature_profiles(dataset, feat_means)
    fig.savefig(outpath / "Multi-omic_profiles.png")

    ## Plot correlations
    fig = plot_feature_correlations(dataset, PAIRS_OF_INTEREST)
    fig.savefig(outpath / "Feature_correlations.png")

    fig = plot_feature_correlations(scaled_dataset, PAIRS_OF_INTEREST)
    fig.savefig(outpath / "Feature_correlations_scaled.png")

    # Write tsv files with feature values for all samples in both datasets:
    save_splitted_datasets(
        SETTINGS, PROJECT_NAME, dataset, all_feature_names, N_SAMPLES, outpath
    )
