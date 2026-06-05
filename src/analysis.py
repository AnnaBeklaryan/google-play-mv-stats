import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas.plotting import scatter_matrix
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "googleplaystore.csv"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def clean_size(value):
    value = str(value)
    if "M" in value:
        return float(value.replace("M", ""))
    if "k" in value:
        return float(value.replace("k", "")) / 1024
    return np.nan


def load_and_clean_data():
    df = pd.read_csv(DATA_PATH)

    df = df[df["Rating"] <= 5].copy()

    df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")

    df["Installs"] = (
        df["Installs"]
        .astype(str)
        .str.replace("+", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df["Installs"] = pd.to_numeric(df["Installs"], errors="coerce")

    df["Price"] = df["Price"].astype(str).str.replace("$", "", regex=False)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    df["Size"] = df["Size"].apply(clean_size)

    data = df[["Rating", "Reviews", "Size", "Installs", "Price"]].dropna()

    data["log_Reviews"] = np.log1p(data["Reviews"])
    data["log_Installs"] = np.log1p(data["Installs"])
    data["log_Price"] = np.log1p(data["Price"])

    return data


def create_scatter_matrix(data):
    plot_data = data[
        ["Rating", "log_Reviews", "Size", "log_Installs"]
    ].sample(1000, random_state=42)

    scatter_matrix(
        plot_data,
        figsize=(10, 10),
        diagonal="kde",
        alpha=0.55,
        s=8,
        color="navy",
    )

    plt.suptitle("Scatter Plot Matrix of Google Play Store Variables", fontsize=16)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "scatter_matrix_googleplay.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / "scatter_matrix_googleplay.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_regression(data):
    y = data["Rating"]
    x = data[["log_Reviews", "Size", "log_Installs", "log_Price"]]
    x = sm.add_constant(x)

    model = sm.OLS(y, x).fit()
    print(model.summary())
    return model


def main():
    data = load_and_clean_data()
    print("Cleaned data shape:", data.shape)

    create_scatter_matrix(data)
    model = run_regression(data)


if __name__ == "__main__":
    main()