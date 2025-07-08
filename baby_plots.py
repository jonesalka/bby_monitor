# %%
# -*- coding: utf-8 -*-

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import sys
from typing import Tuple
import os
from matplotlib.colors import to_hex
import plotly.graph_objects as go

DEA_farver = {
    "Bordeaux": "#441926",
    "Brændt Rød": "#ED3B2E",
    "Sort": "#000000",
    "Paper": "#f2f2eb",
    "Purple": "#65397F",
    "Dark Rose": "#CDB7B8",
    "Mint": "#3D6D79",
    "Ice": "#DCE9EA",
    "Orange": "#F05924",
    "Warm": "#D2D0BE",
    "Beige": "#DED2C3",
    "Sand": "#eeeeea",
}


def make_color_scale(ramp_colors: list, show_colormap=True):
    """
    Given a list of color-codes, returns a continous scale
    through each color, in the order provided
    """
    from colour import Color
    from matplotlib.colors import LinearSegmentedColormap

    color_ramp = LinearSegmentedColormap.from_list(
        "my_list", [Color(c1).rgb for c1 in ramp_colors]
    )
    if show_colormap:
        plt.figure(figsize=(15, 3))
        plt.imshow(
            [list(np.arange(0, len(ramp_colors), 0.1))],
            interpolation="nearest",
            origin="lower",
            cmap=color_ramp,
        )
        plt.xticks([])
        plt.yticks([])
    return color_ramp


def get_evenly_spaced_colors(cmap, n_colors):
    """
    Returns n_colors evenly spaced colors from a matplotlib colormap.
    Args:
        cmap (matplotlib.colors.Colormap): The colormap to sample from.
        n_colors (int): Number of colors to return.
    Returns:
        List of hex color strings.
    """
    return [to_hex(cmap(i / (n_colors - 1))) for i in range(n_colors)]


diverging_colormap = make_color_scale(
    [
        DEA_farver[color]
        for color in [
            "Brændt Rød",
            "Sort",
            "Mint",
            "Bordeaux",
            "Purple",
            "Orange",
            # "Paper",
        ]
    ],
    show_colormap=False,
)


def scrape_populære_babynavne() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scrapes popular baby names by year from the Danish Statistics website.
    This function uses Selenium to automate a browser and extract tables of the most popular
    boy and girl names for each available year from the website:
    https://www.dst.dk/da/Statistik/emner/borgere/navne/navne-til-nyfoedte
    The function iterates through all available years, collects the names and their statistics,
    and combines the data into a single pandas DataFrame. Each row in the resulting DataFrame
    contains the name, its statistics, the year, and the gender ("Pige" for girls, "Dreng" for boys).
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames containing the popular baby names for each year and gender,
                                           with columns for name, statistics, year ("År"), and gender ("køn").
    """
    url = "https://www.dst.dk/da/Statistik/emner/borgere/navne/navne-til-nyfoedte"

    driver = webdriver.Chrome()
    driver.maximize_window()
    driver.get(url)

    wait = WebDriverWait(driver, 10)

    all_pigenavne = []
    all_drengenavne = []
    years = []

    while True:
        # Wait for tables to load
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "row")))
        # row_div = driver.find_element(By.CLASS_NAME, "row")
        result_div = driver.find_element(By.XPATH, '//*[@id="navnepopaarresult"]')
        table_divs = result_div.find_elements(
            By.CSS_SELECTOR, "div.col-md-6.names__doublecol"
        )

        # Get year
        # The year is inside a div like: <div style="margin-top:10px;font-weight:bold" id="yearlabel">2023</div>
        year = driver.find_element(By.ID, "yearlabel").text.strip()
        years.append(year)

        # Get tables and their names
        for div in table_divs:
            caption = div.find_element(By.CLASS_NAME, "names__headerName").text.strip()
            table_html = div.find_element(By.TAG_NAME, "table").get_attribute(
                "outerHTML"
            )
            df = pd.read_html(table_html)[0]
            df["År"] = year
            if caption == "Pigenavne":
                all_pigenavne.append(df)
            elif caption == "Drengenavne":
                all_drengenavne.append(df)

        # Try to go to previous year
        try:
            select_elem = wait.until(
                EC.presence_of_element_located((By.ID, "navnepopaar"))
            )
            select = Select(select_elem)
            current_index = select.options.index(select.first_selected_option)
            if current_index + 1 >= len(select.options):
                break  # No more years
            select.select_by_index(current_index + 1)
            # Click the button to update
            button = driver.find_element(By.ID, "navnepopaarknap")
            button.click()
            time.sleep(0.5)  # Wait for page to update
        except Exception:
            break

    driver.quit()

    Pigenavne = pd.concat(all_pigenavne, ignore_index=True).assign(køn="Pige")
    Drengenavne = pd.concat(all_drengenavne, ignore_index=True).assign(køn="Dreng")

    return Pigenavne, Drengenavne


def load_navne_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the popular baby names data from CSV files.
    If the files do not exist, it scrapes the data and saves it to CSV.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames containing the popular baby names
    """

    for df in ["Pigenavne", "Drengenavne"]:
        if f"{df}.csv" not in os.listdir():
            Pigenavne, Drengenavne = scrape_populære_babynavne()
            Pigenavne.to_csv("Pigenavne.csv", index=False)
            Drengenavne.to_csv("Drengenavne.csv", index=False)
            break
        else:
            Pigenavne = pd.read_csv("Pigenavne.csv")
            Drengenavne = pd.read_csv("Drengenavne.csv")
    return Drengenavne, Pigenavne


# %%


def most_popular_by_period(
    df,
    start=1985,
    period=10,
    top_x=5,
) -> pd.DataFrame:
    """
    Returns a DataFrame of the most popular names by period.
    Parameters:
        df (pd.DataFrame): Input DataFrame containing at least the columns 'År' (year), 'Navn' (name), and 'Antal' (count).
        start (int, optional): The starting year for the analysis. Defaults to 1985.
        period (int, optional): The length of each period (in years). Defaults to 10.
        top_x (int, optional): The number of top names to return per period. Defaults to 5.
    Returns:
        pd.DataFrame: A DataFrame with columns ['Period', 'Navn', 'Antal'], where each row represents one of the top names in each period, sorted by count.
    Notes:
        - The 'Period' column is formatted as "start-end" (e.g., "1985-1994").
        - Only names from years greater than or equal to 'start' are considered.
        - The function returns the top `top_x` names per period based on the 'Antal' column.
    """

    return (
        df.assign(År=df["År"].astype(int))
        .loc[lambda d: d["År"] >= start]
        .assign(Period=lambda d: ((d["År"] - start) // period) * period + start)
        .groupby(["Period", "Navn"], as_index=False)["Antal"]
        .sum()
        .sort_values(["Period", "Antal"], ascending=[True, False])
        .groupby("Period")
        .head(3)
        .assign(
            Period=lambda d: d["Period"].astype(str)
            + "-"
            + (d["Period"] + period - 1).astype(str)
        )
        .loc[:, ["Period", "Navn", "Antal"]]
    )

    # Usage: pop_piger = top_x_names(Pigenavne, top_x=3)


# # Plot a list of colors from the diverging_colormap
# def plot_color_list(color_list, title="Farveskala"):
#     fig, ax = plt.subplots(figsize=(len(color_list), 1.5))
#     for i, color in enumerate(color_list):
#         ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
#         ax.text(
#             i + 0.5,
#             0.5,
#             color,
#             ha="center",
#             va="center",
#             fontsize=10,
#             color="white" if i != 1 else "black",
#         )
#     ax.set_xlim(0, len(color_list))
#     ax.set_ylim(0, 1)
#     ax.axis("off")
#     ax.set_title(title)
#     plt.show()


# # Example usage:
# colors = get_evenly_spaced_colors(diverging_colormap, 20)
# # plot_color_list(colors, title="Eksempel på farver til top navne")


# # Example usage:
# # colors = get_evenly_spaced_colors(diverging_colormap, top_x)
# def plot_name_trends(df, gender, highlight_names=None, ax=None):

#     global diverging_colormap
#     if highlight_names is None:
#         # Pick top 5 names by max count as default
#         highlight_names = (
#             df.groupby("Navn")["Antal"]
#             .max()
#             .sort_values(ascending=False)
#             .head(10)
#             .index.tolist()
#         )

#     for name, color in zip(
#         highlight_names,
#         get_evenly_spaced_colors(diverging_colormap, len(highlight_names)),
#     ):

#         max_year = df.query(f"Navn == '{name}'").loc[
#             lambda _df: _df["Antal"].idxmax(), "År"
#         ]

#         name_df = df.loc[
#             (df["Navn"] == name)
#             # & (df["År"].isin(range(max_year - 7, max_year + 7)))
#         ]
#         # name_df = df[df["Navn"] == name]
#         ax.fill_between(name_df["År"], name_df["Antal"], alpha=0.15, color=color)
#         ax.plot(name_df["År"], name_df["Antal"], label=name, color=color)
#         # Annotate the peak
#         peak = name_df.loc[name_df["Antal"].idxmax()]
#         ax.text(
#             peak["År"],
#             peak["Antal"],
#             name,
#             fontsize=10,
#             weight="bold",
#             color=color,
#             font="Arial",
#         )

#     ax.set_xticks(df["År"].unique())
#     ax.set_xticklabels(df["År"].unique(), rotation=45)
#     ax.set_title(gender)
#     ax.set_xlabel("År")
#     ax.set_ylabel("Antal børn")


# fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

# plot_name_trends(
#     Drengenavne,
#     "Drenge",
#     ax=axes[0],
#     highlight_names=pop_drenge["Navn"].unique(),
# )
# plot_name_trends(
#     Pigenavne,
#     "Piger",
#     ax=axes[1],
#     highlight_names=pop_piger["Navn"].unique(),
# )


# style_plot(fig, axes[0])
# style_plot(fig, axes[1])


# add_main_title_and_subtitle(
#     fig,
#     "Populære drenge- og pigenavne i Danmark",
#     "Fra 1985 til 2023",
# )


# plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.show()


def plot_name_trends_plotly(df, gender, highlight_names=None, colormap=None):
    """
    Plots name trends using Plotly for interactive visualization.
    Args:
        df (pd.DataFrame): DataFrame with columns ['Navn', 'Antal', 'År'].
        gender (str): Gender label for the plot title.
        highlight_names (list, optional): List of names to highlight. If None, top 10 by max count.
        colormap (matplotlib.colors.LinearSegmentedColormap, optional): Colormap to use for the lines.
    Returns:
        plotly.graph_objects.Figure
    """

    if highlight_names is None:
        highlight_names = (
            df.groupby("Navn")["Antal"]
            .max()
            .sort_values(ascending=False)
            .head(10)
            .index.tolist()
        )
    if colormap is not None:
        colors = [
            to_hex(colormap(i / (len(highlight_names) - 1)))
            for i in range(len(highlight_names))
        ]
    else:
        colors = [None] * len(highlight_names)

    fig = go.Figure()
    for idx, name in enumerate(highlight_names):
        name_df = df[df["Navn"] == name]
        color = colors[idx]

        # Add the line+marker trace
        fig.add_trace(
            go.Scatter(
                x=name_df["År"],
                y=name_df["Antal"],
                mode="lines+markers",
                name=name,
                line=dict(width=3, color=color),
                marker=dict(size=5, color=color),
                opacity=0.8,
                fill="tozeroy",
                visible="legendonly",  # Only show when legend or hovered
                hoverinfo="x+y+name",
            )
        )

        # Add the text annotation at the peak
        peak_idx = name_df["Antal"].idxmax()
        peak_row = name_df.loc[peak_idx]
        fig.add_trace(
            go.Scatter(
                x=[peak_row["År"]],
                y=[peak_row["Antal"] * 1.05],  # Move text 5% above the peak
                mode="text",
                text=[name],
                textposition="top center",
                showlegend=False,
                textfont=dict(family="Arial", color=color, size=13),
            )
        )

        opacity = 0.2

        # Add area under the curve
        fig.add_trace(
            go.Scatter(
                x=name_df["År"],
                y=name_df["Antal"],
                mode="lines",
                line=dict(width=0, color=color),
                fill="tozeroy",
                showlegend=False,
                fillcolor=(
                    f"rgba{(*tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)), opacity)}"
                    if color
                    else None
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text=f"Populære {gender}-navne over tid",
            font=dict(family="Arial", size=24, color="#441926", weight="bold"),
            x=0.5,
            xanchor="right",
        ),
        xaxis_title="År",
        yaxis_title="Antal børn",
        legend_title="Navn",
        xaxis=dict(tickmode="array", tickvals=sorted(df["År"].unique())),
        template="plotly_white",
        height=700,
        width=900,
    )
    return fig


if __name__ == "__main__":

    Drengenavne, Pigenavne = load_navne_data()
    period = 5
    top_x = 4
    pop_piger = most_popular_by_period(Pigenavne, top_x=top_x, period=period)
    pop_drenge = most_popular_by_period(Drengenavne, top_x=top_x, period=period)

    # Example usage
    fig = plot_name_trends_plotly(
        Drengenavne,
        "Drenge",
        highlight_names=pop_drenge["Navn"].unique(),
        colormap=diverging_colormap,
    )
    fig.show()

    fig = plot_name_trends_plotly(
        Pigenavne,
        "Piger",
        highlight_names=pop_piger["Navn"].unique(),
        colormap=diverging_colormap,
    )
    fig.show()

# %%
