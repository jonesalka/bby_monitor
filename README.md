# Baby Names Project

This project is focused on analyzing and visualizing popular baby names in Denmark. It utilizes web scraping techniques to gather data on baby names from the Danish Statistics website and presents this data through interactive visualizations.

## Project Structure

- **src/Untitled-1.py**: Contains the functions and logic for scraping and processing baby name data. This includes the `plot_name_trends_plotly` function, which generates interactive line plots for baby names over time.
  
- **plots.qmd**: A Quarto document that utilizes the functions from `Untitled-1.py` to create two line plots. It includes a Danish text description explaining that the plots show the three most popular names for newborns within five-year periods and details how the data selection is performed.

## Usage Instructions

1. Ensure you have the necessary Python packages installed. You can install them using pip:
   ```
   pip install pandas selenium plotly
   ```

2. Run the `src/Untitled-1.py` script to scrape the latest baby name data and save it as CSV files.

3. Open the `plots.qmd` file in a Quarto-compatible environment to render the plots and view the analysis.

## License

This project is licensed under the MIT License. Feel free to modify and use the code as needed.