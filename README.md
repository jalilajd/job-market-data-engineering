# SmartJob Morocco

SmartJob Morocco is a small data engineering and AI project based on job offers collected from the Rekrute website.

The goal is to collect job offers, organize the data, analyze the job market and help users find relevant opportunities.

## Project features

- **Data collection:** scraping job offers from Rekrute.
- **Data processing:** cleaning and transforming data with Python and PySpark.
- **Data warehouse:** storing structured data in PostgreSQL.
- **Data visualization:** exploring job market trends with Power BI.
- **Job search:** searching offers using BM25, Dense and Hybrid search.
- **AI assistant:** using Qwen to answer questions about job offers.

## Technologies

Python, Selenium, PySpark, Apache Airflow, PostgreSQL, Power BI, Streamlit, Sentence Transformers and Qwen.

## Project structure

- `src/` : scraping and data processing scripts.
- `spark/` : Spark configuration and tests.
- `airflow/` : workflow orchestration.
- `database/` : SQL scripts for the data warehouse.
- `powerbi/` : dashboard.
- `rag/` : RAG notebook.
- `streamlit/` : web application.

## Run the application

Install the required packages:

``` bash
pip install -r requirements.txt
```

Start the application:

``` bash
streamlit run streamlit/app.py
```

Some features require the corresponding data files and local model to be available.

## Data

The project uses job offers collected from Rekrute. The collected dataset contains approximately 1,542 offers.

## Author

Data Engineering and Big Data \& AI project.
