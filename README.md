# NYC Bus Delay Analysis and Optimization

## Project Overview

This project analyzes NYC school bus breakdown and delay records to understand the major drivers of delay duration and to build predictive models for incident resolution or delay time. The workflow includes data cleaning, exploratory data analysis, feature engineering, regression modeling, classification modeling, and operational prioritization insights for routes, vendors, boroughs, reasons and time blocks.

The main target variable is:

```text
how_long_delayed
```

This variable represents the delay duration in minutes.

## Objective

The objective of this project is to identify the factors that contribute to school bus delays and build predictive models that can estimate delay duration or classify delay severity.

Key questions addressed include:

- Which vendors, routes, boroughs, and reasons are associated with the highest delays?
- How do delays vary by time of day, weekday, and year?
- Which run types experience the most severe delays?
- Which routes cause the most severe delays for each run type?
- Which vendors have notification gaps for schools and parents?
- Can machine learning models predict delay duration?
- Can classification models categorize incidents into delay severity groups?

## Dataset

The dataset used in this project is:

```text
Bus_Breakdown_and_Delays_Data.csv
```

The dataset contains NYC school bus delay and breakdown incident records, including date/time fields, bus company information, route details, borough/service area, delay reasons, notification status and delay duration.

## Key Features

| Feature | Description |
|---|---|
| `occurred_on` | Timestamp when the incident occurred |
| `created_on` | Timestamp when the record was created |
| `informed_on` | Timestamp when stakeholders were informed |
| `run_type` | Type of bus run |
| `bus_company_name` | Bus vendor/company |
| `vendor_cleaned` | Cleaned vendor name |
| `route_number` | Bus route identifier |
| `reason` | Reason for delay or breakdown |
| `boro` / `service_area` | Borough or service area |
| `schools_serviced` | Schools serviced by the route |
| `num_schools` | Number of schools serviced |
| `has_contractor_notified_schools` | Whether schools were notified |
| `has_contractor_notified_parents` | Whether parents were notified |
| `how_long_delayed` | Delay duration in minutes |


## Modular Project Structure

```text
NYC-Bus-Delay-Analysis-and-Optimization/
│
├── data/
│   └── Bus_Breakdown_and_Delays_Data.csv
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── hyperparameter_tuning.py
│   └── optimization_insights.py
│
├── main.py
├── requirements.txt
├── README.md
├── ABOUT_GITHUB.txt
└── .gitignore
```

## Author

**Girish S Chandrappa**
