# ✈️ Flight Delay Prediction with Apache Spark

A Big Data project for predicting flight arrival delays using Apache Spark MLlib. The project includes exploratory data analysis (EDA), feature engineering, model training with cross-validation, and a production-ready inference application.

**Authors:** Melen Laclais, Carlos Manzano Izquierdo

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Training (Notebook)](#training-notebook)
  - [Inference (Application)](#inference-application)
- [Model Details](#model-details)
- [Performance Metrics](#performance-metrics)

---

## 🎯 Project Overview

This project builds a machine learning pipeline to predict flight arrival delays (`ArrDelay`) using historical flight data from the U.S. Department of Transportation. The solution leverages Apache Spark's distributed computing capabilities to handle large-scale data processing and model training.

### Key Features

- **Distributed Processing:** Utilizes PySpark for scalable data processing
- **Feature Engineering:** Includes plane metadata enrichment, temporal features, and categorical encoding
- **Cross-Validation:** Hyperparameter tuning with k-fold cross-validation
- **Production-Ready:** Standalone inference application executable via `spark-submit`
- **Data Quality:** Robust schema validation and imputation strategies

---

## 📁 Project Structure

```
Spark_project/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── code/
│   ├── notebook.ipynb           # EDA, training, and model development
│   ├── app.py                   # Production inference application
│   ├── plane-data.csv           # Auxiliary plane metadata
│   └── models/
│       └── best_flight_delay_model/  # Saved CrossValidatorModel
├── training_data/
│   └── flight_data/
│       ├── 2006.csv.bz2         # Training data (2006)
│       ├── 2007.csv.bz2         # Training data (2007)
│       ├── airports.csv         # Airport metadata
│       ├── carriers.csv         # Carrier codes
│       ├── plane-data.csv       # Aircraft information
│       └── variable-descriptions.csv
├── test_data/
│   ├── test_data_2008.csv       # Test dataset (2008)
│   └── reduce.py                # Data reduction utility
└── train_dataset/               # Intermediate training files
```

---

## 📦 Requirements

- **Python:** 3.10+
- **Apache Spark:** 4.0.1
- **Conda Environment:** `ds_spark_env` (recommended)

### Python Dependencies

```
pyspark==4.0.1
pandas==2.3.3
numpy==2.2.6
scipy==1.15.2
matplotlib==3.10.7
seaborn==0.13.2
```

---

## 🚀 Installation

### 1. Clone or Download the Project

```bash
cd /path/to/Spark_project
```

### 2. Create Conda Environment

```bash
conda create -n ds_spark_env python=3.10
conda activate ds_spark_env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Spark Installation

```bash
pyspark --version
# Expected: version 4.0.1
```

---

## 📊 Dataset

The project uses the **Airline On-Time Performance Data** from the U.S. Bureau of Transportation Statistics.

### Training Data
- **Years:** 2006-2007
- **Format:** Compressed CSV (`.csv.bz2`)
- **Size:** ~14 million flight records

### Test Data
- **Year:** 2008
- **Format:** CSV

### Key Variables

| Variable | Description |
|----------|-------------|
| `ArrDelay` | Arrival delay in minutes (target) |
| `DepDelay` | Departure delay in minutes |
| `Distance` | Flight distance in miles |
| `UniqueCarrier` | Airline carrier code |
| `Origin` / `Dest` | Airport IATA codes |
| `CRSDepTime` | Scheduled departure time |
| `TaxiOut` | Taxi out time in minutes |

### Forbidden Variables (Data Leakage)

These variables are excluded during training as they contain future information:
- `ArrTime`, `ActualElapsedTime`, `AirTime`, `TaxiIn`
- `CarrierDelay`, `WeatherDelay`, `NASDelay`, `SecurityDelay`, `LateAircraftDelay`

---

## 💻 Usage

### Training (Notebook)

The Jupyter notebook contains the complete ML pipeline:

1. **Start Jupyter:**
   ```bash
   conda activate ds_spark_env
   jupyter notebook code/notebook.ipynb
   ```

2. **Pipeline Steps:**
   - Data loading and schema inspection
   - Exploratory Data Analysis (EDA)
   - Feature engineering (PlaneAge, DepHour, etc.)
   - Data cleaning and imputation
   - Model training with Linear Regression
   - Hyperparameter tuning via CrossValidator
   - Model evaluation and saving

### Inference (Application)

The standalone application loads the pre-trained model and performs predictions on new data.

#### Basic Usage

```bash
cd code/
spark-submit app.py <path_to_test_data.csv>
```

#### Example

```bash
spark-submit app.py ../test_data/test_data_2008.csv
```

#### Application Features

- **Schema Validation:** Strict input schema enforcement
- **Automatic Preprocessing:** Applies same ETL as training
- **Performance Metrics:** Outputs RMSE and R² scores
- **Sample Predictions:** Displays example predictions for verification

#### Expected Output

```
[INFO] Path validation successful. Starting Spark application...
Schema validation passed successfully.
[QA INFO] Total records after join: X
[QA INFO] Final records ready for prediction/test: Y

==================================================
MODEL PERFORMANCE ON TEST DATA
==================================================
Root Mean Squared Error (RMSE): XX.XXXX minutes
R-squared (R2): X.XXXX
==================================================

Sample Predictions:
+-------------+--------+--------+----------+
|UniqueCarrier|DepDelay|ArrDelay|prediction|
+-------------+--------+--------+----------+
...
```

---

## 🤖 Model Details

### Algorithm
- **Model:** Linear Regression
- **Library:** Spark MLlib

### Feature Pipeline

1. **VectorAssembler:** Combines numeric features
2. **StandardScaler:** Normalizes numeric features
3. **StringIndexer:** Encodes categorical variables
4. **OneHotEncoder:** Creates dummy variables for categories
5. **Final VectorAssembler:** Combines all features

### Features Used

| Type | Features |
|------|----------|
| **Numeric** | `DepDelay`, `TaxiOut`, `Distance`, `PlaneAge`, `CRSDepTime`, `CRSArrTime`, `Month`, `DayOfWeek`, `DepHour` |
| **Categorical** | `UniqueCarrier`, `Origin`, `Dest`, `PlaneManufacturer` |

### Hyperparameter Tuning

- **Method:** K-Fold Cross-Validation
- **Metric:** RMSE (Root Mean Squared Error)
- **Parameters Tuned:** `regParam`, `elasticNetParam`

---

## 📈 Performance Metrics

The model is evaluated using:

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error (lower is better) |
| **R²** | Coefficient of determination (higher is better, max 1.0) |

---

## 📝 Notes

- Ensure `plane-data.csv` is present in the `code/` directory for the inference app
- The model handles missing categorical values with mode imputation
- Cancelled and diverted flights are excluded from predictions
- Input data must follow the exact schema defined in `EXPECTED_SCHEMA`

---

## 📄 License

This project was developed for academic purposes at UPM (Universidad Politécnica de Madrid).
