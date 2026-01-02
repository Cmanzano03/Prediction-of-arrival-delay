# Create an application that can be used with different input files without having to change
# a single line of code, both for training and applying the model.

# The source code for the Spark application that:
# ● Loads test data from the specified path.
# ● Loads the best_model.
# ● Process the test data to be able to use the model.
# ● Performs some predictions
# ● Performs a complete performance test on the test data.
# Application should be able to be executed with spark-submit.



import sys
import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType

# -------------------------------------------------------------------------
# SCHEMA DEFINITION
# -------------------------------------------------------------------------
# Strict definition of the training schema to ensure consistency during inference.
# This schema acts as a contract for the input data.
EXPECTED_SCHEMA = StructType([
    StructField("Year", IntegerType(), True),
    StructField("Month", IntegerType(), True),
    StructField("DayofMonth", IntegerType(), True),
    StructField("DayOfWeek", IntegerType(), True),
    StructField("DepTime", IntegerType(), True),
    StructField("CRSDepTime", IntegerType(), True),
    StructField("ArrTime", IntegerType(), True),
    StructField("CRSArrTime", IntegerType(), True),
    StructField("UniqueCarrier", StringType(), True),
    StructField("FlightNum", IntegerType(), True),
    StructField("TailNum", StringType(), True),
    StructField("ActualElapsedTime", IntegerType(), True),
    StructField("CRSElapsedTime", IntegerType(), True),
    StructField("AirTime", IntegerType(), True),
    StructField("ArrDelay", IntegerType(), True),
    StructField("DepDelay", IntegerType(), True),
    StructField("Origin", StringType(), True),
    StructField("Dest", StringType(), True),
    StructField("Distance", IntegerType(), True),
    StructField("TaxiIn", IntegerType(), True),
    StructField("TaxiOut", IntegerType(), True),
    StructField("Cancelled", IntegerType(), True),
    StructField("CancellationCode", StringType(), True),
    StructField("Diverted", IntegerType(), True),
    StructField("CarrierDelay", IntegerType(), True),
    StructField("WeatherDelay", IntegerType(), True),
    StructField("NASDelay", IntegerType(), True),
    StructField("SecurityDelay", IntegerType(), True),
    StructField("LateAircraftDelay", IntegerType(), True)
])

def create_spark_session():
    """
    Initialize the Spark Session with appropriate configurations for Spark 4.0.1.
    """
    spark = SparkSession.builder \
        .appName("FlightDelayPredictionInference") \
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED") \
        .getOrCreate()
    return spark

def validate_input_schema(df):
    """
    Validates that the input DataFrame matches the expected training schema.
    
    Checks for:
    1. Missing columns.
    2. Data type mismatches (e.g., String instead of Integer).
    
    Raises:
        ValueError: If the schema is invalid, stopping execution immediately.
    """
    input_fields = {f.name: f.dataType for f in df.schema.fields}
    expected_fields = {f.name: f.dataType for f in EXPECTED_SCHEMA.fields}
    
    missing_columns = []
    mismatched_types = []
    
    for name, expected_type in expected_fields.items():
        if name not in input_fields:
            missing_columns.append(name)
        else:
            # We check if types are compatible. 
            # Note: We compare the simple string representation to avoid strict object identity issues
            # but usually, strict type checking is safer for ML models.
            current_type = input_fields[name]
            
            # Allow LongType to pass for IntegerType (Spark often infers Integers as Longs)
            is_compatible_numeric = (str(expected_type) == "IntegerType" and str(current_type) == "LongType")
            
            if current_type != expected_type and not is_compatible_numeric:
                mismatched_types.append(f"{name} (Expected {expected_type}, Got {current_type})")
    
    if missing_columns or mismatched_types:
        error_msg = "\n!!! SCHEMA VALIDATION FAILED !!!\n"
        if missing_columns:
            error_msg += f"Missing Columns: {', '.join(missing_columns)}\n"
        if mismatched_types:
            error_msg += f"Type Mismatches: {'; '.join(mismatched_types)}\n"
        error_msg += "The input test data must strictly follow the training schema."
        
        raise ValueError(error_msg)
        
    print("Schema validation passed successfully.")



def preprocess_test_data(spark, raw_df, planes_path):
    """
    Replicates the ETL, Feature Engineering, and Cleaning logic from the notebook.
    This ensures the test data has the exact schema expected by the PipelineModel
    and removes data quality issues that could skew the performance test.
    """
    
    # -------------------------------------------------------------------------
    # 1. INITIAL SETUP & VARIABLE PRUNING
    # -------------------------------------------------------------------------
    
    # Define variables containing future information (data leakage)
    # These are known only after the flight has landed.
    
    forbidden_vars = [
        "ArrTime",
        "ActualElapsedTime",
        "AirTime",
        "TaxiIn",
        "Diverted",
        "CarrierDelay",
        "WeatherDelay",
        "NASDelay",
        "SecurityDelay",
        "LateAircraftDelay"
    ]
    
    # Remove forbidden variables to prevent data leakage
    df_clean = raw_df.drop(*forbidden_vars)    
    
    # Filter out Cancelled and Diverted flights
    # Rationale: The model was trained only on completed flights.
    df_clean = df_clean.filter("Cancelled == 0 AND Diverted == 0")
    
    # Drop 'CRSElapsedTime' (correlated with Distance) and 'FlightNum' (high cardinality/noise)
    df_clean = df_clean.drop("CRSElapsedTime", "FlightNum")
    
    # -------------------------------------------------------------------------
    # 2. AUXILIARY DATA LOADING & CLEANING
    # -------------------------------------------------------------------------
    
    try:
        planes_df = spark.read.option("header", "true").option("inferSchema", "true").csv(planes_path)
        # Normalizing keys for a safer join
        planes_clean = planes_df.dropna(subset=["year", "manufacturer"]) \
                                .select(F.upper(F.trim(F.col("tailnum"))).alias("TailNum_Ref"), 
                                        F.col("year").alias("PlaneYear"),
                                        F.col("manufacturer").alias("PlaneManufacturer"))
    except Exception as e:
        raise IOError(f"Failed to load or process auxiliary planes file at {planes_path}: {str(e)}")
    # -------------------------------------------------------------------------
    # 3. ENRICHMENT & FEATURE ENGINEERING
    # -------------------------------------------------------------------------

    # Ensure TailNum in flight data is also normalized before join
    df_clean = df_clean.withColumn("TailNum", F.upper(F.trim(F.col("TailNum"))))
    
    enriched_df = df_clean.join(F.broadcast(planes_clean), 
                                df_clean.TailNum == planes_clean.TailNum_Ref, "left")
    
    
    # -------------------------------------------------------------------------
    # 4. FEATURE ENGINEERING: DERIVED VARIABLES
    # -------------------------------------------------------------------------          

    # Calculate PlaneAge: Flight Year - Manufactured Year
    # Note: Drops 'TailNum' and 'PlaneYear' immediately after calculation to keep schema clean.
    enriched_df = enriched_df.withColumn(
        "PlaneAge", 
        (F.col("Year") - F.expr("try_cast(PlaneYear AS INT)")).cast("int")
    ).drop("TailNum", "PlaneYear")

    # -------------------------------------------------------------------------
    # 5. ROBUST IMPUTATION & SANITIZATION (Notebook Logic Adaptation)
    # -------------------------------------------------------------------------
    
    # IMPUTATION:
    # To prevent "Unseen Label" errors in the model's StringIndexer, we must fill 
    # missing categorical values with a label known to the model during training.
    
    # Mode from training set (Must match training logic exactly)
    TRAIN_MODE_MANUFACTURER = "BOEING" 
    
    enriched_df = enriched_df.fillna({
        "PlaneAge": 7,  # Median from training EDA
        "PlaneManufacturer": TRAIN_MODE_MANUFACTURER 
    })

    # STRING SANITIZATION (Adapted from Notebook Step 3):
    # Trims whitespace to avoid categories like " AIRBUS" vs "AIRBUS".
    # Normalizes 'PlaneManufacturer' to lower case as done in training.
    enriched_df = enriched_df.withColumn(
        "PlaneManufacturer", 
        F.lower(F.trim(F.col("PlaneManufacturer")))
    )
    
    # SANITIZATION FOR CATEGORICALS:
    # Ensure standard categorical columns are trimmed (Notebook logic 'trim(col_name)')
    # This prevents mismatches due to accidental leading/trailing spaces in CSVs.
    for col_name in ["UniqueCarrier", "Origin", "Dest"]:
        enriched_df = enriched_df.withColumn(col_name, F.trim(F.col(col_name)))
    
    # -------------------------------------------------------------------------
    # 6. FINAL FEATURE ENGINEERING & LOGICAL FILTERING
    # -------------------------------------------------------------------------

    # Calculate DepHour: Extract scheduled hour from CRSDepTime
    enriched_df = enriched_df.withColumn(
        "DepHour", 
        F.floor(F.expr("try_cast(CRSDepTime AS INT)") / 100).cast("int")
    )

    # DATA INTEGRITY FILTERING (Adapted from Notebook Step 5):
    # 1. ArrDelay IS NOT NULL: Required for the "Performance Test" requested in requirements.
    #    (If this were pure production inference, we would accept Nulls, but for evaluation we cannot).
    # 2. PlaneAge >= 0: Removes data corruption where PlaneYear > FlightYear.
    enriched_df = enriched_df.filter(
        (F.col("ArrDelay").isNotNull()) & 
        (F.col("PlaneAge") >= 0) 
    )
# -------------------------------------------------------------------------
    # 7. FINAL ROBUST IMPUTATION (Global Safety Net)
    # -------------------------------------------------------------------------
    
    # We use pre-calculated statistics from the training phase to fill any 
    # remaining nulls. This prevents the PipelineModel from crashing.
    # Logic: 
    # - Numeric: Use Medians/Means to minimize variance impact.
    # - Categoric: Use the "Mode" (most frequent) to ensure compatibility with OHE.
    
    imputation_values = {
        'CRSArrTime': 1490.96,
        'CRSDepTime': 1324.43,
        'DayOfWeek': 3.94,
        'DepDelay': 8.76,
        'DepHour': 12.98,
        'Dest': 'ATL',
        'Distance': 689.85,
        'Month': 6.62,
        'Origin': 'ATL',
        'PlaneAge': 9.59,
        'PlaneManufacturer': 'boeing',
        'TaxiOut': 16.0,
        'UniqueCarrier': 'WN'
        }

    # Apply the imputation map only to columns that exist in the current DataFrame
    # This prevents errors if some columns were dropped earlier
    final_imputation = {col: val for col, val in imputation_values.items() if col in enriched_df.columns}
    
    final_df = enriched_df.fillna(final_imputation)
    # -------------------------------------------------------------------------
    # 8. SCHEMA ENFORCEMENT
    # -------------------------------------------------------------------------
    
    # Explicitly select columns in the order expected by the model's VectorAssembler.
    # This acts as a final safeguard against schema mismatches.
    final_columns = [
        "Month", "DayOfWeek", "DepHour", 
        "UniqueCarrier", "Origin", "Dest", "PlaneManufacturer", 
        "DepDelay", "TaxiOut", "Distance", "PlaneAge", 
        "CRSDepTime", "CRSArrTime", "ArrDelay"
    ]
    
    # Defensive selection: Ensure all columns exist before selecting
    # (In case 'Year' or others are still lingering, they are dropped here implicitly)
    return final_df.select(*final_columns)

def main(test_data_path):
    # Initialize Spark
    spark = create_spark_session()
    
    # Hardcoded paths based on project structure (can be moved to args if needed)
    # The model path is retrieved from the notebook's final save 
    model_path = "models/best_flight_delay_model"
    #PATH TO AUXILIARY DATA, MODIFY IF NEEDED TO MATCH YOUR STRUCTURE
    planes_path = "../training_data/flight_data/plane-data.csv"

    try:
        print(f"Loading test data from: {test_data_path}")
        
        # OPTION: Force the schema on read using our defined struct.
        # This prevents Spark from inferring wrong types (e.g. "NA" as string in an Int column).
        raw_test_df = spark.read.option("header", "true") \
                                .option("nullValue", "NA") \
                                .schema(EXPECTED_SCHEMA) \
                                .csv(test_data_path)

        # Validate strictly (Double check)
        validate_input_schema(raw_test_df)

        # Apply same preprocessing logic used during training
        processed_test_df = preprocess_test_data(spark, raw_test_df, planes_path, airports_path)

        # Load the best_model (PipelineModel includes all preprocessing stages) 
        print(f"Loading trained PipelineModel from: {model_path}")
        model = PipelineModel.load(model_path)

        # Perform predictions
        # The transform() method handles Indexing, Encoding, and Scaling automatically 
        predictions = model.transform(processed_test_df)

        # Perform performance evaluation
        # We use RMSE (Primary) and R2 (Secondary) as defined in the notebook [cite: 12, 13]
        evaluator_rmse = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="rmse")
        evaluator_r2 = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="r2")

        rmse = evaluator_rmse.evaluate(predictions)
        r2 = evaluator_r2.evaluate(predictions)

        print("\n" + "="*50)
        print("MODEL PERFORMANCE ON TEST DATA")
        print("="*50)
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f} minutes")
        print(f"R-squared (R2): {r2:.4f}")
        print("="*50)

        # Show a sample of predictions for manual verification
        print("\nSample Predictions:")
        predictions.select("UniqueCarrier", "DepDelay", "ArrDelay", "prediction").show(10)

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    # Ensure the test data path is passed as an argument
    if len(sys.argv) < 2:
        print("Usage: spark-submit app.py <test_data_csv_path>")
        sys.exit(1)
    
    main(sys.argv[1])