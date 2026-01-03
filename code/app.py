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
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml import PipelineModel # Keep this if you use it elsewhere
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

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
    # Set log level to ERROR to suppress INFO and WARN messages
    spark.sparkContext.setLogLevel("ERROR")
    
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
    df_clean = df_clean.filter("Cancelled == 0 AND Diverted == 0").drop("Cancelled", "Diverted", "CancellationCode")
    
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
    
    # QA Checkpoint: Monitor data loss and join success rate
    total_count = enriched_df.count()
    null_planes = enriched_df.filter(F.col("PlaneManufacturer").isNull()).count()
    print(f"[QA INFO] Total records after join: {total_count}")
    if total_count == 0:
        print("[QA CRITICAL] No records found after joining with plane metadata. Check input data quality.")
        sys.exit(1) # Stop execution before it fails in MLlib
    else:    
        print(f"[QA INFO] Flights without plane metadata (will be imputed): {null_planes} ({(null_planes/total_count)*100:.2f}%)")
    
    
    # -------------------------------------------------------------------------
    # 4. FEATURE ENGINEERING: DERIVED VARIABLES
    # -------------------------------------------------------------------------          

    # Calculate PlaneAge: Flight Year - Manufactured Year
    # Note: Drops 'TailNum' and 'PlaneYear' immediately after calculation to keep schema clean.
    enriched_df = enriched_df.withColumn(
        "PlaneAge", 
        (F.col("Year") - F.expr("try_cast(PlaneYear AS INT)")).cast("int")
    ).drop("TailNum", "PlaneYear", "TailNum_Ref", "Year")

    # -------------------------------------------------------------------------
    # 5. ROBUST IMPUTATION & SANITIZATION (Notebook Logic Adaptation)
    # -------------------------------------------------------------------------
    
    # IMPUTATION:
    # To prevent "Unseen Label" errors in the model's StringIndexer, we must fill 
    # missing categorical values with a label known to the model during training.
    
    # Mode from training set (Must match training logic exactly)
    TRAIN_MODE_MANUFACTURER = "boeing" 
    
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
        # Integer Columns: Must use integer values
        'CRSArrTime': 1491,   # Rounded from 1490.96
        'CRSDepTime': 1324,   # Rounded from 1324.43
        'DayOfWeek': 4,
        'DepDelay': 9,        # Rounded from 8.76
        'DepHour': 13,        # Rounded from 12.98
        'Distance': 690,      # Rounded from 689.85
        'Month': 7,           # Rounded from 6.62
        'PlaneAge': 10,       # Rounded from 9.59
        'TaxiOut': 16,        # Explicit integer
        
        # String Columns: Matching types
        'Dest': 'ATL',
        'Origin': 'ATL',
        'PlaneManufacturer': 'boeing',
        'UniqueCarrier': 'WN'
    }

    # Apply the imputation map only to columns that exist in the current DataFrame
    # This prevents errors if some columns were dropped earlier
    final_imputation = {col: val for col, val in imputation_values.items() if col in enriched_df.columns}
    
    final_df = enriched_df.fillna(final_imputation)
    # -------------------------------------------------------------------------
    # 8. SCHEMA ENFORCEMENT
    # -------------------------------------------------------------------------

    final_count = final_df.count()
    print(f"[QA INFO] Final records ready for prediction/test: {final_count}")

    if final_count == 0:
        print("[QA CRITICAL] No valid records found for prediction. Check input data quality.")
        sys.exit(1) # Stop execution before it fails in MLlib
    
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
    
# -------------------------------------------------------------------------
    # DYNAMIC PATH RESOLUTION
    # -------------------------------------------------------------------------
    # 1. Get the absolute path of the current script (app.py)
    # This works regardless of where you call spark-submit from.
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Pointing to 'code/'
    
    # 2. Get the project root (one level up from 'code/')
    project_root = os.path.dirname(script_dir)
    
    # 3. Build paths relative to the script location
    # Model is inside 'code/models/...'
    model_path = os.path.join(script_dir, "models", "best_flight_delay_model")
    
    # Plane data is inside 'training_data/flight_data/...'
    planes_path = os.path.join(project_root, "training_data", "flight_data", "plane-data.csv")

    # Log resolved paths for debugging (QA visibility)
    print(f"[DEBUG] Script directory: {script_dir}")
    print(f"[DEBUG] Resolved Model path: {model_path}")
    print(f"[DEBUG] Resolved Planes path: {planes_path}")

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
        processed_test_df = preprocess_test_data(spark, raw_test_df, planes_path)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        else:
            # -------------------------------------------------------------------------
            # LOAD MODEL (Handling CrossValidator vs Pipeline)
            # -------------------------------------------------------------------------
            print(f"[EXECUTION] Loading Model from: {model_path}")
            
            try:
                # We first try to load it as a CrossValidatorModel
                # because the notebook likely saved the entire CV process.
                cv_model = CrossValidatorModel.load(model_path)
                model = cv_model.bestModel
                print("[INFO] CrossValidatorModel detected. Best model extracted successfully.")
            except Exception:
                # Fallback for simple PipelineModels
                print("[INFO] Attempting to load as standard PipelineModel...")
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
        print("\nSample Predictions (Only a few columns to avoid collapsing the terminal):")
        predictions.select("UniqueCarrier", "DepDelay", "ArrDelay", "prediction").show(5)

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    # 1. Check number of arguments
    if len(sys.argv) < 2:
        print("[ERROR] Missing required argument.")
        print("Usage: spark-submit app.py <test_data_csv_path>")
        sys.exit(1)
    
    test_path = sys.argv[1]

    # 2. Check if the path exists (Local Filesystem Check)
    # Note: os.path.exists works for local paths. 
    # If using HDFS/S3, this check might need to be bypassed.
    if not os.path.exists(test_path):
        print(f"[ERROR] The file path provided does not exist: {test_path}")
        sys.exit(1)

    # 3. Check if it's a file and not a directory
    if not os.path.isfile(test_path):
        print(f"[ERROR] The path provided is not a file: {test_path}")
        sys.exit(1)

    # 4. (Optional) Check file extension
    valid_extensions = ('.csv', '.csv.bz2', '.bz2')
    if not test_path.lower().endswith(valid_extensions):
        print(f"[WARNING] The file '{test_path}' extension is not standard. "
              "Spark will attempt to infer the codec, but this might fail.")

    print(f"[INFO] Path validation successful. Starting Spark application for: {test_path}")
    
    # Proceed to main execution
    main(test_path)