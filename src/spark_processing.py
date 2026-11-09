
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "rekrute_raw.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

# =====================================================
# 2. CREATE SPARK SESSION
# =====================================================


spark = (
    SparkSession.builder
    .appName("JobMarketDataEngineering")
    .master("local[*]")
    .config("spark.driver.memory", "2g")
    .config("spark.hadoop.io.native.lib.available", "false")
    .config("spark.hadoop.fs.file.impl",
            "org.apache.hadoop.fs.RawLocalFileSystem")
    .getOrCreate()
)


spark.sparkContext.setLogLevel("ERROR")
print("Hadoop version:", spark._jvm.org.apache.hadoop.util.VersionInfo.getVersion())
try:
    # =================================================
    # 3. READ RAW CSV DATA
    # =================================================

    print("\n========== READING RAW DATA ==========")

    df_raw = (
        spark.read
        .format("csv")
        .option("header", "true")
        .option("multiLine", "true")
        .option("quote", '"')
        .option("escape", '"')
        .option("encoding", "UTF-8")
        .option("mode", "PERMISSIVE")
        .load(str(CSV_PATH))
    )

    # Rename technical column
    df = df_raw.withColumnRenamed("git status", "offer_id")

    # =================================================
    # 4. DATASET EXPLORATION
    # =================================================

    print("\n========== DATASET INFORMATION ==========")

    total_offers = df.count()

    print(f"Number of offers: {total_offers}")
    print(f"Number of columns: {len(df.columns)}")

    print("\n========== SCHEMA ==========")
    df.printSchema()

    print("\n========== FIRST 5 OFFERS ==========")

    df.select(
        "offer_id",
        "source",
        "title",
        "company",
        "location",
        "publication_date"
    ).show(5, truncate=False)

    # =================================================
    # 5. DATA QUALITY CHECKS
    # =================================================

    print("\n========== DATA QUALITY ==========")

    print("Missing values per column:")

    for column in df.columns:
        missing_count = df.filter(
            col(column).isNull() | (col(column) == "")
        ).count()

        print(f"{column}: {missing_count}")

    # Check duplicate offer IDs
    duplicate_count = (
        df.count() - df.select("offer_id").distinct().count()
    )

    print(f"\nDuplicate offer IDs: {duplicate_count}")

    # =================================================
    # 6. END
    # =================================================

    print("\n========== SPARK PROCESSING COMPLETED ==========")

    from pyspark.sql.functions import col, trim, when, lit

    # =====================================================
    # DATA CLEANING
    # =====================================================

    print("\n========== DATA CLEANING ==========")

    # 1. Remove duplicate offers
    df_clean = df.dropDuplicates(["offer_id"])

    # 2. Replace empty strings and NULL values
    # 3. Remove unnecessary spaces
    for column in df_clean.columns:
        df_clean = df_clean.withColumn(
            column,
            when(
                col(column).isNull() | (trim(col(column)) == ""),
                lit("Not Mentioned")
            ).otherwise(trim(col(column)))
        )

    print("Cleaning completed.")

    # 4. Verify missing values
    print("\n========== MISSING VALUES AFTER CLEANING ==========")

    for column in df_clean.columns:
        missing_count = df_clean.filter(
            col(column).isNull() | (col(column) == "")
        ).count()

        print(f"{column}: {missing_count}")

    # 5. Display cleaned data
    print("\n========== CLEANED DATA ==========")

    df_clean.show(5, truncate=False)

    print("Total offers after cleaning:", df_clean.count())

    OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "rekrute_cleaned.parquet"

    pdf = df_clean.toPandas()
    pdf.to_parquet(OUTPUT_PATH, engine="pyarrow", index=False)

    print("Parquet file saved successfully!")





finally:
    spark.stop()
