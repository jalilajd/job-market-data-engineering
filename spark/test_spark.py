from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("RekruteSparkTest")
    .master("local[*]")
    .getOrCreate()
)

data = [
    ("Data Engineer", "Casablanca"),
    ("Data Scientist", "Rabat"),
    ("Développeur Python", "Tanger")
]

df = spark.createDataFrame(data, ["job_title", "city"])

df.show()

print("Nombre d'offres :", df.count())

spark.stop()