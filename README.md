# DAE-CORNERSTONEPROJECT
from google.colab import files

uploaded = files.upload()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load the dataset
df = pd.read_csv("package_tourism.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

# Clean duplicate rows
df = df.drop_duplicates()

# Fill missing values for 'Place_Tourism4' and 'Place_Tourism5' which have NaNs
for col in ["Place_Tourism4", "Place_Tourism5"]:
    df[col] = df[col].fillna(df[col].mode()[0])


# Destination columns
place_columns = [
    "Place_Tourism1",
    "Place_Tourism2",
    "Place_Tourism3",
    "Place_Tourism4",
    "Place_Tourism5"
]


# Clean destination names
for column in place_columns:
    df[column] = df[column].str.strip()


# Count destinations in each package
df["Number_of_Destinations"] = df[place_columns].notna().sum(axis=1)

print("\nNumber of destinations in each package:")
print(df[["Package", "City", "Number_of_Destinations"]].head(10))


# ==========================================
# GRAPH 1: PACKAGES PER CITY
# ==========================================

city_count = df["City"].value_counts()

print("\nPackages per city:")
print(city_count)

plt.figure(figsize=(8, 5))

plt.bar(city_count.index, city_count.values)

plt.title("Number of Packages per City")
plt.xlabel("City")
plt.ylabel("Number of Packages")

plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# ==========================================
# GRAPH 2: DESTINATIONS PER PACKAGE
# ==========================================

places_count = df["Number_of_Destinations"].value_counts().sort_index()

print("\nDestinations per package:")
print(places_count)

plt.figure(figsize=(8, 5))

plt.bar(
    places_count.index.astype(str),
    places_count.values
)

plt.title("Number of Destinations per Package")
plt.xlabel("Number of Destinations")
plt.ylabel("Number of Packages")

plt.tight_layout()
plt.show()


# ==========================================
# GRAPH 3: AVERAGE DESTINATIONS BY CITY
# ==========================================

average_destinations = df.groupby(
    "City"
)["Number_of_Destinations"].mean()

print("\nAverage destinations by city:")
print(average_destinations)

plt.figure(figsize=(8, 5))

plt.bar(
    average_destinations.index,
    average_destinations.values
)

plt.title("Average Destinations by City")
plt.xlabel("City")
plt.ylabel("Average Number of Destinations")

plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# ==========================================
# GRAPH 4: TOP 10 DESTINATIONS
# ==========================================

all_destinations = pd.concat([
    df["Place_Tourism1"],
    df["Place_Tourism2"],
    df["Place_Tourism3"],
    df["Place_Tourism4"],
    df["Place_Tourism5"]
])

all_destinations = all_destinations.dropna()

destination_count = all_destinations.value_counts()

top10 = destination_count.head(10)

print("\nTop 10 destinations:")
print(top10)

plt.figure(figsize=(10, 6))

plt.barh(
    top10.index,
    top10.values
)

plt.title("Top 10 Tourist Destinations")
plt.xlabel("Number of Packages")
plt.ylabel("Destination")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()


# ==========================================
# BASIC STATISTICS
# ==========================================

print("\nBasic Statistics:")
print(df["Number_of_Destinations"].describe())


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n========== PROJECT SUMMARY ==========")

print("Total packages:", len(df))

print("Total cities:", df["City"].nunique())

print(
    "Total unique destinations:",
    all_destinations.nunique()
)

print(
    "Average destinations per package:",
    np.mean(df["Number_of_Destinations"])
)

print(
    "Most popular destination:",
    destination_count.index[0]
)
