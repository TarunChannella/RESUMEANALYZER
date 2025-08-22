import pandas as pd
from pymongo import MongoClient
import os
import re
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
DATABASE_NAME = 'jobsinline'
COLLECTION_NAME = 'job_roles_data'
EXCEL_FILE_PATH = r"C:\Users\tarun\OneDrive\Desktop\New2\backend\jobrolespskillsframeworks.xlsx"
  # Do not change this

MODEL_PATH = "job_role_model.pkl"  # ML model will be saved here


# ----------------------------
# Utility Functions
# ----------------------------
def standardize_job_name(job_name: str) -> str:
    """Standardize job names for better matching."""
    if not isinstance(job_name, str):
        return ""

    standardized = re.sub(r'[^\w\s]', '', job_name.lower().strip())
    standardized = re.sub(r'\b(senior|junior|lead|principal)\b', '', standardized)
    standardized = re.sub(r'\b(i|ii|iii|iv|intern)\b', '', standardized)
    standardized = re.sub(r'\s+', ' ', standardized).strip()
    return standardized


def clean_list_field(value: str) -> list:
    """Convert comma-separated strings to list of lowercase items."""
    if not isinstance(value, str):
        return []
    return [item.strip().lower() for item in value.split(',') if item.strip()]


# ----------------------------
# Migration + ML Training
# ----------------------------
def migrate_data():
    try:
        # Load Excel
        df = pd.read_excel(EXCEL_FILE_PATH, sheet_name='Sheet1')
        print("✅ Excel file loaded successfully.")

        # Connect MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]

        # Drop old collection
        if COLLECTION_NAME in db.list_collection_names():
            collection.drop()
            print(f"⚠️ Existing collection '{COLLECTION_NAME}' dropped.")

        # Standardize job roles
        df['standardized_job_role'] = df['JOB ROLES'].apply(standardize_job_name)

        # Clean skills & frameworks
        if 'PROGRAMMING SKILLS' in df.columns:
            df['PROGRAMMING SKILLS'] = df['PROGRAMMING SKILLS'].apply(clean_list_field)
        if 'FRAMEWORKS' in df.columns:
            df['FRAMEWORKS'] = df['FRAMEWORKS'].apply(clean_list_field)

        # Insert into MongoDB
        data_to_insert = df.to_dict('records')
        result = collection.insert_many(data_to_insert)
        print(f"✅ Inserted {len(result.inserted_ids)} job role documents.")

        # ----------------------------
        # Train ML Model
        # ----------------------------
        print("\n🤖 Training ML model on job roles...")
        X = df['standardized_job_role']
        y = df['JOB ROLES']  # keep original as label

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression(max_iter=1000))
        ])

        pipeline.fit(X, y)
        joblib.dump(pipeline, MODEL_PATH)
        print(f"✅ ML model trained & saved at {MODEL_PATH}")

        # Sample verification
        print("\n📌 Sample job roles in database:")
        for i, doc in enumerate(collection.find().limit(5)):
            print(f"{i+1}. Original: {doc['JOB ROLES']} -> Standardized: {doc['standardized_job_role']}")

    except FileNotFoundError:
        print(f"❌ Error: The file '{EXCEL_FILE_PATH}' was not found.")
    except Exception as e:
        print(f"❌ An error occurred during migration: {e}")
    finally:
        if 'client' in locals():
            client.close()
            print("🔌 MongoDB connection closed.")


if __name__ == '__main__':
    migrate_data()
