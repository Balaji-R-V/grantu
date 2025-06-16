import pandas as pd
import psycopg2
from psycopg2 import sql

# Database connection parameters
db_params = {
    'dbname': 'profiles_db',
    'user': 'postgres',
    'password': '#####',  
    'host': 'localhost',
    'port': '5432'  # Default PostgreSQL port
}

# Path to your Excel file
excel_path = r"C:\Users\NANDHINI\Desktop\ProductX\AI_Chatbot_Part\synthetic_profiles_with_varied_orgs.xlsx"

def clean_data(value):
    """Clean data and handle None/NaN values"""
    if pd.isna(value) or value == '' or str(value).lower() == 'nan':
        return None
    return str(value).strip()

# Read Excel file
try:
    df = pd.read_excel(excel_path)
    print(f"Successfully read Excel file with {len(df)} rows")
    print("Column names:", df.columns.tolist())
    
    # Display first few rows to verify data
    print("\nFirst 3 rows of data:")
    print(df.head(3))
    
except Exception as e:
    print(f"Error reading Excel file: {e}")
    exit()

# Connect to PostgreSQL
try:
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    print("Successfully connected to PostgreSQL")

    # Clear existing data (optional - remove this if you want to keep existing data)
    cursor.execute("DELETE FROM profiles;")
    print("Cleared existing data from profiles table")

    # Insert data into the profiles table
    successful_inserts = 0
    failed_inserts = 0
    
    for index, row in df.iterrows():
        try:
            insert_query = sql.SQL("""
                INSERT INTO profiles (full_name, years_exp, current_org, past_org, skill_set, linkedin_profile)
                VALUES (%s, %s, %s, %s, %s, %s)
            """)
            
            # Clean and prepare data
            full_name = clean_data(row['Full Name'])
            years_exp = clean_data(row['Year of Exp'])  # Keep as text (6+ yr, 7+ yr, etc.)
            current_org = clean_data(row['Current Organisation'])
            past_org = clean_data(row['Past Organisation'])
            skill_set = clean_data(row['Skill Set'])
            linkedin_profile = clean_data(row['LinkedIn Profile'])
            
            cursor.execute(insert_query, (
                full_name,
                years_exp,
                current_org,
                past_org,
                skill_set,
                linkedin_profile
            ))
            successful_inserts += 1
            
            # Print progress every 10 records
            if (index + 1) % 10 == 0:
                print(f"Processed {index + 1} records...")
                
        except Exception as row_error:
            failed_inserts += 1
            print(f"Error inserting row {index + 1}: {row_error}")
            print(f"Row data: {row.to_dict()}")

    # Commit the transaction
    conn.commit()
    print(f"\n=== IMPORT SUMMARY ===")
    print(f"Successfully inserted: {successful_inserts} records")
    print(f"Failed insertions: {failed_inserts} records")
    print(f"Total records processed: {len(df)}")

except psycopg2.Error as db_error:
    print(f"Database Error: {db_error}")
except Exception as e:
    print(f"General Error: {e}")
finally:
    if 'cursor' in locals():
        cursor.close()
    if 'conn' in locals():
        conn.close()
    print("Database connection closed")

print("Excel to PostgreSQL import complete!")

# Verify the import
try:
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM profiles;")
    count = cursor.fetchone()[0]
    print(f"\nVerification: Total records in database: {count}")
    
    # Show first 3 records
    cursor.execute("SELECT full_name, years_exp, current_org, linkedin_profile FROM profiles LIMIT 3;")
    sample_records = cursor.fetchall()
    
    print("\nSample records in database:")
    for i, record in enumerate(sample_records, 1):
        print(f"{i}. Name: {record[0]}, Experience: {record[1]}, Company: {record[2]}")
        print(f"   LinkedIn: {record[3]}")
    
except Exception as e:
    print(f"Error during verification: {e}")
finally:
    if 'cursor' in locals():
        cursor.close()
    if 'conn' in locals():
        conn.close()