import os
import csv
from typing import List

INPUT_CSV_DIR: str = "./stage-2/crawler_data"
OUTPUT_CSV_DIR: str = "./stage-2/cleaned_data"

os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

def clean_title(title: str) -> str:
    cleaned = title.strip()
    cleaned_lower = cleaned.lower()
    if cleaned_lower.startswith("re:") or cleaned_lower.startswith("fw:"):
        return ""
    return cleaned_lower

def process_csv_file(filename: str, input_filepath: str, output_filepath: str) -> None:
    with open(input_filepath, "r", encoding="utf-8") as in_file, \
         open(output_filepath, "w", newline='', encoding="utf-8") as out_file:
        reader = csv.DictReader(in_file)
        fieldnames: List[str] = reader.fieldnames
        if fieldnames is None:
            print(f"No fieldnames in CSV file: {input_filepath}")
            return
        
        output_fieldnames = ["Board", "Title"]
        writer = csv.DictWriter(out_file, fieldnames=output_fieldnames)
        writer.writeheader()
        for row in reader:
            title: str = clean_title(row.get("Title", ""))
            if title == "":
                continue

            new_row = {
                "Board": filename,
                "Title": title
            }
            writer.writerow(new_row)

def clean_data() -> None:
    for filename in os.listdir(INPUT_CSV_DIR):
        if filename.endswith(".csv"):
            input_filepath = os.path.join(INPUT_CSV_DIR, filename)
            output_filepath = os.path.join(OUTPUT_CSV_DIR, f"cleaned_{filename}")
            print(f"Processing {input_filepath} -> {output_filepath}")
            process_csv_file(filename.removesuffix(".csv"), input_filepath, output_filepath)
            
    print("Data cleaning complete.")

if __name__ == "__main__":
    clean_data()