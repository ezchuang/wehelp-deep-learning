import os
import csv

def count_csv_rows(folder_path: str, skip_header: bool = True) -> int:
    total_rows = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, mode='r', encoding='utf-8') as csv_file:
                    reader = csv.reader(csv_file)
                    first_line = True
                    for row in reader:
                        if first_line and skip_header:
                            first_line = False
                            continue
                        total_rows += 1
            except Exception as e:
                print(f"Load {file_path} Failed: {e}")
    return total_rows

if __name__ == "__main__":
    folder = "stage-2\\crawler_data"
    total = count_csv_rows(folder, skip_header=True)
    print(f"Total crawled rows: {total}")

    folder = "stage-2\\cleaned_data"
    total = count_csv_rows(folder, skip_header=True)
    print(f"Total cleaned rows: {total}")

    folder = "stage-2\\tokenized_data"
    total = count_csv_rows(folder, skip_header=False)
    print(f"Total tokenized rows: {total}")

    folder = "stage-2\\embedded_data"
    total = count_csv_rows(folder, skip_header=False)
    print(f"Total embedded rows: {total}")
