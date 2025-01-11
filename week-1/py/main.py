from urllib import request
from urllib.parse import urlencode, urljoin
from functools import reduce
import os
import json
import csv
import statistics

DIR = "./week-1/py"
TASK_1_FILE = "products.txt"
TASK_2_FILE = "best-products.txt"
TASK_4_FILE = "standardization.csv"

PCHOME_URL = "https://ecshweb.pchome.com.tw/search/v4.3/all/results"
CATE_ID = "DSAA31"
ATTRIBUTE = "G26I2272" # i5
PAGE_COUNT = 40

class Scraper():
    def __init__(self, target_url: str):
        self.target_url = target_url

    def fetch_data(self, cate_id: str = None, attr: str = "", page_count: int = PAGE_COUNT, init_page: int = 1) -> list:
        all_prods = []
        url_params = {
            "cateid": cate_id,
            "attr": attr,
            "pageCount": page_count,
            "page": init_page
        }
        total_page = init_page

        while url_params["page"] <= total_page:
            try:
                query_url = f"{self.target_url}?{urlencode(url_params)}"
                
                with request.urlopen(query_url) as res:
                    data = json.loads(res.read().decode("utf-8")) # I forgot to use json.load() 
                    
                    if not data:
                        raise Exception("There is no data in the response.")
                    if not data["TotalPage"]:
                        raise Exception("There is no total page in the response.")
                    if not data["Prods"]:
                        raise Exception("There is no product in the response.")
                    
                    total_page = data["TotalPage"]
                    all_prods.extend(data["Prods"])

            except Exception as e:
                print(e)
                break

            url_params["page"] += 1

        return all_prods

class TaskHelper():
    @staticmethod
    def find_IDs(data: list[dict]) -> list:
        res_mapped = map(lambda d: d.get("Id", ""), data)
        return list(res_mapped)
    
    @staticmethod
    def find_best_product(data: list[dict]) -> list:
        res_filtered = filter(lambda d: d.get("ratingValue") and d.get("ratingValue", 0) > 4.9 and d.get("reviewCount", 0) >= 1, data)
        res_mapped = map(lambda d: d.get("Id", ""), res_filtered)
        return list(res_mapped)
    
    @staticmethod
    def calculate_ave_price(data: list[dict]) -> float:
        total_price, count = reduce(
            lambda acc, d: (
                acc[0] + d.get("Price", 0),
                acc[1] + 1
            ), data, (0, 0))
        return total_price / count
    
    @staticmethod
    def calculate_z_score(data: list[dict]) -> list:
        price_list = list(map(lambda d: d.get("Price", 0), data))
        price_pstdev = statistics.pstdev(price_list)
        price_mean = statistics.mean(price_list)

        res_mapped = map(
            lambda d: [
                d.get("Id", ""),
                d.get("Price", 0),
                round((d.get("Price", 0) - price_mean) / price_pstdev, 2)
            ], data)
        return list(res_mapped)

class FileWriter():
    @staticmethod
    def write_to_file(file_name: str, data: list) -> bool:
        file_path = os.path.join(DIR, file_name)
        try:
            if file_name.endswith(".txt"):
                return FileWriter.write_to_file_txt(file_path, data)
            elif file_name.endswith(".csv"):
                return FileWriter.write_to_file_csv(file_path, data)
            else:
                raise Exception("The extension is not supported.")
        except Exception as e:
            print(e)
            return False
        
    @staticmethod
    def write_to_file_txt(file_path: str, data: list) -> bool:
        try:
            with open(file_path, mode = "w") as file:
                file.write("\n".join(data))
            return True
        except Exception as e:
            print(f"Error writing TXT file: {e}")
            return False
    
    @staticmethod
    def write_to_file_csv(file_path: str, data: list) -> bool:
        try:
            with open(file_path, mode = "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(data)
            return True
        except Exception as e:
            print(f"Error writing CSV file: {e}")
            return False

class TaskHandler():
    def __init__(self, scraper: Scraper):
        self.scraper = scraper

    def task_1(self):
        products = scraper.fetch_data(cate_id = CATE_ID)
        data_cleaned_task_1 = TaskHelper.find_IDs(products)
        result = FileWriter.write_to_file(TASK_1_FILE, data_cleaned_task_1)
        if not result:
            print("Error occurred in Task 1")

    def task_2(self):
        products = scraper.fetch_data(cate_id = CATE_ID)
        data_cleaned_task_2 = TaskHelper.find_best_product(products)
        result = FileWriter.write_to_file(TASK_2_FILE, data_cleaned_task_2)
        if not result:
            print("Error occurred in Task 2")

    def task_3(self):
        products = scraper.fetch_data(cate_id = CATE_ID, attr = ATTRIBUTE)
        calculate_res = TaskHelper.calculate_ave_price(products)
        print(calculate_res)
        if not calculate_res:
            print("Error occurred in Task 3")

    def task_4(self):
        products = scraper.fetch_data(cate_id = CATE_ID)
        data_cleaned_task_4 = TaskHelper.calculate_z_score(products)
        result = FileWriter.write_to_file(TASK_4_FILE, data_cleaned_task_4)
        if not result:
            print("Error occurred in Task 4")


if __name__ == "__main__":
    scraper = Scraper(PCHOME_URL)
    task_handler = TaskHandler(scraper)
    task_handler.task_1()
    task_handler.task_2()
    task_handler.task_3()
    task_handler.task_4()
