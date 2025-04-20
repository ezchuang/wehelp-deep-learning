from typing import List, Tuple, Set
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import csv
import re
import requests
from constants import DATA_SOURCE_URL, BOARDS, DATA_SOURCE_COOKIES, CRAWLER_DATA_PATH
from bs4 import BeautifulSoup

BATCH_SIZE: int = 10

def fetch_index(url: str) -> str:
    response: requests.Response = requests.get(url, cookies=DATA_SOURCE_COOKIES)
    response.raise_for_status()
    return response.text

def parse_index(html: str) -> List[Tuple[str, str, str, str]]:
    soup: BeautifulSoup = BeautifulSoup(html, "html.parser")
    articles: List[Tuple[str, str, str, str]] = []
    for entry in soup.find_all("div", class_="r-ent"):
        title: str = ""
        push: str = ""
        date_str: str = ""
        link: str = ""

        # Pushes
        nrec_div = entry.find("div", class_="nrec")
        push = nrec_div.get_text(strip=True) if nrec_div else ""
        
        # Date
        date_div = entry.find("div", class_="date")
        date_str = date_div.get_text(strip=True) if date_div else ""
        
        # Title and URL
        title_div = entry.find("div", class_="title")
        if title_div:
            a_tag = title_div.find("a")
            if a_tag:
                title = a_tag.get_text(strip=True)
                link = DATA_SOURCE_URL + a_tag['href']
            else:
                continue
        else:
            continue
        
        articles.append((title, push, date_str, link))
    return articles

def writer_thread(csv_filename: str, q: "queue.Queue[Tuple[str, str, str, str]]", stop_event: threading.Event) -> None:
    with open(csv_filename, "w", newline='', encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        
        # Header
        writer.writerow(["Title", "Push", "Date", "URL"])

        while not stop_event.is_set() or not q.empty():
            try:
                data: Tuple[str, str, str, str] = q.get(timeout=1)
                writer.writerow(data)
                q.task_done()
            except queue.Empty:
                continue

def crawl(board: str, target_count: int = 200000) -> None:
    os.makedirs(CRAWLER_DATA_PATH, exist_ok=True)
    
    # Get the latest index page
    index_url: str = f"{DATA_SOURCE_URL}/bbs/{board}/index.html"
    index_html: str = fetch_index(index_url)
    soup: BeautifulSoup = BeautifulSoup(index_html, "html.parser")
    
    # Get index of latest page(assign it to current index)
    btns = soup.select("div.btn-group-paging a")
    prev_page_url: str = btns[1]['href']
    match = re.search(r'index(\d+).html', prev_page_url)
    if match:
        current_index: int = int(match.group(1)) + 1  # Latest index page number
    else:
        print("Cannot parse the index number.")
        return

    # Writer thread for CSV output
    results_queue: "queue.Queue[Tuple[str, str, str, str]]" = queue.Queue()
    stop_event: threading.Event = threading.Event()
    csv_filename: str = os.path.join(CRAWLER_DATA_PATH, f"{board}.csv")
    writer_t = threading.Thread(target=writer_thread, args=(csv_filename, results_queue, stop_event))
    writer_t.start()

    seen_links: Set[str] = set()  # Filter out duplicate articles
    fetched_count: int = 0

    # Crawl pages from the latest index backwards until target_count is reached
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        futures = {}

        while fetched_count < target_count and current_index > 0:
            for _ in range(BATCH_SIZE):
                if current_index <= 0:
                    break

                url: str = f"{DATA_SOURCE_URL}/bbs/{board}/index{current_index}.html"
                futures[executor.submit(fetch_index, url)] = current_index
                current_index -= 1

            for future in as_completed(futures):
                try:
                    index_html: str = future.result()
                except Exception as e:
                    print(f"Failed to fetch page {futures[future]}, skipping.")
                    continue

                articles = parse_index(index_html)
                for title, push, date_str, link in articles:
                    if link in seen_links:
                        continue
                    seen_links.add(link)

                    results_queue.put((title, push, date_str, link))
                    fetched_count += 1
                    if fetched_count >= target_count:
                        break

                if fetched_count >= target_count:
                    break

            futures.clear()

    results_queue.join()
    stop_event.set()
    writer_t.join()
    print(f"Completed crawling {fetched_count} records and saved to {csv_filename}")

if __name__ == "__main__":
    for board in BOARDS:
        crawl(board)
