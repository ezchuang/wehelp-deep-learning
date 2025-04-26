import logging
import sys
import os
from datetime import datetime

def default_logger(name: str = "my_logger", log_level=logging.INFO, log_dir="logs"):
    # 建立 logs 目錄（如果不存在）
    os.makedirs(log_dir, exist_ok=True)

    # 檔案名稱加上 timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{name}_{timestamp}.log")

    # 建立 logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False  # 避免重複輸出

    # formatter 格式設定
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # FileHandler：寫入檔案
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # StreamHandler：輸出到 terminal
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
