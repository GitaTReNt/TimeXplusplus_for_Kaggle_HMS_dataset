import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def convert_parquet_to_npy(parquet_folder, output_folder, fixed_length=10000):
    """
    将指定目录下的 .parquet 文件转换为 .npy 文件
    :param parquet_folder: 保存 .parquet 文件的路径
    :param output_folder: 转换后保存 .npy 文件的路径
    :param fixed_length: 固定的时间步长（默认为 50 秒 * 200 Hz = 10000）
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(parquet_folder):
        if file_name.endswith(".parquet"):
            file_path = os.path.join(parquet_folder, file_name)
            output_path = os.path.join(output_folder, file_name.replace(".parquet", ".npy"))

            # 加载 .parquet 文件
            with pq.ParquetFile(file_path) as parquet_file:
                table = parquet_file.read_row_group(0).to_pandas()
                eeg_data = table.values  # 转换为 NumPy 数组

                # 确保固定时间步长
                if eeg_data.shape[0] < fixed_length:
                    eeg_data = np.pad(
                        eeg_data, ((0, fixed_length - eeg_data.shape[0]), (0, 0)), mode="constant"
                    )
                elif eeg_data.shape[0] > fixed_length:
                    eeg_data = eeg_data[:fixed_length]

            # 保存为 .npy 文件
            np.save(output_path, eeg_data)
            print(f"Saved {output_path}")

parquet_folder = "E:/kaggle/hms/test_eegs"  # 原始 .parquet 文件路径
output_folder = "E:/kaggle/hms/test_eegs_npy"  # 转换后的 .npy 文件路径
convert_parquet_to_npy(parquet_folder, output_folder)
