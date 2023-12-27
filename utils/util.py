import os
import re
import torch
import datetime

def getSaveFileName(model_name = 'CNN', path = './res/'):
    # 获取当前时间
    current_time = datetime.datetime.now()

    # 格式化时间作为文件名
    # formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    formatted_time = current_time.strftime("_%Y-%m-%d-%H-%M")

    # 创建文件名
    file_name = path + model_name + str(formatted_time)

    return file_name

def getDevice(model_type = None):
    if model_type == "CNN" or model_type == "ResNet":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return "cpu"
    # return device

def find_max_epoch_file(folder_path='.', search_string=''):
    pth_files = [file for file in os.listdir(folder_path) if file.endswith('.pth') and search_string in file]

    if not pth_files:
        return None  # If no matching files are found, return None

    max_number = float('-inf')  # Initialize the maximum number to negative infinity
    max_pth_file = None

    for pth_file in pth_files:
        match = re.search(r'_(\d+)_', pth_file)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number
                max_pth_file = pth_file

    return max_pth_file
