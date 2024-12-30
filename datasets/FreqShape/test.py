import torch

# Load the two .pt files
file1_path = 'split=1.pt'
file2_path = 'split=2.pt'

# Load the datasets
data1 = torch.load(file1_path)
data2 = torch.load(file2_path)

# Determine the sizes of the two datasets
size1 = len(data1)
size2 = len(data2)

# Calculate the proportion
proportion = size1 / size2 if size2 != 0 else None

print(size1, size2, proportion)

# Attempt to inspect the contents of the .pt files
def inspect_file(file_path):
    try:
        data = torch.load(file_path)
        # Return a summary of the contents if possible
        if isinstance(data, (list, tuple)):
            return {'type': type(data), 'length': len(data), 'sample_content': data[:2]}  # First 2 items as sample
        elif isinstance(data, dict):
            return {'type': type(data), 'keys': list(data.keys()), 'sample_content': {k: data[k] for k in list(data.keys())[:2]}}
        else:
            return {'type': type(data), 'content': data}
    except Exception as e:
        return {'error': str(e)}

content1 = inspect_file(file1_path)
content2 = inspect_file(file2_path)

print(content1, content2)

# Function to calculate the train and val sizes and their ratio
def calculate_ratios(data):
    try:
        train_loader_size = len(data['train_loader']) if hasattr(data['train_loader'], '__len__') else None
        val_size = len(data['val'][0]) if isinstance(data['val'], tuple) and hasattr(data['val'][0], '__len__') else None
        if train_loader_size is not None and val_size is not None:
            ratio = train_loader_size / val_size if val_size != 0 else None
            return train_loader_size, val_size, ratio
        else:
            return train_loader_size, val_size, None
    except Exception as e:
        return None, None, str(e)

# Load the files
file1_data = torch.load(file1_path)
file2_data = torch.load(file2_path)

# Calculate ratios
file1_ratios = calculate_ratios(file1_data)
file2_ratios = calculate_ratios(file2_data)

print(file1_ratios, file2_ratios)

