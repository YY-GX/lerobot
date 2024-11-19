import os
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader
import pickle
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from itertools import groupby
from operator import itemgetter
from tqdm import tqdm

NUM_PROCESS = 32

# Step 1: Define directory for storing the dataset
DATA_DIR = "data/lerobot/"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Step 2: Download and save the original dataset
print("Downloading the original dataset...")
dataset = LeRobotDataset(
    "yygx/libero44_dataset_converted_externally_to_rlds_to_hg",
    split="train",
)

original_dataset_path = os.path.join(DATA_DIR, "original_dataset.pkl")
with open(original_dataset_path, 'wb') as f:
    pickle.dump(dataset, f)
print(f"Original dataset saved to {original_dataset_path}")

# Step 3: Split the dataset into 44 parts and remove 'observation.images.wrist_image'
tasks = defaultdict(list)

# Group indices by language instruction more efficiently using multiprocessing
process_num = min(NUM_PROCESS, cpu_count())
print(f">> NUM of CPU: {cpu_count()}")


def group_indices(args):
    start_idx, end_idx = args
    local_tasks = defaultdict(list)
    for idx in tqdm(range(start_idx, end_idx), desc=f"Processing indices from {start_idx} to {end_idx}"):
        data_point = dataset[idx]
        language_instruction = data_point["language_instruction"]
        local_tasks[language_instruction].append(idx)
    return local_tasks


# Split dataset indices into chunks for multiprocessing
chunk_size = len(dataset) // process_num
chunks = [(i * chunk_size, (i + 1) * chunk_size if i != process_num - 1 else len(dataset)) for i in range(process_num)]

with Pool(processes=process_num) as pool:
    results = list(
        tqdm(pool.imap(group_indices, chunks), total=len(chunks), desc="Grouping instructions with multiprocessing"))


# Merge results from all processes using multiprocessing
def merge_results(result_chunks):
    merged_tasks = defaultdict(list)
    for result in result_chunks:
        for key, value in result.items():
            merged_tasks[key].extend(value)
    return merged_tasks


# Perform final merge to create the complete tasks dictionary
tasks = merge_results(results)


# Remove 'observation.images.wrist_image' key from each data point
def remove_wrist_image_key(data_point):
    data_point = dict(data_point)
    if "observation.images.wrist_image" in data_point:
        del data_point["observation.images.wrist_image"]
    return data_point


class ModifiedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        data_point = self.original_dataset[idx]
        return remove_wrist_image_key(data_point)


# Function to process and save each split dataset
def process_and_save_split(args):
    instruction, indices = args
    split_dataset = torch.utils.data.Subset(dataset, indices)
    modified_dataset = ModifiedDataset(split_dataset)
    dataset_filename = f"split_dataset_{instruction.replace(' ', '_')}.pkl"
    dataset_path = os.path.join(DATA_DIR, dataset_filename)
    with open(dataset_path, 'wb') as f:
        pickle.dump(modified_dataset, f)
    print(f"Saved split dataset '{instruction}' to {dataset_path}")


# Use multiprocessing to process and save datasets in parallel
num_processes = min(NUM_PROCESS, cpu_count())
with Pool(processes=num_processes) as pool:
    list(tqdm(pool.imap(process_and_save_split, tasks.items()), total=len(tasks), desc="Saving split datasets"))


# Step 4: Verification script to check if all split datasets sum to the original dataset
def validate_split(original_dataset, split_datasets_dir):
    total_length = 0
    for file_name in tqdm(os.listdir(split_datasets_dir), desc="Validating split datasets"):
        if file_name.startswith("split_dataset_") and file_name.endswith(".pkl"):
            with open(os.path.join(split_datasets_dir, file_name), 'rb') as f:
                split_dataset = pickle.load(f)
                total_length += len(split_dataset)
    original_length = len(original_dataset)

    assert total_length == original_length, (
        f"Validation failed: Total length of split datasets ({total_length}) does not match "
        f"original dataset length ({original_length})"
    )
    print("Validation passed: The split datasets sum up to the original dataset length.")


# Run validation
validate_split(dataset, DATA_DIR)
