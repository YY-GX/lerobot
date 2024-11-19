import torch
from collections import defaultdict
import pprint
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Load the original dataset
dataset = LeRobotDataset(
    "yygx/libero44_dataset_converted_externally_to_rlds_to_hg",
    split="train",
)

# Step 1: Split dataset into 44 parts based on the language instruction
tasks = defaultdict(list)

# Iterate over the dataset and group indices by language instruction
for idx, data_point in enumerate(dataset):
    language_instruction = data_point["language_instruction"]
    tasks[language_instruction].append(idx)

# Create a dictionary of datasets for each task
split_datasets = {
    instruction: torch.utils.data.Subset(dataset, indices)
    for instruction, indices in tasks.items()
}


# Step 2: Remove the 'observation.images.wrist_image' key from each data point
def remove_wrist_image_key(data_point):
    # Make a copy of the data point without modifying the original
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


# Apply the modification to each of the split datasets
modified_split_datasets = {
    instruction: ModifiedDataset(split_dataset)
    for instruction, split_dataset in split_datasets.items()
}


# Step 3: Test script to validate that the 44 datasets sum up to the original dataset
def validate_split(original_dataset, split_datasets):
    total_length = sum(len(split_dataset) for split_dataset in split_datasets.values())
    original_length = len(original_dataset)

    assert total_length == original_length, (
        f"Validation failed: Total length of split datasets ({total_length}) does not match "
        f"original dataset length ({original_length})"
    )
    print("Validation passed: The split datasets sum up to the original dataset length.")


# Run validation
validate_split(dataset, modified_split_datasets)

# Example usage: Iterate over one of the modified datasets
instruction_example = list(modified_split_datasets.keys())[0]
print(f"Example task: {instruction_example}")
example_dataset = modified_split_datasets[instruction_example]
for data_point in example_dataset:
    pprint.pprint(data_point)
    break
