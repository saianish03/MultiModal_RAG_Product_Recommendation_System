import os
import datasets


def get_file_names(dataset_folder: str):
    ids = []
    uris = []
    for filename in os.listdir(dataset_folder):
        if filename.endswith(".png"):
            file_path = os.path.join(dataset_folder, filename)
            id = filename.split("/")[-1].rstrip(".png").split("_")[-1]
            ids.append(id)
            uris.append(file_path)
    return ids, uris


def get_metadata(dataset: datasets.DatasetDict | datasets.Dataset | datasets.IterableDataset | datasets.IterableDatasetDict,
    num_images: int = 500):
    columns = ["parent_asin", "title", "description", "main_category", "store", "average_rating", "rating_number", "price"] #, "details"] --> ignoring details for now
    metadata = {}

    for idx, i in enumerate(dataset):
        if idx == num_images:
            break
        dict = {}
        for col in columns:
            try:
                dict[col] = i[col]
            except:
                raise ValueError(f"{col} not present in dataset in row {idx}")

        metadata[i['parent_asin']] = dict

    return metadata
