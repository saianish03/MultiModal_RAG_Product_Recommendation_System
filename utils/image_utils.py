import os
import urllib
from PIL import Image
import matplotlib.pyplot as plt
import datasets
from tqdm import trange


def is_saved_images(
        dataset_folder: str,
        num_images: int
    ):
    if len(os.listdir(dataset_folder)) == num_images:
        return True
    return False


def show_image_from_uri(uri: str):
    if isinstance(uri, str):
        image = None
        with urllib.request.urlopen(uri) as url:
            image = Image.open(url)
        return image
    else:
        raise ValueError(f"Not a valid image link! {uri}")


def show_image_from_path(path: str):
    if isinstance(path, str):
        image = Image.open(path)
        plt.imshow(image)
        plt.axis("off")
        plt.show()
    else:
        raise ValueError(f"Not a valid image path! {path}")


def open_example_image(
        data: datasets.DatasetDict | datasets.Dataset | datasets.IterableDataset | datasets.IterableDatasetDict, 
        idx: int, 
        execute: bool
    ):
    if not execute:
        return
    print(data.num_rows)
    product_image = data["train"][idx]["image"]
    img = show_image_from_uri(uri = product_image)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def save_all_images(
        dataset: datasets.DatasetDict | datasets.Dataset | datasets.IterableDataset | datasets.IterableDatasetDict, 
        dataset_folder: str,
        num_images: int = 1000
    ):
    # check if dataset_folder exists else make dir
    os.makedirs(dataset_folder, exist_ok = True)

    # check if num_images already saved in dataset_folder:
    if is_saved_images(dataset_folder, num_images):
        print(f"{dataset_folder} dir already contains first {num_images} images")
        return

    for i in trange(num_images, desc="Saving images"):
        uri = dataset["train"][i]["image"]
        prod_id = dataset["train"][i]["parent_asin"]
        image = show_image_from_uri(uri)
        image.save(os.path.join(dataset_folder, f"image_{prod_id}.png"))
    print(f"Saved first {num_images} to folder: {dataset_folder}") 
