from chromadb.types import Collection
from dotenv import load_dotenv
from tqdm import trange
from typing import Union, List
from datetime import datetime
from datasets import Dataset, load_dataset


import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader


from utils.text_preprocess import preprocess_dataset
from utils.image_utils import (is_saved_images, 
                            show_image_from_uri, 
                            show_image_from_path, 
                            open_example_image, 
                            save_all_images) 
from utils.data_utils import get_file_names, get_metadata


# folder where images are present
DATASET_FOLDER = "./products_dataset/AMAZON-Products-2023"

# vector db of amazon dataset
PATH = "./data/products_base.db"


def get_or_create_vector_db(path: str) -> Collection:
    # setup chromaDB to create embeddings
    image_loader = ImageLoader()
    embedding_function = OpenCLIPEmbeddingFunction()
    chroma_client = chromadb.PersistentClient(path=path)

    product_collection = chroma_client.get_or_create_collection(
        "base_products_collection",
        embedding_function=embedding_function,
        data_loader = image_loader,
        metadata = {
            "description": "A vector database storing amazon product images and other metadata like product name, description, category, price, average rating, number of ratings, store name, date first available", 
            "createdAt": str(datetime.now())
        }
    )

    return product_collection


def add_images_metadata_to_vectordb(
        dataset: Dataset,
        collection: Collection,
        path: str, 
        dataset_folder: str
    ):
    ids, uris = get_file_names(dataset_folder)
    metadata_dict = get_metadata(dataset)

    metadata = [metadata_dict[asin] for asin in ids]

    collection.add(
        ids = ids, 
        uris = uris,
        metadatas = metadata
    )
    print(f"{collection.count()} images and their metadata added to Vector Database located at {path}")
    return collection


def query_db(
        query: Union[str, List[str]], 
        collection: Collection, 
        n_results: int = 5
    ):
    print(f"Querying the database for: {query}")
    # Ensure query_texts is always a list of strings
    if isinstance(query, str):
        query_texts = [query]
    else:
        query_texts = query
    result = collection.query(
        query_texts=query_texts, n_results=n_results, include=["uris", "distances", "metadatas"]
    )
    return result


def print_results(results):
    for idx, uri in enumerate(results["uris"][0]):
        print("ID: ", results["uris"][0][idx])
        print("Distance: ", results["distances"][0][idx])
        print(f"Path: {uri}")
        print(f"Title: ", results["metadatas"][0][idx]["title"])
        print(f"Description: ", results["metadatas"][0][idx]["description"])
        print(f"Rating: ", results["metadatas"][0][idx]["average_rating"])
        print(f"Price: ", results["metadatas"][0][idx]["price"])
        show_image_from_path(uri)
        print()


def get_collection(product_dataset_name: str = "Amazon-2023"):
    # create vector db:
    product_collection = get_or_create_vector_db(PATH)
    
    return product_collection


def load_data_into_collection(product_dataset_name: str = "Amazon-2023", show_image: bool = False):
    raw_data = load_dataset("milistu/AMAZON-Products-2023")

    # clean the dataset
    cleaned_data = preprocess_dataset(dataset = raw_data["train"])

    # show an example image:
    open_example_image(data = cleaned_data, idx = 100, execute=show_image)

    # create vector db:
    product_collection = get_or_create_vector_db(PATH)

    # add images and metadata to vector db:
    add_images_metadata_to_vectordb(dataset = cleaned_data, collection = product_collection, path = PATH, dataset_folder = DATASET_FOLDER)

    return product_collection