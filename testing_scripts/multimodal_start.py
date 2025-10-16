import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import matplotlib.pyplot as plt

# creating a chromadb object
chroma_client = chromadb.PersistentClient(path="./data/chroma.db")

image_loader = ImageLoader()
embedding_function = OpenCLIPEmbeddingFunction()

# create collection - vector database:
collection = chroma_client.get_or_create_collection(
    "multimodal_collection", 
    embedding_function=embedding_function,
    data_loader = image_loader
)

# adding images to the collection using add() for first time adding data to vectordb, then update() to update existing images

# collection.update(
#     ids = ["0", "1"],
#     uris = ["./images/lion.jpg", "./images/tiger.jpg"], # image paths
#     metadatas = [{"category": "animal"}, {"category": "animal"}]
# )

collection.add(
    ids = ["E23", "E25", "E33"],
    uris = ["./images/E23-2.jpg", "./images/E25-2.jpg", "./images/E33-2.jpg"], # image paths
    metadatas = [{
        "item_id": "E23",
        "category": "food",
        "item_name": "Braised Fried Tofu with Greens"
    }, 
    {
        "item_id": "E25",
        "category": "food",
        "item_name": "Sauteed Assorted Vegetables"
    },
    {
        "item_id": "E33",
        "category": "food",
        "item_name": "Kung Pao Tofu"
    }]
)

# print(collection.count()) # 2

# function to print results of a query - results is a dict: {ids, dist, data, ...}; Each item is a 2d list

def print_query_result(query_list: list, query_result: dict) -> None:
    result_count = len(query_result["ids"][0])

    for i in range(len(query_list)):
        print(f"Results for query: {query_list[i]}")

        for j in range(result_count):
            id = query_result["ids"][i][j]
            distance = query_result["distances"][i][j]
            data = query_result["data"][i][j]
            document = query_result["documents"][i][j]
            metadata = query_result["metadatas"][i][j]
            uri = query_result["uris"][i][j]

            print(
                f"id: {id}, distance: {distance}, metadata: {metadata}, document: {document}, "
            )

            print(f"data: {uri}")
            plt.imshow(data)
            plt.axis("off")
            plt.show()
        print()
        print()

query_list = ["food with carrots", "food without tofu"]

# querying the vector database:
query_results = collection.query(
    query_texts=query_list,
    n_results = 2,
    include = ["documents", "distances", "metadatas", "data", "uris"]
)

print_query_result(query_list, query_results)