from multiprocessing import Value
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import base64

from dotenv import load_dotenv

load_dotenv()

VISION_MODELS = [
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o4-mini",
    "gpt-5",
    "gpt-5-chat-latest",
    "gpt-4o",
    "4o",
    "4.1",
    "4.5",
    "4o-mini",
    "o1",
    "o1-pro",
    "o3",
    "computer-use-preview"
]


def format_prompt_inputs(data, user_query):
    print("Formatting prompt inputs...")
    inputs = {}

    # add user query to dict
    inputs["user_query"] = user_query

    # get first two image paths
    image_path_1 = data["uris"][0][0]
    image_path_2 = data["uris"][0][1]

    # encode images to base64
    with open(image_path_1, "rb") as image_file:
        image_data_1 = image_file.read()
    inputs["image_data_1"] = base64.b64encode(image_data_1).decode("utf-8")

    with open(image_path_2, "rb") as image_file:
        image_data_2 = image_file.read()
    inputs["image_data_2"] = base64.b64encode(image_data_2).decode("utf-8")


    inputs["title_1"] = data["metadatas"][0][0]["title"]
    inputs["description_1"] = data["metadatas"][0][0]["description"]
    inputs["price_1"] = data["metadatas"][0][0]["price"]

    inputs["title_2"] = data["metadatas"][0][1]["title"]
    inputs["description_2"] = data["metadatas"][0][1]["description"]
    inputs["price_2"] = data["metadatas"][0][1]["price"]

    print("Prompt inputs formatted successfully...")
    return inputs


def get_vision_model(model_name = 'gpt-4o', temperature = 0.0, **kwargs):
    """
    Load the right OpenAI Vision supported models and return LangChain's ChatOpenAI object.
    Supports only OpenAI Models for now.
    """
    
    if model_name not in VISION_MODELS:
        raise ValueError("Wrong OpenAI vision model name. Choose the right one!")
    
    vision_model = ChatOpenAI(
        model = model_name,
        temperature = temperature,
        **kwargs
    )

    return vision_model


def get_image_prompt_template(system_prompt=None, assistant_prompt=None):

    image_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    system_prompt if system_prompt else 
                    "You are a knowledgeable and helpful product assistant that provides detailed information about Amazon products."
                    "When answering the user's question, always use the given image context and metadata."
                    "Be sure to include important details like **title, features, use-cases, and especially the price** of the product in your response."
                    "If multiple products are shown, compare their features and prices when relevant."
                )
                +
                (
                    assistant_prompt if assistant_prompt else 
                    "Maintain a more conversational tone, don't make too many lists or bullet points. Use markdown formatting for highlights, emphasis, and structure."
                )
            ),
            (
                "user",
                [
                    {"type": "text", "text": "{user_query}"},
                    
                    {"type": "text", "text": "**Product 1 Details**\n"
                                            "- **Title**: {title_1}\n"
                                            "- **Description**: {description_1}\n"
                                            "- **Price**: {price_1}"},
                    {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_data_1}"},

                    {"type": "text", "text": "**Product 2 Details**\n"
                                            "- **Title**: {title_2}\n"
                                            "- **Description**: {description_2}\n"
                                            "- **Price**: {price_2}"},
                    {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_data_2}"},
                ],
            ),
        ]
    )


    # image_prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             system_prompt if system_prompt else "You are a knowledge base assistant, focusing on providing helpful information about all kinds of products from Amazon. Answer the user's question using the given image context with direct references to the parts of images provided."
    #             +
    #             assistant_prompt if assistant_prompt else "Maintain a more conversational tone, don't make too many lists or bullet points. Use markdown formatting for highlights, emphasis, and structure."
    #         ),
    #         (
    #             "user",
    #             [
    #                 {
    #                     "type": "text",
    #                     "text": "what are the best features of this product and what are some good ways to use this product?"
    #                 },
    #                 {
    #                     "type": "image_url",
    #                     "image_url": "data:image/jpeg;base64,{image_data_1}",
    #                 },
    #                 {
    #                     "type": "image_url",
    #                     "image_url": "data:image/jpeg;base64,{image_data_2}",
    #                 }
    #             ],
    #         ),
    #     ]
    # )
    
    return image_prompt