from search.search_query import (get_collection,
                                load_data_into_collection, 
                                query_db,
                                print_results,
                                )
from utils.image_utils import show_image_from_path
from utils.langchain import (format_prompt_inputs,
                            get_vision_model,
                            get_image_prompt_template,
                            )
from langchain_core.output_parsers import StrOutputParser

import streamlit as st

import warnings
warnings.filterwarnings("ignore")

st.title("Multi-Modal RAG Product Search and Recommendation System")


@st.cache_resource
def get_cached_collection():
    return get_collection()


@st.cache_resource
def get_cached_vision_model(model_name="gpt-4o", temperature=0.0):
    return get_vision_model(model_name, temperature)


if __name__ == "__main__":
    # load data and metadata into vectorDB and get collection
    # product_collection = load_data_into_collection()

    # query and get results:
    # query = "Advanced DJ Controller"
    # results = query_db(query = query, collection = product_collection, n_results = 5)

    # # print results
    # print_results(results)

    # Setting up the RAG Flow:
    # 1. user submits a query (question, query, etc)
    # 2. query is first sent to multimodal database (retrieval function first)
    # 3. these images are passed to along with prompt to a vision model where it will use image context and answer
    # the prompt as a final output


    print("Welcome to Multimodal RAG Product Search!")
    
    # get collection
    product_collection = get_cached_collection()

    # load the vision model 
    vision_model = get_cached_vision_model(model_name = "gpt-4o", temperature = 0.0)
    
    # output parser
    parser = StrOutputParser()

    # get the prompt template
    image_prompt = get_image_prompt_template()

    # create the chain:
    vision_chain = image_prompt | vision_model | parser

    # # enter the query
    # print("Please enter your query about a product.")
    # query = input("Enter your query: \n")

    query = st.text_input("Enter your query (ex: 'advanced dj controller')")

    # display input query:
    if query:
        st.write(f"Your query: {query}")
    
        # fetch the images from VectorDB based on text query
        with st.spinner("Retrieving images..."):
            results = query_db(query = query, collection = product_collection, n_results = 2)
        
        # display the retrieved images
        st.write("Here are the top products based on your query:")
        for i in range(len(results["uris"][0])):
            st.image(image = results["uris"][0][i], caption = results["metadatas"][0][i]["title"])

        # format the prompt_inputs according to the results and query
        # invoke the chain with the prompt input
        with st.spinner("Generating suggestions..."):
            prompt_input = format_prompt_inputs(data = results, user_query = query)
            response = vision_chain.invoke(prompt_input)

        # print the response:
        st.markdown("\n Here is some information about the product query: \n")
        st.write(response)
        
        # display the location of the image files
        # print("\n Images URI: \n")
        # print("Image 1: ", results["uris"][0][0])
        # print("Image 2: ", results["uris"][0][1])
