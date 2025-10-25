# MultiModal RAG Product Recommendation System

A sophisticated retrieval-augmented generation (RAG) system that combines computer vision and natural language processing to provide intelligent product recommendations from Amazon's product catalog. The system leverages multimodal embeddings to understand both textual queries and product images, enabling accurate semantic search and contextual product analysis.

## System Architecture

The system implements a two-stage RAG pipeline that processes user queries through semantic vector search followed by VLM analysis. The architecture consists of three primary components:

1. **Vector Database Layer**: ChromaDB with OpenCLIP embeddings for multimodal similarity search
2. **Data Processing Pipeline**: Automated text normalization and image preprocessing utilities
3. **Language Model Integration**: OpenAI's vision models for contextual product analysis

## Key Features

- **Multimodal Search**: Combines text and image embeddings for comprehensive product retrieval
- **Semantic Understanding**: Advanced text preprocessing with Unicode normalization and homoglyph handling
- **Vision-Language Analysis**: VLM integration for detailed product feature extraction and comparison
- **Streamlit Interface**: Interactive web application for real-time product search and recommendations
- **Scalable Architecture**: Modular design supporting dataset expansion and model customization

## Technical Implementation

The system processes Amazon product data through a sophisticated preprocessing pipeline that normalizes text fields, cleans metadata, and generates consistent product identifiers. Vector embeddings are created using OpenCLIP's multimodal capabilities, enabling semantic search across both product descriptions and visual features. The retrieval component identifies relevant products based on query similarity, while the generation component uses vision-language models to provide detailed product analysis and recommendations.

## Prerequisites

- Python 3.8 or higher
- OpenAI API key with GPT-4o access
- Sufficient disk space for product images and vector database storage

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MultiModal_RAG_Product_Recommendation_System
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage


The application will open in your default web browser. Enter product queries in natural language to search and receive detailed product recommendations.


### Data Loading

To populate the vector database with product data:
```bash
python -c "from search.search_query import load_data_into_collection; load_data_into_collection()"
```

### Command Line Testing

For development and testing purposes, use the testing scripts:
```bash
python testing_scripts/multimodal_final.py
```

### Running the Streamlit Application

Launch the interactive web interface:
```bash
streamlit run app.py
```

## Project Structure

```
MultiModal_RAG_Product_Recommendation_System/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── search/
│   ├── __init__.py
│   └── search_query.py            # Vector database operations
├── utils/
│   ├── __init__.py
│   ├── data_utils.py              # Data processing utilities
│   ├── image_utils.py             # Image handling functions
│   ├── langchain.py               # Language model integration
│   ├── text_preprocess.py         # Text normalization pipeline
│   └── homoglyphs.py              # Unicode normalization
├── testing_scripts/
│   ├── multimodal_final.py        # Streamlit testing interface
│   └── multimodal_start.py        # Command line testing
└── data/                          # Vector database storage
```

## Configuration

The system supports various configuration options through environment variables and function parameters:

- **Model Selection**: Choose from supported OpenAI vision models in `utils/langchain.py`
- **Database Path**: Configure vector database location in `search/search_query.py`
- **Dataset Size**: Adjust the number of products processed via the `num_images` parameter
- **Search Results**: Modify the number of retrieved products using the `n_results` parameter

## Performance Considerations

The system is optimized for efficient retrieval and generation:

- **Caching**: Streamlit resources are cached to minimize redundant computations
- **Batch Processing**: Images and metadata are processed in optimized batches
- **Memory Management**: Large datasets are handled through streaming and pagination
- **API Optimization**: OpenAI API calls are structured for minimal latency
