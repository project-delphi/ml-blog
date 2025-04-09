# PDF Document Classifier

This application provides a FastAPI service for classifying PDF documents using RAG (Retrieval-Augmented Generation) and LLMs. It assigns one of three ordinal labels — `bad`, `neutral`, `good` — to documents based on their contents.

## Setup

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key (if using LLM classification):

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Running the Application

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

The server will start at `http://localhost:8000`.

## API Endpoints

### 1. Train the Classifier

- **Endpoint**: `/train`
- **Method**: POST
- **Input**: 
  - `pdfs`: List of PDF files
  - `labels`: List of labels ('bad', 'neutral', 'good')
- **Output**: Status message

### 2. Predict Document Class

- **Endpoint**: `/predict`
- **Method**: POST
- **Input**: 
  - `pdf`: Single PDF file
- **Output**: Predicted class ('bad', 'neutral', 'good')

## API Documentation

Once the server is running, you can access the interactive API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
