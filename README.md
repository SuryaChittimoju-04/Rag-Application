# RAG Application On HealthCare

This repository contains a Flask-based backend application designed to facilitate healthcare-related tasks using Hugging Face API models. The application allows users to upload data, which is then stored in a vector database (QDrant) using an embedding model for generating embeddings. These embeddings can later be retrieved by a retriever component.

## Features

- **Data Upload**: Users can upload healthcare-related data, such as medical records, images, or text documents.

- **Embedding Generation**: Uploaded data is processed using an embedding model to generate embeddings, which are then stored in QDrant for efficient retrieval.

- **Embedding Retrieval**: Users can query the stored embeddings to retrieve relevant information or perform similarity searches.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gbnsolutions22/RAGApplicationOnHealthCare.git

2. Navigate to the project directory:
    ```bash
    cd RAGApplicationOnHealthCare
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
## Configuration
Before running the application, you need to configure the following:

1. Hugging Face API Token: Obtain a Hugging Face API token and set it as an environment variable named HUGGINGFACE_TOKEN.
2. QDrant Configuration: Configure the connection details for your QDrant instance in the config.py file.
## Usage
1. Start the Flask application:
    ```bash
    python app.py
2. Access the application in your web browser at http://localhost:5000.
3. Upload your healthcare data and perform embedding generation and retrieval as needed.
## Deployment
The backend application can be deployed as a Docker image for easier deployment and scalability:

1. Build the Docker image:
    ```bash
    docker build -t rag-healthcare-app .
2. Run the Docker container:
    ```bash
    docker run -p 5000:5000 rag-healthcare-app
## Contributing
Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (git checkout -b feature/fooBar).
3. Commit your changes (git commit -am 'Add some fooBar').
4. Push to the branch (git push origin feature/fooBar).
5. Create a new Pull Request.
## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- [Hugging Face](https://huggingface.co) Face for providing state-of-the-art NLP models and APIs.
- [QDrant](https://qdrant.tech) for the vector database used for efficient similarity searches.
Feel free to use this template for your project. Let me know if you need further assistance!
