# <img width="202" alt="logo" src="https://github.com/arthur-samuel-thinkai/dayem/assets/170200420/0b5097f4-cd82-4ca9-b1a3-8d19bfb48d10">
### Sustainable Development Assistant


## Table Of Contents
- [Who am I?](#who-am-i)Conclusion
- [Why I am here?](#why-i-am-here)
- [How Do I work?](#how-do-i-work)
- [Research and development progress & Document Retrieval and QA System](#Research-and-development-progress-&-Document-Retrieval-and-QA-System)
- [Series Forecasting Approaches for Weather Prediction](#Series-Forecasting-Approaches-for-Weather-Prediction)
- [Conclusion](#Conclusion)
  


# Who am I?       
Hello, I am DAYEM.

I am an innovative generative AI application designed to forecast the weather in Morocco using advanced meteorological data. I utilize Machine Learning and Artificial Intelligence to provide precise weather predictions, which are integrated with a large language model (LLM) to offer actionable recommendations based on the forecasted weather conditions.

For instance, users can ask me whether installing solar panels in a specific area would be advisable based on the expected weather patterns. My LLM will analyze the weather data and deliver tailored advice. Additionally, users can seek customized insights for various projects or inquiries, ensuring they receive comprehensive and contextually relevant guidance.

My goal is to be a powerful tool that combines accurate weather forecasting with intelligent decision-making support, enhancing the ability of individuals and businesses to make informed choices in Morocco.

# Why I am here?
I was created by three data enthusiasts who aim to use AI for the greater good. Sustainable development in Morocco is a crucial initiative, so here I am, THE GREAT DAYEM, ready to play a significant role in supporting it.

# How Do I work?

![WhatsApp Image 2024-05-19 à 02 46 44_88ad2d6e](https://github.com/arthur-samuel-thinkai/dayem/assets/170200420/be1bc58e-0126-4773-b25d-7c6064f1cc1a)

# 
# Research and development progress & Document Retrieval and QA System

This project implements a document retrieval and question-answering (QA) system using PDF documents. The system evolves through several stages, progressively improving its capabilities. Below is a detailed description of the project's evolution, including the choices made and their justifications.

## Early Stage: Initial Setup and Basic Implementation

### Goals
- Set up a basic pipeline to process and split PDF documents.
- Implement a simple document retrieval system.
- Begin with foundational components and libraries.

### Steps and Justifications
1. **Library Installation and Setup:**
   - Installed essential libraries:
     ```bash
     !pip install transformers sentence-transformers langchain torch faiss-cpu numpy
     !pip install langchain_community
     !pip install pypdf
     ```
   - **Justification:** These libraries provide a robust foundation for handling NLP tasks, vector embeddings, and PDF processing.

2. **Document Loading and Splitting:**
   - Loaded PDF files from a directory using `PyPDFDirectoryLoader`.
   - Implemented `RecursiveCharacterTextSplitter` to split documents into manageable chunks.
   - **Justification:** Splitting documents helps manage large text files more efficiently and allows for better retrieval performance.

3. **Initial Embedding Generation:**
   - Chose `sentence-transformers/all-mpnet-base-v2` for generating embeddings.
   - **Justification:** This model provides high-quality embeddings that capture semantic meaning well.

4. **Vector Store Setup:**
   - Utilized `FAISS` to create a vector store from the document embeddings.
   - **Justification:** FAISS is a powerful tool for similarity search and nearest neighbor retrieval.

### Challenges
- Handling large documents efficiently.
- Ensuring the quality of the document splitting process.

### Outcomes
- Successfully loaded and split documents.
- Generated initial embeddings and stored them in a vector store.

## Mid Stage: Enhancing Embeddings and Retrieval

### Goals
- Improve the quality and speed of embeddings.
- Optimize the retrieval process.

### Steps and Justifications
1. **Embedding Model Refinement:**
   - Continued using `sentence-transformers/all-mpnet-base-v2` but experimented with `sentence-transformers/all-MiniLM-l6-v2` for faster performance.
   - **Justification:** While `all-mpnet-base-v2` provides high-quality embeddings, `all-MiniLM-l6-v2` offers a balance between speed and performance, suitable for quick experimentation.

2. **Document Embedding and Similarity Search:**
   - Generated embeddings for split documents and tested similarity search with various queries.
   - **Justification:** Validating the embedding quality and retrieval performance ensures that the system can handle real-world queries effectively.

3. **CUDA and GPU Utilization:**
   - Checked for CUDA availability and set up environment variables for GPU usage.
   - **Justification:** Leveraging GPU accelerates embedding generation and model inference, improving overall performance.

### Challenges
- Balancing embedding quality and processing speed.
- Efficiently utilizing available computational resources.

### Outcomes
- Enhanced embedding generation with GPU acceleration.
- Improved retrieval performance with optimized embeddings.

## Final Stage: Implementing Advanced LLM and RetrievalQA

### Goals
- Integrate an advanced language model for better query understanding and response generation.
- Implement a sophisticated RetrievalQA system with contextual prompts.

### Steps and Justifications
1. **Advanced LLM Integration:**
   - Chose `gradientai/Llama-3-8B-Instruct-Gradient-1048k` for the language model.
   - **Justification:** This model provides strong performance for instruction-based tasks, making it suitable for generating detailed and context-aware responses.

2. **RetrievalQA System Setup:**
   - Created a `RetrievalQA` instance using the advanced LLM and FAISS retriever.
   - Designed a custom prompt template incorporating temperature, humidity, and precipitation information.
   - **Justification:** A sophisticated QA system with contextual prompts ensures accurate and relevant answers tailored to specific user queries.

3. **Performance Optimization:**
   - Fine-tuned the prompt template and retrieval parameters to balance response quality and processing time.
   - **Justification:** Fine-tuning these parameters helps achieve a responsive and accurate system, critical for real-time applications.

### Challenges
- Ensuring the LLM generates contextually accurate and useful responses.
- Balancing the complexity of the prompt with response quality.

### Outcomes
- Successfully integrated an advanced LLM with a customized RetrievalQA system.
- Achieved improved response quality and relevance for user queries.

## Evolution Summary and Justification

- **Early Stage:** Focused on setting up the foundational components with a basic retrieval system. Chose `sentence-transformers/all-mpnet-base-v2` for its strong embedding quality and FAISS for efficient similarity search.
- **Mid Stage:** Enhanced the system's performance by experimenting with faster embedding models and utilizing GPU acceleration. Continued refining the retrieval process to ensure accuracy and efficiency.
- **Final Stage:** Integrated an advanced language model to handle complex queries and implemented a sophisticated RetrievalQA system with contextual prompts. Fine-tuned the system to balance response quality and performance.

By progressively enhancing the components and justifying each choice based on performance and requirements, the project evolved from a basic retrieval system to a sophisticated QA system capable of delivering accurate and contextually relevant answers.


# Series Forecasting Approaches for Weather Prediction

In this project, we explored three different approaches for weather prediction using time series data: Prophet, TSMixer, and ARIMA. Below is a detailed explanation of each approach, including the results obtained and the reasoning behind our final choice.

## Prophet Approach

### Overview

Prophet is a forecasting tool developed by Facebook that is particularly good for handling time series data with strong seasonal effects and missing data. It decomposes time series into trend, seasonality, and holiday effects.

### Steps Taken

- Loaded and preprocessed the weather data, including temperature, humidity, and precipitation.
- Configured separate Prophet models for each variable.
- Trained the models on the training set and made predictions on the test set.
- Evaluated the models using Mean Squared Error (MSE).

### Results

- Prophet managed to capture the overall trend and seasonality in the data reasonably well.
- However, it struggled with the higher frequency variations and noise present in the weather data, leading to moderate accuracy.
- ![image](https://github.com/arthur-samuel-thinkai/dayem/assets/170200420/3e766a20-aebd-4a2a-ab18-9451c6d4fc82)
- ![image](https://github.com/arthur-samuel-thinkai/dayem/assets/170200420/fdf62ab4-03b5-4c7a-8be1-c3bfe3b93424)



### Conclusion

Prophet's strengths in handling missing data and seasonal trends were evident, but its performance was not satisfactory enough for our needs due to its handling of high-frequency variations.

## TSMixer Approach

### Overview

TSMixer is a deep learning-based approach specifically designed for time series forecasting. It uses a combination of temporal convolutional layers and mixer layers to capture both local and global dependencies in the data.

### Steps Taken

- Preprocessed the data and split it into sequences suitable for TSMixer.
- Configured the TSMixer model with appropriate parameters.
- Trained the model on the training set using a Mean Squared Error loss function.
- Evaluated the model's predictions against the test set.

### Results

- TSMixer showed promise in capturing complex patterns in the data, thanks to its deep learning architecture.
- Despite this, the model was prone to overfitting, and the computational cost was high.
- The predictions were inconsistent, particularly for longer-term forecasts.
- ![image](https://github.com/arthur-samuel-thinkai/dayem/assets/170200420/853c9492-9e3c-4b27-9d01-8be09ffca5a9)
- ![image](https://github.com/arthur-samuel-thinkai/dayem/assets/170200420/f4538b1f-a0be-4381-955f-9660d6f61e0f)



### Conclusion

While TSMixer demonstrated the ability to model intricate patterns in the data, its tendency to overfit and the computational intensity made it less practical for our application.

## ARIMA Approach

### Overview

ARIMA (AutoRegressive Integrated Moving Average) is a classical statistical method for time series forecasting. It is well-suited for univariate time series data and can effectively model the data's temporal dependencies.

### Steps Taken

- Loaded and preprocessed the weather data for temperature and precipitation.
- Configured and fitted separate ARIMA models for each variable.
- Made forecasts and evaluated the model's performance using visual plots and statistical metrics.

### Results

- ARIMA provided accurate short-term forecasts for both temperature and precipitation.
- The model's simplicity and effectiveness in capturing the autocorrelation in the data resulted in lower prediction errors compared to Prophet and TSMixer.
- The visual plots showed that ARIMA's forecasts closely followed the actual test data.
- ![image](https://github.com/arthur-samuel-thinkai/dayem/assets/170200420/aa9fd330-0e70-49e7-a279-8c9ee832c20d)

### Conclusion

ARIMA outperformed both Prophet and TSMixer in terms of accuracy and reliability. Its straightforward implementation and lower computational cost made it the most practical choice for our project.





