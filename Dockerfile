FROM python:3.10.12

# Set the working directory within the Docker image
WORKDIR /app

# Copy the Streamlit app code into the image
COPY app2.py .

# Install the required dependencies
RUN pip install nltk numpy scikit-learn scipy streamlit pillow toml

# Run nltk.download() 
RUN python -m nltk.downloader punkt

# Set the working directory within the image
WORKDIR /app

# Run the Streamlit app
CMD ["streamlit", "run", "app2.py"]


