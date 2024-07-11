FROM python:3.10.14-slim

# Set the working directory in the container
WORKDIR /app

COPY requirements.txt /requirements.txt

# Install gcc, python3-dev, libgl1-mesa-glx, and libglib2.0-0
# RUN apt-get update && \
#     apt-get install -y gcc python3-dev libgl1-mesa-glx libglib2.0-0
# RUN pip install -e .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY packages /packages
COPY models /models
COPY preprocessor /preprocessor
COPY ml_logic /ml_logic
COPY setup.py /setup.py

#Run container locally
#CMD uvicorn packages.fast_api:app --reload --host 0.0.0.0


#Run container deployed -> GCP
CMD uvicorn packages.fast_api:app --reload --host 0.0.0.0 --port $PORT

# Set the environment variable for the Streamlit app
ENV STREAMLIT_APP=interface/frontend.py

# Command to run the Streamlit application
CMD ["streamlit", "run", "interface/frontend.py"]

##docker build . -t api
##docker run -p 8080:8000 api

##gcp
## docker build . -t eu.gcr.io/$GCP_PROJECT/$DOCKER_IMAGE
## docker run -e PORT=8000 -p 8000:8000 eu.gcr.io/$GCP_PROJECT/$DOCKER_IMAGE
## docker push eu.gcr.io/$GCP_PROJECT/$DOCKER_IMAGE
