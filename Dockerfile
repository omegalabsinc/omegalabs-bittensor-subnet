FROM --platform=linux/amd64 python@sha256:370c586a6ffc8c619e6d652f81c094b34b14b8f2fb9251f092de23f16e299b78

# Install software-properties-common to add repositories
RUN apt-get -y update && apt-get install -y \
    git libsndfile1 build-essential ffmpeg libpq-dev \
    pkg-config libmysqlclient-dev curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app/

# Install python requirements
COPY ./requirements.txt ./requirements.txt
COPY ./requirements_api.txt ./requirements_api.txt

RUN python -m pip install --upgrade pip setuptools wheel uv
RUN python -m uv pip install --no-cache-dir -r requirements_api.txt --prerelease=allow

COPY . .
RUN python -m pip install -e . --no-cache-dir

# Runtime env variables
ENV PORT=8002

EXPOSE 8002

CMD ["python", "validator-api/app.py"]
