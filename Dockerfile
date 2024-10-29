FROM python:3.10-slim-bookworm

RUN apt-get update \
    && apt-get install -y \
    build-essential \
    protobuf-compiler \
    git \
    wget \
    && apt-get clean && apt-get autoremove

ENV PYTHONUNBUFFERED 1

COPY poetry.lock pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false    

ARG DEV=false
RUN if [ "$DEV" = "true" ] ; then poetry install --with dev --no-root ; else poetry install --only main --no-root ; fi

ENV PYTHONPATH "${PYTHONPATH}:/app" 

COPY . /app

EXPOSE 8080

CMD ["fastapi", "run", "app/app.py", "--host", "0.0.0.0", "--port", "8080"]