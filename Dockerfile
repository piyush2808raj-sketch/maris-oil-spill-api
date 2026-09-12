FROM python:3.10

RUN useradd -m -u 1000 user

WORKDIR /app

COPY --chown=user requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . /app

USER user

ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]