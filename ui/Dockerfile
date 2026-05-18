FROM python:3.13-slim
WORKDIR /app
COPY ./index.html ./
EXPOSE 5173
CMD ["python", "-m", "http.server", "5173", "--bind", "0.0.0.0"]
