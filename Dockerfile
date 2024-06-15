FROM python:3.8

WORKDIR /app

COPY requirements.txt .

RUN apt update
RUN apt install libhdf5-dev -y
RUN export HDF5_DIR=/usr/lib/aarch64-linux-gnu/hdf5



RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

WORKDIR /app/application

CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"]