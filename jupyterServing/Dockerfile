FROM jupyter/base-notebook:python-3.9.7

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Copy requirements
COPY requirements.txt /tmp/

# Install packages
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r /tmp/requirements.txt
