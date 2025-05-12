# ChXpert-Chest-X-rays-Prediction

<!-- 
Discuss: Value proposition: Your will propose a machine learning system that can be 
used in an existing business or service. (You should not propose a system in which 
a new business or service would be developed around the machine learning system.) 
Describe the value proposition for the machine learning system. What’s the (non-ML) 
status quo used in the business or service? What business metric are you going to be 
judged on? (Note that the “service” does not have to be for general users; you can 
propose a system for a science problem, for example.)
-->
Value Proposition
We propose to develop an online machine learning system for automated classification of chest X-rays to detect the presence of any potential abnormality from a list of 14 pathologies, such as cardiomegaly, edema, pneumonia, and pleural effusion. The goal is to integrate this tool into existing radiology workflows in hospitals and diagnostic labs, where it can assist radiologists by prioritizing scans likely to contain critical findings.

In the current non-ML status quo, all X-rays are manually reviewed by radiologists, regardless of their diagnostic complexity or urgency. This creates a bottleneck, especially in high-volume clinical settings, leading to delays in the diagnosis and treatment of patients with serious conditions.

Our system addresses this issue by automatically screening all incoming chest X-rays and assigning a priority score based on the model's confidence in the presence of any abnormality. Images flagged as high-risk (potentially abnormal) will be placed at the top of the radiologist's review queue, enabling faster intervention for patients who need it most. Importantly, scans predicted as benign are not excluded from human review, but are instead deprioritized and reviewed with relative leisure. This ensures that the human-in-the-loop workflow is maintained while leveraging automation for triage.

To ensure long-term adaptability and performance, we will develop an end-to-end, cloud-native retraining pipeline. In this system, once a radiologist has reviewed a scan, their final diagnosis (the true label) is stored and automatically fed back into the system. This labeled data becomes part of the continual learning pipeline, allowing the model to retrain periodically on the most recent, high-quality labeled examples. This process enhances the system’s sensitivity (true positive rate) and overall robustness over time by correcting any systematic biases or emerging distribution shifts.

Because this system operates in a medical decision support context, model performance must meet strict clinical standards. The key concern is minimizing false negatives — cases where the model incorrectly classifies an abnormal scan as normal — as these can lead to missed diagnoses. Therefore, the model will be evaluated primarily using Area Under the ROC Curve (AUROC), which reflects the model's ability to rank abnormal cases ahead of normal ones. In addition to AUROC, we will monitor other classification metrics such as sensitivity (recall), specificity, precision, and F1 score, with a particular emphasis on high recall to ensure minimal missed detections.

The proposed system can be valuable in multiple real-world scenarios:

Hospitals and diagnostic centers, where radiologist bandwidth is limited.

Teleradiology services, where scans from multiple locations are reviewed remotely.

Health insurance companies, where initial ML-based assessments could help streamline the process of verifying diagnoses before approving claims, thereby reducing claim processing time.

While the deployment of such a system raises important ethical and regulatory considerations — especially regarding trust, liability, and transparency — these challenges are acknowledged and are beyond the current project’s technical scope.


### Contributors

<!-- Table of contributors and their roles. 
First row: define responsibilities that are shared by the team. 
Then, each row after that is: name of contributor, their role, and in the third column, 
you will link to their contributions. If your project involves multiple repos, you will 
link to their contributions in all repos here. -->

| Name                            | Responsible for				 | Link to their commits in this repo |
|---------------------------------|----------------------------------------------|------------------------------------|
| Aditya Gouroju                  | Model training and training infrastructure   |       https://github.com/sid150/ChXpert-Chest-X-rays-Prediction/commits/main/?author=agouroju                             |
| Ruchit Jathania                 | Model Serving/Monitoring			 |     https://github.com/sid150/ChXpert-Chest-X-rays-Prediction/commits/main/?author=RuchitJathania                               |
| Sidharth Jain                   | Data Pipeline 				 |      https://github.com/sid150/ChXpert-Chest-X-rays-Prediction/commits/main/?author=sid150                              |



### System diagram
<img width="1503" alt="Image" src="https://github.com/user-attachments/assets/c5327eb2-3474-4a60-a473-2e092ac14af6" />
<!-- Overall digram of system. Doesn't need polish, does need to show all the pieces. 
Must include: all the hardware, all the containers/software platforms, all the models, 
all the data. -->

### Summary of Outside Materials

| Name                         | How it was created                                                                                                                                   | Conditions of use                                                                                                     |
|------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| **CheXpert**                 | Created by Stanford ML Group using 224,316 chest radiographs from 65,240 patients at Stanford Hospital. Labels were extracted using NLP from reports. [Paper](https://arxiv.org/abs/1901.07031) | Openly available for **research purposes only**. Requires acceptance of [data use agreement](https://stanfordmlgroup.github.io/competitions/chexpert/) |
| **DenseNet121 (ImageNet)**   | Trained on ImageNet-1k dataset by the original DenseNet authors. Available in `torchvision.models`. [Paper](https://arxiv.org/abs/1608.06993)        | Licensed under [BSD 3-Clause License](https://github.com/pytorch/vision/blob/main/LICENSE); free for research and commercial use |
| **ViT-Large (ImageNet-21k)** | Pretrained by Google Research on ImageNet-21k (14M images), optionally fine-tuned on ImageNet-1k. Available via Hugging Face. [Paper](https://arxiv.org/abs/2010.11929) | Licensed under [Apache 2.0 License](https://github.com/google-research/vision_transformer); free for research and commercial use |



### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), 
how much/when, justification. Include compute, floating IPs, persistent storage. 
The table below shows an example, it is not a recommendation. -->

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| `m1.xxlarge` VMs | 3 for entire project duration   | Need at least 1 server with persistent storage of greater than 450 GB due to dataset    |
| `gpu_a100`     | 4 hour block twice a week  | ViT model training               |
| Floating IPs    | 1 for entire project duration, 1 for sporadic use |                |


### Detailed design plan


####  Set Up Persistent Infrastructure

-   **Compute Resource**: Provisioned a `m1.xlarge` VM instance on `KVM@TACC` with a floating IP using the `python-chi` API.
    
-   **Security Groups**: Configured ports (`22`, `8888`, `8000`, `9000`, `9001`) for SSH, Jupyter, MLFlow, and MinIO access.
    
-   **SSH Access**: Connected securely using provided SSH keys.
    

####  Prepare Object Storage

-   **Storage Service**: OpenStack Swift-based **object store** on `CHI@TACC`.
    
-   **Container**: Created container `object-persist-project29` 
    

####   Authenticate Storage Access

-   **Credential Setup**: Generated an OpenStack application credential (`ID`, `Secret`, `User ID`).
    
-   **Rclone Configuration**:
    
    -   Config file stored at [`config/rclone.conf`]

```
[chi_tacc]
type = swift
user_id = YOUR_USER_ID
application_credential_id = APP_CRED_ID
application_credential_secret = APP_CRED_SECRET
auth = https://chi.tacc.chameleoncloud.org:5000/v3
region = CHI@TACC
```

        
    -   Verified access with `rclone lsd chi_tacc:`
        

####  Dockerized ETL Pipeline

All stages are defined in [`docker-compose-etl.yaml`](https://github.com/sid150/ChXpert-Chest-X-rays-Prediction/blob/main/docker/docker-compose.yaml) and share the volume `chexpert-data`.

**Extract Stage** – Downloads dataset  
Container downloads and unzips the CheXpert ZIP to `/data/CheXpert`.

**Transform Stage** – Organizes data  
[transform-data.py](https://github.com/sid150/ChXpert-Chest-X-rays-Prediction/blob/main/docker/docker-compose.yaml)  
Python script structures images by class/label under `/data/CheXpert/train/[label]`.

**Load Stage** – Uploads to object store  
[docker-compose.yaml](https://github.com/sid150/ChXpert-Chest-X-rays-Prediction/blob/main/docker/docker-compose.yaml)  
Pushes processed dataset to `object-persist-netID` using `rclone`.

#### Execution (Run Sequentially)


`docker compose -f docker/docker-compose-etl.yaml run extract-data
docker compose -f docker/docker-compose-etl.yaml run transform-data export RCLONE_CONTAINER=object-persist-project29
docker compose -f docker/docker-compose-etl.yaml run load-data` 

####   Persisted Dataset Usage

-   Training containers can now **pull from object storage**, avoiding repeated downloads.
## Set Up the Block Storage Volume

1.  SSH into your compute instance (e.g., `node-persist`):
    
    `ssh -i ~/.ssh/id_rsa_chameleon cc@<FLOATING_IP>` 
    
2.  Verify the block storage volume appears by running:
 
    `lsblk` 
    
3.  Partition and format the volume:

    `sudo parted -s /dev/vdb mklabel gpt
    sudo parted -s /dev/vdb mkpart primary ext4 0% 100%` 
    
4.  Verify the partition (`vdb1`) is created:
    
    `lsblk` 
    
5.  Format the partition with **ext4** filesystem:
    
    `sudo mkfs.ext4 /dev/vdb1` 
    
6.  Create a directory to mount the volume:

    `sudo mkdir -p /mnt/block
    sudo mount /dev/vdb1 /mnt/block` 
    
7.  Change ownership of the mount point:
    
    `sudo chown -R cc /mnt/block
    sudo chgrp -R cc /mnt/block` 
    
8.  Verify the mount:
    `df -h` 
    

You should see the volume mounted on `/mnt/block`.

----------

## Use Docker Volumes on Persistent Storage

Now, let’s set up **MLFlow** for experiment tracking, **PostgreSQL**, and **MinIO** using Docker Compose, with data persistence backed by the block storage volume.

### Bring Up Services with Docker Compose

1.  Run the following command to bring up the services (MLFlow, PostgreSQL, MinIO, and Jupyter):
    
    `HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4) \
    docker compose -f ~/data-persist-chi/docker/docker-compose-block.yaml up -d` 
    
2.  Retrieve the logs to get the link to the Jupyter notebook interface:
   
    
    `docker logs jupyter` 
    
3.  Open the Jupyter notebook interface by substituting the **floating IP** in the browser:
    
    `http://<FLOATING_IP>:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` 
    
4.  Also, access the MLFlow web UI at:
    
    
    `http://<FLOATING_IP>:8000` 
    

----------

## Verify Persistence After Deleting and Recreating the Instance

### Create a New Instance

1.  Create a new instance:

    `s = server.Server(f"node-persist-{username}", image_name="CC-Ubuntu24.04", flavor_name="m1.large")
    s.submit(idempotent=True)
    s.associate_floating_ip()
    s.refresh()
    s.check_connectivity()` 
    

### Attach the Block Storage Volume to the New Instance

2.  Attach the existing block storage volume to the new instance:
    `cinder_client = chi.clients.cinder()
    volume = [v for v in cinder_client.volumes.list() if v.name == 'block-persist-netID'][0]
    volume_manager = chi.nova().volumes
    volume_manager.create_server_volume(server_id=s.id, volume_id=volume.id)` 
    

### Verify Data Availability

3.  SSH into the new instance and mount the block storage volume:
    `sudo mkdir -p /mnt/block
    sudo mount /dev/vdb1 /mnt/block` 
    
4.  Verify that the data is still available:    
    `ls /mnt/block` 
    
6.  Use Docker Compose to bring the services back up:
    
    `HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4) \
    docker compose -f ~/data-persist-chi/docker/docker-compose-block.yaml up -d` 
    
7.  Open the MLFlow web UI at:
    
    `http://129.114.26.91:8000` 
    
    Confirm that previous experiment logs are present.
    

----------

## Reference: Managing Block Storage Using Python
[block_store.py](https://github.com/sid150/ChXpert-Chest-X-rays-Prediction/blob/main/block_store.py)
### List Volumes

`cinder_client = chi.clients.cinder()
cinder_client.volumes.list()` 

### Create a Block Storage Volume

`volume = cinder_client.volumes.create(name=f"block-persist-python-{username}", size=2)
volume._info` 

### Attach a Volume to a Compute Instance
`server_id = chi.server.get_server(f"node-persist-{username}").id volume_manager = chi.nova().volumes
volume_manager.create_server_volume(server_id=s.id, volume_id=volume.id)` 

### Delete the Volume
`volume_manager.delete_server_volume(server_id=s.id, volume_id=volume.id)` 
`cinder_client.volumes.delete(volume=volume)` 

---------


## ContinuousX

For this repository, **Terraform**  is used in a declarative style to provision cloud resources, while automation tools like **Ansible**, **ArgoCD**, and **helm**  are used to manage and configure software components in the infrastructure.

### Tools for IaC

1.  **Ansible**: Ansible is a powerful tool for automating the configuration of servers, installation of software, and deployment of applications. It uses simple YAML-based configuration files (known as playbooks) to automate tasks.
    
2.  **ArgoCD**: ArgoCD is a Kubernetes-native continuous delivery tool that automates the deployment of applications to Kubernetes clusters. It uses Git repositories as the source of truth for the application's desired state and automates the deployment process to ensure the current state matches the desired state defined in Git.
    
3.  **Helm**: Helm is a Kubernetes package manager that simplifies the deployment of applications on Kubernetes by defining reusable and versioned packages called charts. Helm charts allow you to automate the installation and configuration of complex Kubernetes resources.
    
4.  **python-chi**: This tool is a Python-based IaC solution that facilitates the management of infrastructure resources. It’s often used in conjunction with other tools for automating deployment and ensuring consistency.
    

### Example of Declarative IaC using Terraform

With **Terraform**, you write the infrastructure configuration as code. For example, to provision an EC2 instance in Chameleon on [main.tf](https://github.com/sid150/ChXpert-Chest-X-rays-Prediction/blob/main/continuousX/tf/kvm/main.tf)
The  `tf/kvm`  directory in our IaC repository includes the following files,

```
├── data.tf
├── main.tf
├── outputs.tf
├── provider.tf
├── variables.tf
└── versions.tf
```

By executing `terraform apply`, Terraform will read this configuration and provision the infrastructure as described. The benefit is that you can version control the configuration in Git, and automatically deploy the exact same infrastructure on demand.

## Cloud-Native Development


For this project, the model training, evaluation, prediction, and other services are containerized using **Docker**. Each service (e.g., model training, model serving, data preprocessing) is developed as a separate microservice and then packaged into a Docker container. These containers are then orchestrated by **Kubernetes**, which handles scaling and management.

### Example Dockerfile for the Model Service

Here’s an example `Dockerfile`  for containerizing the chest X-ray model:

dockerfile

`# Use a base Python image
FROM python:3.8-slim`

# Install necessary dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the model training code into the container
COPY . /app
WORKDIR /app

# Set the entry point to run the application
`CMD ["python", "app.py"]` `

This `Dockerfile`  specifies the Python version to use, installs dependencies from a `requirements.txt`  file, copies the project files into the container, and sets the entry point to run the application. The container can then be easily deployed on a cloud service or Kubernetes.

## CI/CD and Continuous Model Training

#### CI/CD Workflow for Model Retraining

The CI/CD pipeline for this project automates the entire model lifecycle can be found in the folder [workflows](https://github.com/sid150/ChXpert-Chest-X-rays-Prediction/tree/main/continuousX/k8s)

1.  **Trigger**: The pipeline is triggered when code changes are pushed to the repository, a schedule is met, or external data updates are available.
    
2.  **Retraining**: The pipeline starts by retraining the model with the latest data.
    
3.  **Evaluation**: After training, the model undergoes an offline evaluation to measure its performance (e.g., accuracy, precision).
    
4.  **Optimization**: Post-training optimizations are applied to improve the model's performance for real-time inference.
    
5.  **Testing**: The model is tested to ensure its integration with the other services and APIs.
    
6.  **Dockerization**: The trained model is packaged into a Docker container, which can be easily deployed in cloud or on-premise environments.
    
7.  **Staging Deployment**: The model is deployed in a staging environment for further testing.
    

## Staged Deployment

### What is Staged Deployment?

In this project, we define three primary deployment stages using [ArgoCD](https://github.com/sid150/ChXpert-Chest-X-rays-Prediction/tree/main/continuousX/workflows):

1.  **Staging**: The model is deployed in a non-production environment where it undergoes further testing, validation.
    
2.  **Canary**: A small subset of users is routed to the new model in a canary deployment. This allows real-world testing with limited traffic before full production release.
    
3.  **Production**: Once the model passes the canary phase, it is promoted to production, where it serves all users.
    

### Example Kubernetes Deployment for Staging

Here’s an example Kubernetes YAML configuration for deploying the model in a [**staging**  environment](https://github.com/sid150/ChXpert-Chest-X-rays-Prediction/blob/main/continuousX/k8s/staging/templates/chexpert-app.yaml):

```apiVersion: v1
kind: Service
metadata:
  name: chexpert-app
  namespace: chexpert-staging
spec:
  selector:
    app: chexpert-app
  ports:
    - port: {{ .Values.service.port }}
      targetPort: 8000
  externalIPs:
    - {{ .Values.service.externalIP }}
    
   ```

The Kubernetes configuration deploys the model with three replicas for load balancing in the staging environment.










##  Model Training

1. **SSH into your persistent CPU node** (allocate via Terraform or the Chameleon portal):

   ```bash
   ssh cc@<CPU_NODE_IP>
   cd ChXpert-Chest-X-rays-Prediction/docker
   ```
the docker file can be found at [docker/docker-mlflow-update.yaml](docker/docker-mlflow-update.yaml)


2. **Launch the MLflow stack** with MinIO and Postgres:

   ```bash
   HOST_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
   MINIO_ROOT_PASSWORD={} \
   AWS_ACCESS_KEY_ID={} \
   AWS_SECRET_ACCESS_KEY={} \
   POSTGRES_USER={} \
   POSTGRES_PASSWORD={} \
   POSTGRES_DB={} \
   docker compose -f docker-mlflow-update.yaml up -d
   ```

   * View **MLflow UI** at `http://129.114.26.91:8000`
   * Access **MinIO** at `http://129.114.26.91:9000` (credentials: can be found at [ray_work/environment.txt](ray_work/environment.txt))

3. **Reserve a GPU instance** (via Chameleon portal or Terraform). Record its IP as `<GPU_NODE_IP>`.

4. **SSH into the GPU node** and install all environment dependencies and mount the CheXpert dataset:
the [train_setup.ipynb](train_setup.ipynb) contains the cells to allocate fip and install cuda dependencies and also contains commands that should be run after ssh-ing into the gpu instance to mount object store

   ```bash
   ssh cc@<GPU_NODE_IP>
   ```

   * This creates `/mnt/object` containing the dataset.

5. **On the GPU node**, set up the Ray cluster:
the docker file is [docker/docker-compose-ray-cuda-req.yaml](docker/docker-compose-ray-cuda-req.yaml)

   ```bash
    HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4 ) MINIO_ROOT_USER={}  MINIO_ROOT_PASSWORD={} AWS_ACCESS_KEY_ID={}  AWS_SECRET_ACCESS_KEY={}  POSTGRES_USER={}  POSTGRES_PASSWORD={}  POSTGRES_DB={}  docker compose -f  ChXpert-Chest-X-rays-Prediction/docker/docker-compose-ray-cuda-req.yaml up -d
   ```

6. **Start the Jupyter + Ray client** on the GPU node:

   ```bash
   cd ChXpert-Chest-X-rays-Prediction/docker
   # Build image
   docker build -t jupyter-ray -f Dockerfile.jupyter-ray .
   # Run container
   HOST_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
   docker run -d --rm -p 8888:8888 \
     -v ../:/home/jovyan/work/ \
     --mount type=bind,source=/mnt/object,target=/mnt/ \
     -e RAY_ADDRESS=http://${HOST_IP}:8265/ \
     --name jupyter jupyter-ray
    docker logs jupyter
   ```

   * Open **Jupyter Lab** at logs
   * Ray Dashboard available at `http://<GPU_NODE_IP>:8265`


### 1. Modeling

**Objective**: Predict 14 binary disease labels from 224×224 chest X-ray crops in a patient-wise train/val/test split to avoid data leakage.

**Data Leakage Prevention**
We group all images by patient ID and ensure that each patient’s scans appear in only one of the train, validation, or test sets. Our custom PyTorch DataLoader uses a lookup table mapping patient IDs to splits, so no patient’s images leak across splits.

**Uncertainty Labels Handling**
The CheXpert dataset annotates some labels as “uncertain.” Because this is a triage task, we treat uncertainty as a positive finding. All labels marked uncertain are converted to 1 during preprocessing, prioritizing sensitivity.

**Input Processing**

* Raw grayscale X-rays are resized to 224×224.
* We duplicate the single channel into three channels.

**Model Architecture**
We utilize the [ViT-Base-Patch-16](https://huggingface.co/google/vit-base-patch16-224) pretrained on ImageNet:

* **Patch Embedding**: Splits image into 16×16 patches capturing local details.
* **Multi-Head Self-Attention**: Learns relationships between spatial regions, enabling detection of both localized and diffuse pathologies.
* **Positional Encoding**: Retains global spatial context.
* **Classification Head**: A fully connected layer outputs a 14-dimensional vector with sigmoid activation.

**Use Case**
In a clinical setting, radiologists feed a patient’s scans into the model. The output scores rank potential findings, allowing clinics to triage urgent cases (e.g., pneumothorax first) while optionally reordering labels based on local protocols.



## 2. Train & Retrain

1. **Initial training** using Ray Train and MLflow:

   ```bash
   # From the GPU Jupyter terminal
   cd work/ray_work
   ray job submit \
     --runtime-env runtime.json \
     --working-dir . \
     -- python train_test.py
   ```

   * Training script: [ray_work/train_test.py](ray_work/train_test.py) logs metrics and checkpoints to MLflow on `129.114.26.91:8000`.

2. **Non-interactive retraining** automatically loads the best run checkpoint:

   ```bash
   cd ChXpert-Chest-X-rays-Prediction/ray_work
   ray job submit \
     --runtime-env runtime.json \
     --working-dir . \
     -- python retrain.py
   ```

   * Retrain script: [ray_work/retrain.py](ray_work/retrain.py) fetches the latest best run from MLflow and resumes training.



## 3. Experiment Tracking

* **MLflow UI**: `http://129.114.26.91:8000` (view runs, metrics, and artifacts).
* **Artifacts stored** in MinIO bucket `mlflow`; **metadata** in Postgres DB `mlflowdb`.




## 4. Scheduling and optional

* **Scheduling**: All jobs submitted via `ray job submit` on the GPU node.
* **Resilience**: Implemented Ray Train fault tolerance and automatic retries from latest checkpoint can be found in [ray_work/train_test.py](ray_work/train_test.py) at the end Failconfig().
* **Distributed Training**: Performed Experiments with DDP (2 GPUs) and FSDP accelerated training by changing the config in ray train Trainer(strategy=).
* **Details available** in MLflow at `http://129.114.26.91:8000` under each run's logs and artifacts.


<pre> ``` Unit 6 and 7: MODEL SERVING AND EVALUATION \
Serving from an API endpoint: Describe how you set up the API endpoint. What is the input? What is the output? \
The API endpoint is implemented using FastAPI, which serves as the backend for inference. The Flask frontend provides a web interface for users who can upload chest X-ray images. \
Input: Single Chest X-ray (or any) image from user sent to back end as an encoded string payload. \
Output: The predictions of the model and the confidence levels for each class are then sent to the frontend and displayed. \
Identify requirements: Requirements of the customer include minimal model size for serving (under 5 GB), which is met by both DenseNet and ViT models. The regular Torch ViT model takes ~160 ms to inference a single image (Grafana monitored), well under the required few seconds. \ 
    
Model optimizations: Many model optimization tests were conducted using: - PyTorch compiled model - ONNX model - Graph-optimized ONNX - Dynamic + static quantized ONNX - ONNX run on CUDA, CPU, and OpenVINO EPs Best performance was with optimized ONNX on CUDA. Torch compiled model also outperformed quantized ones. /

System optimizations: Dockerized services with Prometheus + Grafana monitoring, showing GPU/CPU usage, inference times, etc. System Evaluation/Monitoring: ``` </pre> 
Docker Services Overview:
-------------------------
- fastapi_server: Hosts the ML model for inference and exposes metrics at /metrics.
- flask: Frontend for uploading X-ray images and optional user feedback.
- prometheus: Scrapes metrics from FastAPI and cAdvisor.
- grafana_chexpert: Visualizes metrics on port 3100, with auto-provisioned dashboards.
- jupyter: Provides a notebook environment to run evaluation scripts.
- cadvisor: Monitors container-level resource usage (CPU, memory).

How to Start the System:
------------------------
Run the following command to build and start the services:

    docker compose -f docker/docker-compose-prometheus6.yaml up --build -d

This builds and runs all containers in the background.

Accessing Services:
-------------------
Replace A.B.C.D with the IP of your node server.

- Flask UI: http://A.B.C.D:5050
- FastAPI Docs: http://A.B.C.D:8000/docs
- Grafana Dashboard: http://A.B.C.D:3100 (admin / admin)
- Prometheus UI: http://A.B.C.D:9090
- Jupyter Lab: http://A.B.C.D:8888
- cAdvisor: http://A.B.C.D:8080

Grafana Monitoring:
-------------------
Grafana uses automatic provisioning from the mounted directory:

    ../grafana/provisioning/

To view dashboards:
1. Go to http://A.B.C.D:3100
2. Check dashboards for prediction metrics and container resource usage.

Prometheus Metrics Examples:
----------------------------
- rate(prediction_confidence_bucket[1m]) – View prediction confidence trends.
- predicted_class_total – View prediction frequency per class.

Running Model Evaluation:
-------------------------
1. Open Jupyter Lab: http://A.B.C.D:8888 and paste in token from `docker logs jupyter`
2. Open and run onlineEvalTest.ipynb from `work` directory
home/jovyan/work3. The notebook will test model inference and monitor metrics in real-time.

The dataset is mounted into the container at /mnt/dataset from your host’s /mnt/object2.
The grafana panels include service monitoring with requests per second info, average request duration, request duration percentiles, and an error status. The prediction monitoring dashboard includes average prediction confidence info, cumulative prediction confidence info, prediction confidence occurrences across time range, and predicted class totals across time range. 
Additionally, handling human feedback has been implemented.
This MLOps system supports human-in-the-loop feedback on predictions to enhance model monitoring and retraining efforts.

### Feedback Collection Flow

1. **User Interaction (Flask Frontend)**
   - After uploading a chest X-ray image and viewing the model's predictions, users are prompted to optionally provide feedback.
   - For each of the 14 diagnosis classes, users can select:
     - **Present (1)**
     - **Absent (0)**
     - **Uncertain (-1)**
     - Or leave blank to skip

2. **Data Submission to FastAPI**
   - The selected labels, along with the corresponding image (sent via `multipart/form-data`), are submitted to the FastAPI backend.

3. **Storage in MinIO**
   - Feedback data is stored in a MinIO bucket named `newdata`.
     - A CSV file (`new_data.csv`) stores metadata including:
       - `image_id` (UUID)
       - 14 label columns: `class_0` to `class_13`
     - Images are saved under `feedback_images/` using their UUID filenames.

4. **Use for Continuous Evaluation & Retraining**
   - This dataset of human-labeled examples can be used for:
     - Model performance audits
     - Active learning
     - Periodic retraining or fine-tuning

This loop enables better tracking of real-world model performance and user trust alignment.

Define a business-specific evaluation: Describe this hypothetical evaluation; it’s not something you actually implement.
Optional: Develop multiple options for serving: If you attempt this difficulty point, make sure I know it! Show the parts of the repo that implement each option, and show a comparison of the options with respect to performance and cost of deployment on a commercial cloud. (You can use a commercial cloud cost calculator to estimate costs. Show your work.)


