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

| Name                            | Responsible for | Link to their commits in this repo |
|---------------------------------|-----------------|------------------------------------|
| All team members                |                 |                                    |
| Team member 1                   |                 |                                    |
| Team member 2                   |                 |                                    |
| Team member 3                   |                 |                                    |



### System diagram
![Image](https://github.com/user-attachments/assets/148f3cad-d919-46e6-b4bd-3b805bbc56af)
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
| `compute_icelake_650r` VMs | 3 for entire project duration   | Need at least 1 server with persistent storage of greater than 450 GB due to dataset    |
| `gpu_V100`     | 4 hour block twice a week  | ViT model training               |
| Floating IPs    | 1 for entire project duration, 1 for sporadic use |                |


### Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the 
diagram, (3) justification for your strategy, (4) relate back to lecture material, 
(5) include specific numbers. -->

#### Model Training and Training Platforms

##### Unit 4: Model Training at Scale

###### Train and Re-train

We use DenseNet121 and ViT-Large as our baseline models for the chest X-ray classification task. Both models are publicly available and can be imported directly from open-source libraries such as `torchvision.models` (for DenseNet121) and Hugging Face Transformers (for ViT-Large). Our dataset is the CheXpert dataset, an open-source chest X-ray dataset released by Stanford, which contains over 224,000 chest radiographs from 65,000+ patients, occupying approximately 450 GB of storage. If storage limitations arise, we will work with a reduced, representative sample of the dataset.

To support training and re-training, we store the dataset using Chameleon’s persistent volume block storage, which allows seamless mounting of data across training nodes. Each model is modified by adding a small multi-label classification head suitable for the CheXpert task, and we experiment with two approaches for DenseNet: fine-tuning the pretrained model and training from scratch using randomly initialized weights. For ViT-Large, we will only stick with fine-tuning the full network and fine-tuning the last few layers. The dataset is split into training and re-training subsets to simulate a real-world continual learning pipeline. After training, we compare the performance of the two models to determine which performs better on this task.

###### Modeling

We selected DenseNet121 due to its widespread use in chest X-ray classification and its relatively small size of 7.98 million parameters, which enables it to run on low-end GPUs and complete training quickly. This makes DenseNet an ideal choice for establishing a fast and reliable baseline. On the other hand, we use ViT-Large, which has approximately 307 million parameters, as a high-capacity model capable of leveraging the large CheXpert dataset. Its self-attention mechanism allows ViT-Large to capture long-range dependencies across the image, making it well-suited for detecting subtle and diffuse abnormalities such as cardiomegaly and infiltrates, which span larger spatial regions in radiographs.

As ViT-Large cannot run on a single low-end GPU, we will leverage training strategies discussed in Unit 4 such as gradient accumulation, reduced precision (float16), and mixed precision training using AMP. Furthermore, we will experiment with parameter-efficient fine-tuning methods like LoRA and QLoRA, which allow us to fine-tune only a small number of parameters, making large-model fine-tuning feasible within limited memory constraints.

###### Attempting Difficulty Points

To satisfy the “training strategies for large models” requirement, we use ViT-Large to test various techniques including gradient accumulation, mixed precision training, and LoRA/QLoRA. These methods are designed to reduce memory usage and speed up training without compromising model performance. We will conduct experiments measuring the impact of each strategy on memory footprint, training time, and validation AUROC, and report the results using plots and metrics similar to those used in the Unit 4 lab assignment.

We will also attempt to “use distributed training to increase velocity.” For DenseNet121, we can apply DistributedDataParallel (DDP) across multiple GPUs and evaluate how it scales with 1, 2, and 4 GPUs. For ViT-Large, we aim to use FullyShardedDataParallel (FSDP) to reduce memory consumption and enable large-scale fine-tuning. In both cases, we will measure training time per epoch and throughput (images per second) and plot these against the number of GPUs used to evaluate scaling efficiency.

---

##### Unit 5: Model Training Infrastructure and Platform

###### Experiment Tracking

We will track all experiments using MLflow, including model configurations, hyperparameters, training time, memory usage, and performance metrics. Our experiments include training DenseNet121 from scratch versus fine-tuning, as well as full and partial fine-tuning of ViT-Large. For each model, we will perform sweeps over learning rates (e.g., 1e-3, 1e-4, 5e-5) and batch sizes (e.g., 4, 8, 16, depending on GPU memory). We closely monitor GPU utilization, epoch duration, and memory usage to optimize training efficiency.

The main metric we use to evaluate model performance is the Area Under the ROC Curve (AUC), which is standard in medical classification tasks. For each experiment, we log per-class AUC scores along with training curves to assess overfitting and generalization. By systematically comparing these logs, we can select the best hyperparameter settings for each model.

###### Scheduling Training Jobs

We will use Ray Train to schedule and manage distributed training jobs across multiple nodes on Chameleon Cloud. Ray provides a clean abstraction over PyTorch, allowing us to launch training runs using DDP or FSDP with minimal code changes. We will run a Ray cluster and submit jobs to it as part of our continuous training pipeline, which lets us queue experiments such as LoRA vs. QLoRA, AMP vs. FP32, and different GPU counts. This setup helps us maximize cluster utilization and accelerate experimentation.


#### Model serving and monitoring platforms

<!-- Make sure to clarify how you will satisfy the Unit 6 and Unit 7 requirements, 
and which optional "difficulty" points you are attempting. -->
##### Unit 6:
<ins>Serving from an API endpoint:</ins>
	Use FastAPI in conjunction with the frontend and pytorch vision model.
Identify requirements:
    - Model Size: TBD
    - Single Sample Online Inference Latency: less than 1 second
    - Cloud Deployment Concurrency: 4
    - These requirements are determined from a single business/clinic accessing the vision model.
 
<ins>Model optimizations to satisfy requirements:</ins>
	Convert pytorch model to ONNX model and run with ONNX runtime to compare inference times. Compare the regular ONNX model with the dynamically quantized model. Other optimizations may be tested as the project progresses. 

<ins>System optimizations to satisfy requirements:</ins>
	Attempt to use Triton inference server with multiple GPUs with both python backend and ONNX backend to compare performance. 

<ins>Develop multiple options for serving(EXTRA DIFFICULTY POINTS):</ins>
	Compare multiple local on-device machine performance with server-grade CPU inference performance. 

##### Unit 7:
(Note, no lab performed yet, will update accordingly once experience is had)
<ins>Offline evaluation of model:</ins>
	Create automated evaluation plan in conjunction with MLFlow for metrics logging. Create concurrency tests and cloud tests for domain specific use cases, and also a simple standard use case. Test on special test batches of data that was highly misclassified(some certain disease). 
	Finally automatically save and register the model if passing evaluation criteria.
<ins>Load test in staging:</ins>
	This will be part of continuous X pipeline, and like the model serving lab, multiple requests will be made concurrently to simulate the real-world use case described in the business-specific evaluation part. 

<ins>Online evaluation in canary stage:</ins>
	Figure out a way to replicate real world users on certain device types. Similar to load test stage, run concurrent inference tasks as defined by business-specific evaluation and analyze results to see if they meet the requirements posed. Methods used will be similar to lab 7.

<ins>Close the loop:</ins>
	On the front-end website, users uploading X-rays for model prediction will be given a choice of buttons to choose from indicating whether the model’s prediction was accurate or not. This will simulate doctor’s opinion on the X-ray classification and generate new online data with ground truth labels. 

<ins>Define a business-specific evaluation:</ins>
	Deploy in production the model inference service to half of users. Measure, and log the inference latencies. Compare these inference latencies to the accuracy and time it takes for physicians/doctors to notice issues in X-ray. Additionally, evaluate the time saved by using automated model inference on X-rays instead of relying on time it takes for x-ray to be sent to a physician computer, then opened and checked once a doctor is available. Hopefully, this will showcase a benefit of using this ML system to detect issues in X-rays automatically instead of having to spend precious time and human resources to figure out whether an X-ray and patient need to be examined in further detail. 


#### Data pipeline

<!-- Make sure to clarify how you will satisfy the Unit 8 requirements,  and which 
optional "difficulty" points you are attempting. -->

<ins>Persistent Storage:</ins>
	Following Lab 8, we will use persistent storage through Chameleon, and we will store the main dataset of around 400+ Gigabytes, model checkpoints, the CNN model and ViT model, and container images for inferencing. 

<ins>Offline data:</ins>
	We will keep unstructured training data of Chest X-ray images in a repository of persistent storage as defined above.

<ins>Data pipelines:</ins>
	We have a simple ETL pipeline due to using image classification, we will convert the images into the proper input size by resizing to the required shape and still stored as images. During model training or inferencing, the data will be converted to pytorch tensor and normalized. Data source is CheXpert dataset of image files, and resizing will be done through the Pillow library function. 

<ins>Online data:</ins>
	In order to evaluate our MLOps system for chest X-ray image classification in a real-time environment, we will simulate the arrival of "online data", mimicking the behavior of X-ray images being submitted in a clinical setting for inference. This simulation is critical to test the robustness, latency, and scalability of our streaming pipeline and inference service.
We will implement a Python-based data simulator that reads X-ray images from a local directory and sends them to our deployed inference API endpoint. The script will simulate two types of data arrival patterns, single and batch. Simulated data will be data not form CheXpert dataset but other similar positional images of chest X-rays with the ground truth label supplied of either having a medical issue or not. These new images should have an inference performed on them and then put into a folder for future model-retraining if necessary. 


#### Continuous X

<!-- Make sure to clarify how you will satisfy the Unit 3 requirements,  and which 
optional "difficulty" points you are attempting. -->


