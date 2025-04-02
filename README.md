# ChXpert-Chest-X-rays-Prediction


## Title of project

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

<!-- Overall digram of system. Doesn't need polish, does need to show all the pieces. 
Must include: all the hardware, all the containers/software platforms, all the models, 
all the data. -->

### Summary of outside materials

<!-- In a table, a row for each dataset, foundation model. 
Name of data/model, conditions under which it was created (ideally with links/references), 
conditions under which it may be used. -->

|              | How it was created | Conditions of use |
|--------------|--------------------|-------------------|
| CheXpert Dataset form Stanford ML   |  |                    |
| Data set 2   |                    |                   |
| Base model 1 |                    |                   |
| etc          |                    |                   |


### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), 
how much/when, justification. Include compute, floating IPs, persistent storage. 
The table below shows an example, it is not a recommendation. -->

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| `m1.medium` VMs | 3 for entire project duration                     | ...           |
| `gpu_mi100`     | 4 hour block twice a week                         |               |
| Floating IPs    | 1 for entire project duration, 1 for sporadic use |               |
| etc             |                                                   |               |

### Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the 
diagram, (3) justification for your strategy, (4) relate back to lecture material, 
(5) include specific numbers. -->

#### Model training and training platforms

<!-- Make sure to clarify how you will satisfy the Unit 4 and Unit 5 requirements, 
and which optional "difficulty" points you are attempting. -->

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


