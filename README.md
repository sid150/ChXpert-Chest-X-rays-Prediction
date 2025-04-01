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
Serving from an API endpoint:
	Use FastAPI in conjunction with the frontend and pytorch vision model.
Identify requirements:
    - Model Size: TBD
    - Single Sample Online Inference Latency: less than 1 second
    - Cloud Deployment Concurrency: 4
    - These requirements are determined from a single business/clinic accessing the vision model.
 
Model optimizations to satisfy requirements:
	Convert pytorch model to ONNX model and run with ONNX runtime to compare inference times. Compare the regular ONNX model with the dynamically quantized model. Other optimizations may be tested as the project progresses. 

System optimizations to satisfy requirements:
	Attempt to use Triton inference server with multiple GPUs with both python backend and ONNX backend to compare performance. 

Develop multiple options for serving(EXTRA DIFFICULTY POINTS):
	Compare multiple local on-device machine performance with server-grade CPU inference performance. 

##### Unit 7:
(Note, no lab performed yet, will update accordingly once experience is had)
<u>Offline evaluation of model:</u>
	Create automated evaluation plan in conjunction with MLFlow for metrics logging. Create concurrency tests and cloud tests for domain specific use cases, and also a simple standard use case. Test on special test batches of data that was highly misclassified(some certain disease). 
	Finally automatically save and register the model if passing evaluation criteria.
<u>Load test in staging:</u>
	This will be part of continuous X pipeline, and like the model serving lab, multiple requests will be made concurrently to simulate the real-world use case described in the business-specific evaluation part. 

<u>Online evaluation in canary stage:</u>
	Figure out a way to replicate real world users on certain device types. Similar to load test stage, run concurrent inference tasks as defined by business-specific evaluation and analyze results to see if they meet the requirements posed. Methods used will be similar to lab 7.

<u>Close the loop:</u>
	On the front-end website, users uploading X-rays for model prediction will be given a choice of buttons to choose from indicating whether the model’s prediction was accurate or not. This will simulate doctor’s opinion on the X-ray classification and generate new online data with ground truth labels. 

<u>Define a business-specific evaluation:</u>
	Deploy in production the model inference service to half of users. Measure, and log the inference latencies. Compare these inference latencies to the accuracy and time it takes for physicians/doctors to notice issues in X-ray. Additionally, evaluate the time saved by using automated model inference on X-rays instead of relying on time it takes for x-ray to be sent to a physician computer, then opened and checked once a doctor is available. Hopefully, this will showcase a benefit of using this ML system to detect issues in X-rays automatically instead of having to spend precious time and human resources to figure out whether an X-ray and patient need to be examined in further detail. 


#### Data pipeline

<!-- Make sure to clarify how you will satisfy the Unit 8 requirements,  and which 
optional "difficulty" points you are attempting. -->

<u>Persistent Storage:</u>
	Following Lab 8, we will use persistent storage through Chameleon, and we will store the main dataset of around 400+ Gigabytes, model checkpoints, the CNN model and ViT model, and container images for inferencing. 

<u>Offline data:</u>
	We will keep unstructured training data of Chest X-ray images in a repository of persistent storage as defined above.

<u>Data pipelines:</u>
	We have a simple ETL pipeline due to using image classification, we will convert the images into the proper input size by resizing to the required shape and still stored as images. During model training or inferencing, the data will be converted to pytorch tensor and normalized. Data source is CheXpert dataset of image files, and resizing will be done through the Pillow library function. 

<u>Online data:</u>
	In order to evaluate our MLOps system for chest X-ray image classification in a real-time environment, we will simulate the arrival of "online data", mimicking the behavior of X-ray images being submitted in a clinical setting for inference. This simulation is critical to test the robustness, latency, and scalability of our streaming pipeline and inference service.
We will implement a Python-based data simulator that reads X-ray images from a local directory and sends them to our deployed inference API endpoint. The script will simulate two types of data arrival patterns, single and batch. Simulated data will be data not form CheXpert dataset but other similar positional images of chest X-rays with the ground truth label supplied of either having a medical issue or not. These new images should have an inference performed on them and then put into a folder for future model-retraining if necessary. 


#### Continuous X

<!-- Make sure to clarify how you will satisfy the Unit 3 requirements,  and which 
optional "difficulty" points you are attempting. -->


