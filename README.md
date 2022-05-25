# Pytorch/XLA on TPU

A step-by-step guide to setting up a TPU pod, and running your first Pytorch/XLA code on it.

## TPU Setup Instructions:

1. Connect to the TPU cluster on gcloud:

`gcloud container clusters get-credentials sfr-tpu-prem1-232-europe-west4-a --zone europe-west4-a --project salesforce-research-internal`

2. If you are launching TPU pods for the first time, you need to create a GUS ticket requesting TPUs. You can clone this GUS ticket and use it as a template: [GUS ticket sample](https://gus.lightning.force.com/lightning/r/ADM_Work__c/a07EE00000xJc8tYAC/view)

3.  Execute: `kubens` to list the active user namespace. If your username does not show up (typically of form `sfn-ns-<userID>`), then your GUS ticket has not been approved yet. Make sure it gets approved first.

4. Once you see your username on the active user namespace, connect to it using: 
`kubens sfr-ns-<userID>`

5. Different from GCP setup, you don’t need to write your own Dockerfile. Existing Docker images are used in the yaml file for instantiating TPU pods. Pytorch version you want to use needs to be consistent at two places in the yaml file. See `tpu3-8.yaml` above. In this file, replace the `name` field in the `metadata` to your a name of your choice. And replace `darpit` with your userID everywhere in the file.
  
  Notice line 10 and 25 in the file. Line 10 specifies "pytorch-1.9", and line 25 specifies an existing docker image to be used. The docker image name specifies r1.9_3.7. Here 1.9 refers to the Pytorch version, and 3.7 refers to the Python version. Possible Pytorch versions that can be used in line 10 can be found [here](https://cloud.google.com/tpu/docs/supported-tpu-versions). The possible docker images that can be used can be found [here](https://github.com/pytorch/xla/). The version specified in the two must match. Note that the latter link does not exhaustively list all the docker images available. You can typing the link (eg. `gcr.io/tpu-pytorch/xla:r1.9_3.7`) in your browser and check if that image exists.

6. Now, to launch a TPU pod using this yaml file, execute:

`kubectl create -f  tpu3-8.yaml`

7. To check the status of your pod, execute:

`kubectl get pods`

It will look something like this once the container is running:
![Alt text](images/get_pods.png?raw=true "")

8. Once created, to connect to the pod, execute: 

`kubectl exec sfr-tpu-darpit3-8-2-pgkmw -i -t -- bash`

9. To delete the pod, note that TPU pods are deleted by jobID. To get the jobID, execute: kubectl get jobs. It will look like this:
![Alt text](images/get_jobs.png?raw=true "")

Delete the pod use command: `kubectl delete job sfr-tpu-darpit3-8-2`

10. After connecting to a pod, once you go to `/export/home/`, you will notice that it is empty. This is because this folder shows the TPU volume which is different from the file volume used when launching GPU pods.

11. Copy files from local machine to the TPU pod using command:
 
`kubectl cp file.py sfr-tpu-darpit3-8-2-pgkmw:/export/home/`

## Run Pytorch/XLA code:

1. As a first code, let's run mnist_pytorch_xla.py. But first setup some environment variables:

`export TPU_IP=$(env | grep KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS | sed 's/KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS=grpc:\/\///g' | head -n 1)`
  
`export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP"`

`python mnist_pytorch_xla.py`


## Errors/Problems I experienced during the setup process:

1. I faced an error: 

`ConnectionResetError: [Errno 104] Connection reset by peer.`

This could be avoided by reducing the `num_workers` where the Pytorch dataset is defined. But this was not the root cause. It was the problem described in the next point.

2. Got another (related) error: 
 
`ERROR: Unexpected bus error encountered in worker.`

This might be caused by insufficient shared memory (shm). Figured out that this error was because of my config in yaml file. I had not specified `dshm` (shared memory) in `volumes` and `volumeMounts`. See [this](https://medium.com/dive-into-ml-ai/raising-shared-memory-limit-of-a-kubernetes-container-62aae0468a33) link for details.
