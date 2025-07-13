# DualReal: Adaptive Joint Training for Lossless Identity-Motion Fusion in Video Customization

<video src="https://github.com/wenc-k/dualreal-customization/releases/download/demo/demo.mp4" controls style="max-width: 100%; height: auto; border-radius: 4px;">
  Your browser does not support the video tag.
</video>

🎉🎉 Our paper, “DualReal: Adaptive Joint Training for Lossless Identity-Motion Fusion in Video
Customization” accepted by ICCV 2025!
**Our [project page](https://wenc-k.github.io/dualreal-customization/).**

## TODO List

- [x] Release the paper and project page. Visit [https://wenc-k.github.io/dualreal-customization/](https://wenc-k.github.io/dualreal-customization/) 
- [x] Release the inference code.
- [x] Release test cases with our pretrained model, prompts, and reference image.
- [x] Release the training code.


## Requirements
The training and inference are conducted on 1 A100 GPU (80GB VRAM)
## Setup
```
git clone https://github.com/wenc-k/DualReal.git
cd DualReal
```


## Environment
All the tests are conducted in Linux. To set up our environment in Linux, please run:
```
conda create -n dualreal python=3.10 -y
conda activate dualreal

pip install -r requirements.txt
```

```
cd train

git clone https://github.com/huggingface/diffusers.git
cd diffusers 
pip install -e .
```


## Checkpoints
1. please download the pre-trained CogVideoX-5b checkpoints from [here](https://huggingface.co/THUDM/CogVideoX-5b), and put the whole folder under `DualReal`, it should look like `DualReal/CogVideoX-5b`

2. please download the open_clip_pytorch_model.bin from [here](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main), and put the file under `train/pretrained/`, it should look like `DualReal/train/pretrained/open_clip_pytorch_model.bin`

3. We provide sample test cases with customization weights (e.g., *dog & guitar and penguin & skateboarding*) from [here](https://huggingface.co/wenc-k/DualReal/tree/main), please put the file under `inference/pretrained_customization/`


## Inference
1. make sure to update the following path variables in `inference/run.sh`:
- `ADAPTER_PATH`: Path to the trained `.pth` file containing the weights of both the adapter and controller modules
- `PROMPT_PATH`: Path to the text file where each line specifies the prompt used for a corresponding inference case
- `REF_IMG_PATH`: Path to the reference appearance image (for motion adapter)
- `OUTPUT_PATH`: Directory where inference results will be saved
- `CLIP_PATH`: **Absolute Path** to the pre-trained CLIP model. (e.g., {}/DualReal/train/pretrained/open_clip_pytorch_model.bin )

2. you can use the provided test weights, prompts, and reference images for inference.

3. start inference by executing:
```
bash run.sh
```


## Training
1. a sample test case is provided under the following directory: train/test_data/

2. start test training by executing:

```
bash train_dog_guitar.sh
```

This script will automatically load the test data and begin the training process with the default configuration.

## Customize Your Identity and Motion!
1. first, you need to prepare your dataset.
```
data
  ├──identity-A (motion-B)
        ├── videos
        ├── prompts.txt
        ├── videos.txt
```
- `videos`: Contains the video files
- `prompts.txt`: Contains the prompts
- `videos.txt`: Contains the list of video files in the videos/ directory


2. make sure to update the following path variables in `train_{}_{}.sh` to match your own dataset structure:
- `ID_PATH`: Path to the subject identity images (identity-A)
- `REF_PATH`: Path to the reference appearance image (for motion adapter)
- `MOTION_PATH`: Path to the motion videos (motion-B)
- `OUTPUT_PATH`: Directory where training results and checkpoints will be saved

3. start training by executing:
```
bash train_{}_{}.sh
```


## Citation:
Don't forget to cite this source if it proves useful in your research!
```
@article{wang2025dualreal,
  title={DualReal: Adaptive Joint Training for Lossless Identity-Motion Fusion in Video Customization},
  author={Wang, Wenchuan and Huang, Mengqi and Tu, Yijing and Mao, Zhendong},
  journal={arXiv preprint arXiv:2505.02192},
  year={2025}
}
```