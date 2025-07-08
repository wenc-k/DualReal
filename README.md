# DualReal: Adaptive Joint Training for Lossless Identity-Motion Fusion in Video Customization

🎉🎉 Our paper, “DualReal: Adaptive Joint Training for Lossless Identity-Motion Fusion in Video
Customization” accepted by ICCV 2025!
**Our [project page](https://wenc-k.github.io/dualreal-customization/).**

## Showcase


## TODO List

- [x] Release the paper and project page. Visit [https://wenc-k.github.io/dualreal-customization/](https://wenc-k.github.io/dualreal-customization/) 
- [x] Release the code.


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

2. please download the open_clip_pytorch_model.bin from [here](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main), and put the file under `DualReal/train/pretrained`, it should look like `DualReal/train/pretrained/open_clip_pytorch_model.bin`


## Training Instructions
1. a sample test case is provided under the following directory: train/test_data/

2. start test training by executing:

```
bash train_dog_guitar.sh
```

This script will automatically load the test data and begin the training process with the default configuration.

## Customize Your Identity and Motion!
1. first, you need to prepare your dataset.

```
identity/motion
├── videos
├── prompts.txt
├── videos.txt
```

- `videos`: Contains the video files
- `prompts.txt`: Contains the prompts
- `videos.txt`: Contains the list of video files in the videos/ directory


2. make sure to update the following path variables in `train_{}_{}.sh` to match your own dataset structure:
- `ID_PATH`: Path to the subject identity image(s)
- `REF_PATH`: Path to the reference appearance images (for motion adapter)
- `MOTION_PATH`: Path to the motion videos
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