import argparse
import os
import json

import torch
from diffusers import CogVideoXDPMScheduler, AutoencoderKLCogVideoX
from model.dit import CogVideoXTransformer3DModelWithAdapter
from model.pipeline import CogVideoXPipeline
from diffusers.utils import export_to_video
from transformers import AutoTokenizer, T5EncoderModel

from train.src.clip import FrozenOpenCLIPCustomEmbedder
from train.src import transforms as data 
from PIL import Image

def motion_ref_image_encoder(pretrained, image_path, device, dtype=torch.bfloat16):
    # clip init
    image_encoder = FrozenOpenCLIPCustomEmbedder(pretrained, device=device)
    image_encoder.to(device=device, dtype=dtype)
    # load img
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # trans img
    vit_trans = data.Compose([
        #data.CenterCropWide(size=(256, 256)), # change if different img size
        data.Resize([224, 224]),
        data.ToTensor(),
        data.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

    with torch.no_grad():
        image_tensor = vit_trans(image)
        image_tensor = image_tensor.unsqueeze(0)
        y_visual = image_encoder(image=image_tensor.to(device=device, dtype=dtype))
        y_visual = y_visual.unsqueeze(1) # [1,1,1024]

    return y_visual.to(device=device, dtype=dtype)

def load_cogvideox_pipeline(args, dtype=torch.bfloat16):
    
    cogvideox_tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    cogvideox_text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    cogvideox_text_encoder = cogvideox_text_encoder.to(dtype)

    cogvideox_vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    cogvideox_vae = cogvideox_vae.to(dtype)

    cogvideox_scheduler = CogVideoXDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    cogvideox_transformer = CogVideoXTransformer3DModelWithAdapter.from_pretrained(  
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=dtype,
        revision=args.revision,
        variant=args.variant,
        spatial_adapter_alpha=args.spatial_adapter_alpha,
        temporal_adapter_alpha=args.temporal_adapter_alpha,
        use_IdAdapter=args.use_IdAdapter,
        use_MotionAdapter=args.use_MotionAdapter,
        spatial_adapter_hidden_dim=args.spatial_adapter_hidden_dim,
        temporal_adapter_hidden_dim=args.temporal_adapter_hidden_dim
    )

    if args.use_IdAdapter:
        if args.use_id_condition:
            cogvideox_transformer.set_id_adapter(conditioning_dim=args.spatial_adapter_condition_dim)
        else:
            cogvideox_transformer.set_id_adapter()
    
    if args.use_MotionAdapter:
        if args.use_motion_condition:
            cogvideox_transformer.set_motion_adapter(conditioning_dim=args.temporal_adapter_condition_dim)
            print("\033[1;31m set motion adapter with motion conditioning... \033[0m")
        else:
            cogvideox_transformer.set_motion_adapter()
    
    if args.use_controller:
        cogvideox_transformer.set_controller(output_dim=args.hypernet_outputdim)

    adapter_dict = torch.load(args.adapter_path, map_location='cpu')
    transformer_dict = cogvideox_transformer.state_dict()
    transformer_dict.update(adapter_dict)
    cogvideox_transformer.load_state_dict(transformer_dict, strict=False)

    print("transformer init successfully...")

    cogvideox_transformer = cogvideox_transformer.to(dtype)
    cogvideox_transformer.eval()

    print("from pretrained path:", args.pretrained_model_name_or_path)
    
    pipe = CogVideoXPipeline(
        tokenizer=cogvideox_tokenizer,
        text_encoder=cogvideox_text_encoder,
        vae=cogvideox_vae,
        scheduler=cogvideox_scheduler,
        transformer=cogvideox_transformer,
    ) 
    return pipe
    
def log_validation(
    pipe,
    seed,
    pipeline_args,
    output_path,
    order,
    use_gpu_accelerate=False,
):
    if use_gpu_accelerate:
        pipe = pipe.to("cuda")

    # run inference
    generator = torch.Generator(device="cuda").manual_seed(seed)
    video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]
    print("generate video successful...")
    # save video
    current_file_path = os.path.join(output_path, f"video_case_{order}.mp4")
    export_to_video(video, current_file_path, fps=8)
    print("\033[1;31m saved video to: \033[0m", current_file_path)
    # log json
    json_data = {
        "file_path": current_file_path,
        "prompt": pipeline_args["prompt"],
    }
    return json_data

def load_prompts(file_path: str) -> list[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def generate_video(
    args,
    prompt: str,
    output_path: str = "./output",
    guidance_scale: float = 6.0,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    use_dynamic_cfg: bool = False,
):
    print("use_gpu_accelerate:", args.use_gpu_accelerate)

    y_image = None
    if args.ref_image_path:
        print(" Loading reference image... ")
        y_image = motion_ref_image_encoder(
            pretrained=args.clip_pretrained, 
            image_path=args.ref_image_path,
            device="cuda",
            dtype=dtype,
            )

    pipe = load_cogvideox_pipeline(args, dtype)

    # 2. Set Scheduler.
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # 3. Enable CPU offload for the model.
    if not args.use_gpu_accelerate:
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    prompt_list = load_prompts(prompt)

    json_datas = []
    video_dir_path = os.path.join(output_path, "videos")
    os.makedirs(video_dir_path, exist_ok=True)
    for i, prompt in enumerate(prompt_list):
        print(f"\033[1;31m Generating {i+1}/{len(prompt_list)}: {prompt} \033[0m")
        pipeline_args = {
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "use_dynamic_cfg": use_dynamic_cfg,
            "height": 480,
            "width": 720,
            "y_image": y_image,
        }

        json_data = log_validation(
            pipe=pipe,
            seed=seed,
            pipeline_args=pipeline_args,
            order=i+1,
            output_path=video_dir_path,
            use_gpu_accelerate=args.use_gpu_accelerate,
        )
        json_datas.append(json_data)

    flie_prompt_json = os.path.join(output_path, "prompt.json")
    with open(flie_prompt_json, "w") as f:
        json.dump(json_datas, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, default=None, help="The description of the video to be generated")
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default="../CogVideoX-5b", help="The path of the pre-trained model to be used"
    )
    parser.add_argument(
        "--output_path", type=str, default="./output.mp4", help="The path where the generated video will be saved"
    )
    parser.add_argument(
        "--image_dataset_folder", type=str, default=""
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    # adapters
    parser.add_argument(
        "--use_controller",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--hypernet_outputdim",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--spatial_adapter_alpha",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--temporal_adapter_alpha",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--use_IdAdapter",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--use_MotionAdapter",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--use_id_condition",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--use_motion_condition",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--spatial_adapter_hidden_dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--temporal_adapter_hidden_dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--temporal_adapter_condition_dim",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--spatial_adapter_condition_dim",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--ref_image_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default="",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="", 
    )    
    parser.add_argument(
        "--use_gpu_accelerate",
        type=bool,
        default=False, 
    )
    parser.add_argument(
        "--use_dynamic_cfg",
        type=bool,
        default=False, 
    )

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    print("dtype: ", dtype)

    generate_video(
        args,
        prompt=args.prompt,
        output_path=args.output_path,
        guidance_scale=args.guidance_scale,
        dtype=dtype,
        seed=args.seed,
        use_dynamic_cfg=args.use_dynamic_cfg,
    )