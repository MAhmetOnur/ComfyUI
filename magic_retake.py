import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import logging

# Import ComfyUI's argument parser
from comfy.cli_args import parser as comfy_parser

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS


def main():
    # Add our custom arguments to ComfyUI's parser
    comfy_parser.add_argument("--gender", type=str, choices=["male", "female"], required=True, help="Specify the gender for the prompts")
    comfy_parser.add_argument("--concept", type=str, required=True, help="Specify the concept for the prompts")
    comfy_parser.add_argument("--num_prompts", type=int, default=2, help="Specify the number of prompts to generate")
    comfy_parser.add_argument("--output_path", type=str, default=".", help="Specify the output directory for generated images")

    # Parse all arguments
    args = comfy_parser.parse_args()

    # Create output directory
    concept_words = args.concept.split()[:2]
    output_dir = os.path.join(args.output_path, f"{'_'.join(concept_words)}_Magic_Retake_Content")
    os.makedirs(output_dir, exist_ok=True)

    # Import the prompt_generator function
    from prompt_generator import prompt_generator

    # Generate prompts using command-line arguments
    prompts_data = prompt_generator(gender=args.gender, concept=args.concept, num_prompts=args.num_prompts)

    # Save the JSON file to the output directory
    import json
    json_filename = f"prompts_{args.gender}_{args.concept.replace(' ', '_')}.json"
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, 'w') as json_file:
        json.dump(prompts_data, json_file, indent=2)

    logging.info(f"Prompts JSON saved to: {json_path}")

    import_custom_nodes()
    with torch.inference_mode():
        # Initialize and load models only once
        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        dualcliploader_11 = dualcliploader.load_clip(
            clip_name1="t5xxl_fp16.safetensors",
            clip_name2="clip_l.safetensors",
            type="flux",
        )

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_10 = vaeloader.load_vae(vae_name="ae.safetensors")
        vaeloader_47 = vaeloader.load_vae(vae_name="vae-ft-mse-840000.safetensors")

        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unetloader_12 = unetloader.load_unet(
            unet_name="flux1-dev.safetensors", weight_dtype="default"
        )

        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        checkpointloadersimple_38 = checkpointloadersimple.load_checkpoint(
            ckpt_name="epicrealism-v5.safetensors"
        )

        upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
        upscalemodelloader_52 = upscalemodelloader.load_model(
            model_name="4x-UltraSharp.pth"
        )

        # Initialize other non-changing components
        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        ksamplerselect_16 = ksamplerselect.get_sampler(sampler_name="heun")

        modelsamplingflux = NODE_CLASS_MAPPINGS["ModelSamplingFlux"]()
        modelsamplingflux_30 = modelsamplingflux.patch(
            max_shift=1.15,
            base_shift=0.5,
            width=1088,
            height=1920,
            model=get_value_at_index(unetloader_12, 0),
        )

        # Function to process a single text input
        def process_text(text, prefix):
            cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            cliptextencode_6 = cliptextencode.encode(
                text=text,
                clip=get_value_at_index(dualcliploader_11, 0),
            )

            randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
            randomnoise_25 = randomnoise.get_noise(noise_seed=random.randint(1, 2**64))

            emptysd3latentimage = NODE_CLASS_MAPPINGS["EmptySD3LatentImage"]()
            emptysd3latentimage_27 = emptysd3latentimage.generate(
                width=1088, height=1920, batch_size=1
            )

            fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
            fluxguidance_26 = fluxguidance.append(
                guidance=3.5, conditioning=get_value_at_index(cliptextencode_6, 0)
            )

            basicguider = NODE_CLASS_MAPPINGS["BasicGuider"]()
            basicguider_22 = basicguider.get_guider(
                model=get_value_at_index(modelsamplingflux_30, 0),
                conditioning=get_value_at_index(fluxguidance_26, 0),
            )

            basicscheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()
            basicscheduler_17 = basicscheduler.get_sigmas(
                scheduler="beta",
                steps=30,
                denoise=1,
                model=get_value_at_index(modelsamplingflux_30, 0),
            )

            samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
            samplercustomadvanced_13 = samplercustomadvanced.sample(
                noise=get_value_at_index(randomnoise_25, 0),
                guider=get_value_at_index(basicguider_22, 0),
                sampler=get_value_at_index(ksamplerselect_16, 0),
                sigmas=get_value_at_index(basicscheduler_17, 0),
                latent_image=get_value_at_index(emptysd3latentimage_27, 0),
            )

            vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(samplercustomadvanced_13, 0),
                vae=get_value_at_index(vaeloader_10, 0),
            )

            vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
            vaeencode_48 = vaeencode.encode(
                pixels=get_value_at_index(vaedecode_8, 0),
                vae=get_value_at_index(vaeloader_47, 0),
            )

            cliptextencode_49 = cliptextencode.encode(
                text="", clip=get_value_at_index(checkpointloadersimple_38, 1)
            )

            cliptextencode_50 = cliptextencode.encode(
                text="", clip=get_value_at_index(checkpointloadersimple_38, 1)
            )

            saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()
            ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
            imageupscalewithmodel = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()
            imagescale = NODE_CLASS_MAPPINGS["ImageScale"]()

            saveimage_9 = saveimage.save_images(
                filename_prefix=f"{prefix}_flux",
                images=get_value_at_index(vaedecode_8, 0),
                output_path=output_dir
            )

            ksampler_40 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=30,
                cfg=3,
                sampler_name="dpmpp_3m_sde",
                scheduler="karras",
                denoise=0.3,
                model=get_value_at_index(checkpointloadersimple_38, 0),
                positive=get_value_at_index(cliptextencode_49, 0),
                negative=get_value_at_index(cliptextencode_50, 0),
                latent_image=get_value_at_index(vaeencode_48, 0),
            )

            vaedecode_43 = vaedecode.decode(
                samples=get_value_at_index(ksampler_40, 0),
                vae=get_value_at_index(vaeloader_47, 0),
            )

            imageupscalewithmodel_53 = imageupscalewithmodel.upscale(
                upscale_model=get_value_at_index(upscalemodelloader_52, 0),
                image=get_value_at_index(vaedecode_43, 0),
            )

            imagescale_57 = imagescale.upscale(
                upscale_method="nearest-exact",
                width=2160,
                height=3840,
                crop="disabled",
                image=get_value_at_index(imageupscalewithmodel_53, 0),
            )

            saveimage_44 = saveimage.save_images(
                filename_prefix=f"{prefix}_realism",
                images=get_value_at_index(imagescale_57, 0),
                output_path=output_dir
            )

            return

        # Process each prompt pair
        for prompt_pair in prompts_data["prompts"]:
            prompt_id = prompt_pair["prompt_id"]
            
            # Process white prompt
            process_text(prompt_pair["white_prompt"], f"{prompt_id}_w")
            logging.info(f"Processed white prompt: {prompt_id}")

            # Process black prompt
            process_text(prompt_pair["black_prompt"], f"{prompt_id}_b")
            logging.info(f"Processed black prompt: {prompt_id}")

if __name__ == "__main__":
    main()