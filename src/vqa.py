"""
Main entrypoint for multimodal input (text, video, audio) to generate multimodal output (text, audio, animation).
Example run: python src/vqa.py --model_path="./model_checkpoints/04-23_18-53-28/checkpoint-1000" --video_path="./data/videos/sample.mp4" --audio_path="./data/audio/sample.wav" --user_question="What is happening in the video?"
"""

import argparse
import sys
import logging
import torch
from model.modeling_llama import MultimodalLlamaForConditionalGeneration
from processing_llama import MultiModalLlamaProcessor
from utils.utils import get_available_device
from helper_functions import preprocess_video, preprocess_audio

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to the pretrained model weights')
    parser.add_argument('--video_path', required=True, help='Path to the prompt video file')
    parser.add_argument('--audio_path', required=True, help='Path to the prompt audio file')
    parser.add_argument('--user_question', required=True, help='The question to ask about the video to the model. Example: "What is happening in the video?"')

    args = parser.parse_args(sys.argv[1:])
    logging.info(f"Parameters received: {args}")

    logging.info("Loading pretrained model...")
    device = get_available_device()
    multimodal_llama_model = MultimodalLlamaForConditionalGeneration.from_pretrained(args.model_path,
                                                                                     device_map="cpu",
                                                                                     torch_dtype=torch.bfloat16).eval()

    processor = MultiModalLlamaProcessor.from_pretrained(args.model_path)

    logging.info("Running model for VQA...")

    prompt = f"user: {args.user_question} \nassistant:"

    # Preprocess video and audio
    video_features = preprocess_video(args.video_path)
    audio_features = preprocess_audio(args.audio_path)

    # Prepare the inputs
    inputs = processor(text=prompt, videos=video_features, audio=audio_features, return_tensors='pt').to(device, torch_dtype=torch.bfloat16)

    # Generate output from the model
    output = multimodal_llama_model.generate(**inputs, max_new_tokens=200, do_sample=False)

    # Decode text output
    text_output = processor.decode(output[0][2:], skip_special_tokens=True)
    logging.info(f"Model text answer: {text_output}")

    # For demonstration, let's assume the model returns audio and animation features as part of its outputs
    # Here we mock this as normally these would come from a different part of the model's output
    audio_output = b''  # Placeholder for actual audio output
    animation_output = b''  # Placeholder for actual animation output

    # Save audio and animation outputs
    with open("output_audio.wav", "wb") as f:
        f.write(audio_output)

    with open("output_animation.fbx", "wb") as f:
        f.write(animation_output)

    logging.info(f"Model audio answer saved to: output_audio.wav")
    logging.info(f"Model animation answer saved to: output_animation.fbx")

if __name__ == '__main__':
    main()