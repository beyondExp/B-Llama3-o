import json
import os
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from helper_functions import preprocess_audio, preprocess_animation
import logging

IGNORE_INDEX = -100

class SupervisedDataset(Dataset):
    def __init__(self, data_path, image_folder, image_processor, tokenizer: PreTrainedTokenizer, preprocess_func, is_multimodal=True):
        logging.info(f"Tokenizer in SupervisedDataset: {type(tokenizer)}")
        self.data = json.load(open(data_path, "r"))
        self.image_folder = image_folder
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.preprocess_func = preprocess_func
        self.is_multimodal = is_multimodal

        self.conversations, self.targets = self.process_data(self.data)

    def process_data(self, data):
        sources = [item['conversations'] for item in data]
        sources = self.preprocess_func(sources, self.is_multimodal)
        targets = [item['label'] for item in data]
        return sources, targets

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        target = self.targets[idx]

        # Tokenize and process images, audio, and animation data if available
        input_ids = self.tokenizer(conversation, return_tensors='pt', truncation=True, padding=True).input_ids.squeeze()
        pixel_values = None
        audio_features = None
        animation_features = None

        if self.is_multimodal:
            image_path = os.path.join(self.image_folder, f"{idx}.jpg")
            if os.path.exists(image_path):
                pixel_values = self.image_processor(images=image_path, return_tensors="pt").pixel_values.squeeze()

            audio_path = os.path.join(self.image_folder, f"{idx}.wav")
            if os.path.exists(audio_path):
                audio_features = torch.tensor(preprocess_audio(audio_path))

            animation_path = os.path.join(self.image_folder, f"{idx}.fbx")
            if os.path.exists(animation_path):
                animation_features = torch.tensor(preprocess_animation(animation_path))

        return {
            'input_ids': input_ids,
            'pixel_values': pixel_values,
            'audio_features': audio_features,
            'animation_features': animation_features,
            'labels': torch.tensor(target)
        }

class DataCollatorForSupervisedDataset:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        batch = self.tokenizer.pad(
            examples,
            padding=True,
            return_tensors="pt",
        )

        pixel_values = torch.stack([example['pixel_values'] for example in examples if example['pixel_values'] is not None])
        audio_features = torch.stack([example['audio_features'] for example in examples if example['audio_features'] is not None])
        animation_features = torch.stack([example['animation_features'] for example in examples if example['animation_features'] is not None])

        labels = torch.stack([example['labels'] for example in examples])

        return {
            'input_ids': batch['input_ids'],
            'pixel_values': pixel_values,
            'audio_features': audio_features,
            'animation_features': animation_features,
            'labels': labels
        }
