from dataset.data_classes import SupervisedDataset, DataCollatorForSupervisedDataset
from dataset.data_utils import get_preprocess_func
import logging

def create_dataset(tokenizer, image_processor, data_path, image_folder, image_aspect_ratio, is_multimodal, config):
    logging.info(f"Tokenizer in create_dataset: {type(tokenizer)}")
    preprocess_func = get_preprocess_func(config.text_model_id)
    logging.info(f"Preprocess function type: {type(preprocess_func)}")

    dataset = SupervisedDataset(
        data_path=data_path,
        image_folder=image_folder,
        image_processor=image_processor,
        tokenizer=tokenizer,
        preprocess_func=lambda x, y: preprocess_func(x, tokenizer, y),  # Correctly pass tokenizer here
        is_multimodal=is_multimodal
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer)

    return {
        'train_dataset': dataset,
        'data_collator': data_collator
    }
