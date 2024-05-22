import torch
from transformers import AutoImageProcessor, AutoTokenizer, Wav2Vec2Processor, CLIPFeatureExtractor
from transformers import AutoConfig, AutoModel

from utils.constants import IMAGE_TOKEN, PAD_TOKEN, LORA_CONFIG
from model.configuration_llama import MultimodalLlamaConfig
from model.modeling_llama import MultimodalLlamaForConditionalGeneration
from processing_llama import MultiModalLlamaProcessor

def build_model(text_model_id,
                vision_model_id,
                audio_model_id=None,
                animation_model_id=None,
                freeze_multimodal_projector=False,
                freeze_language_model=False,
                freeze_vision_model=False,
                freeze_audio_model=False,
                freeze_animation_model=False,
                device="cuda",
                use_bfloat16=True,
                load_in_4bit=False):
    """
    Build model and related components.
    """
    tokenizer = AutoTokenizer.from_pretrained(text_model_id)

    tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})

    if tokenizer.pad_token is None or tokenizer.pad_token != PAD_TOKEN:
        tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
        tokenizer.pad_token = PAD_TOKEN

    tokenizer_len = len(tokenizer)

    multimodal_llama_config = MultimodalLlamaConfig(
        vision_model_id=vision_model_id,
        text_model_id=text_model_id,
        audio_model_id=audio_model_id,
        animation_model_id=animation_model_id,
        tokenizer_len=tokenizer_len,
        lora_config=LORA_CONFIG,
        freeze_multimodal_projector=freeze_multimodal_projector,
        freeze_language_model=freeze_language_model,
        freeze_vision_model=freeze_vision_model,
        load_in_4bit=load_in_4bit
    )

    # Image processor
    image_processor = AutoImageProcessor.from_pretrained(vision_model_id) if vision_model_id else None
    audio_processor = Wav2Vec2Processor.from_pretrained(audio_model_id) if audio_model_id else None
    animation_processor = CLIPFeatureExtractor.from_pretrained(animation_model_id) if animation_model_id else None

    processor = MultiModalLlamaProcessor(
        image_processor=image_processor,
        audio_processor=audio_processor,
        animation_processor=animation_processor,
        tokenizer=tokenizer
    )

    # Language model
    multimodal_llama_model = MultimodalLlamaForConditionalGeneration(multimodal_llama_config).to(device)

    if use_bfloat16:
        multimodal_llama_model = multimodal_llama_model.to(torch.bfloat16)

    # Freeze additional models if specified
    if freeze_audio_model and audio_model_id:
        audio_config = AutoConfig.from_pretrained(audio_model_id)
        audio_model = AutoModel.from_config(audio_config)
        for param in audio_model.parameters():
            param.requires_grad = False

    if freeze_animation_model and animation_model_id:
        animation_config = AutoConfig.from_pretrained(animation_model_id)
        animation_model = AutoModel.from_config(animation_config)
        for param in animation_model.parameters():
            param.requires_grad = False

    return dict(
        tokenizer=tokenizer,
        model=multimodal_llama_model,
        processor=processor,
        config=multimodal_llama_config
    )
