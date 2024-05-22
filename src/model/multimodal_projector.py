from torch import nn
from transformers.activations import ACT2FN

from model.configuration_llama import MultimodalLlamaConfig

class MultiModalLlamaProjector(nn.Module):
    def __init__(self, config: MultimodalLlamaConfig):
        super().__init__()

        # Projectors for different modalities
        self.vision_projector = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.audio_projector = nn.Linear(config.audio_config.hidden_size, config.text_config.hidden_size, bias=True) if config.audio_config else None
        self.animation_projector = nn.Linear(config.animation_config.hidden_size, config.text_config.hidden_size, bias=True) if config.animation_config else None

        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, features, modality):
        if modality == "vision":
            hidden_states = self.vision_projector(features)
        elif modality == "audio" and self.audio_projector:
            hidden_states = self.audio_projector(features)
        elif modality == "animation" and self.animation_projector:
            hidden_states = self.animation_projector(features)
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states