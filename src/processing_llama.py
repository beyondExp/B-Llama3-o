from typing import List, Optional, Union
import torch
from transformers import ProcessorMixin, TensorType
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput, TruncationStrategy
from transformers.utils import PaddingStrategy

class MultiModalLlamaProcessor(ProcessorMixin):
    r"""
    Constructs a MultiModalLlama processor which wraps an image processor, audio processor, animation processor, and a tokenizer into a single processor.

    [`MultiModalLlamaProcessor`] offers all the functionalities of [`CLIPImageProcessor`], [`Wav2Vec2Processor`], and [`LlamaTokenizerFast`]. See the
    [`~MultiModalLlamaProcessor.__call__`] and [`~MultiModalLlamaProcessor.decode`] for more information.

    Args:
        image_processor ([`CLIPImageProcessor`], *optional*):
            The image processor is a required input.
        audio_processor ([`Wav2Vec2Processor`], *optional*):
            The audio processor is an optional input.
        animation_processor ([`CLIPFeatureExtractor`], *optional*):
            The animation processor is an optional input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "audio_processor", "animation_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    audio_processor_class = "Wav2Vec2Processor"
    animation_processor_class = "CLIPFeatureExtractor"
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast", "PreTrainedTokenizerFast", "PreTrainedTokenizer")

    def __init__(self, image_processor=None, audio_processor=None, animation_processor=None, tokenizer=None):
        self.image_processor = image_processor
        self.audio_processor = audio_processor
        self.animation_processor = animation_processor
        self.tokenizer = tokenizer

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        audio: Optional[torch.Tensor] = None,
        animations: Optional[torch.Tensor] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwargs` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the docstring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            audio (`torch.Tensor`, `List[torch.Tensor]`):
                The audio or batch of audio to be prepared.
            animations (`torch.Tensor`, `List[torch.Tensor]`):
                The animation or batch of animation to be prepared.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **audio_values** -- Audio values to be fed to a model. Returned when `audio` is not `None`.
            - **animation_values** -- Animation values to be fed to a model. Returned when `animations` is not `None`.
        """
        if images is not None:
            pixel_values = self.image_processor(images, return_tensors=return_tensors)["pixel_values"]
        else:
            pixel_values = None
        if audio is not None:
            audio_values = self.audio_processor(audio, return_tensors=return_tensors)["input_values"]
        else:
            audio_values = None
        if animations is not None and self.animation_processor is not None:
            animation_values = self.animation_processor(animations, return_tensors=return_tensors)["pixel_values"]
        else:
            animation_values = None

        text_inputs = self.tokenizer(
            text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
        )

        data = {**text_inputs}
        if pixel_values is not None:
            data["pixel_values"] = pixel_values
        if audio_values is not None:
            data["audio_values"] = audio_values
        if animation_values is not None:
            data["animation_values"] = animation_values

        return BatchFeature(data=data)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names if self.image_processor is not None else []
        audio_processor_input_names = self.audio_processor.model_input_names if self.audio_processor is not None else []
        animation_processor_input_names = self.animation_processor.model_input_names if self.animation_processor is not None else []
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names + audio_processor_input_names + animation_processor_input_names))
