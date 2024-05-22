import tokenizers
import torch
import transformers
import logging
from packaging import version
from constants import IGNORE_INDEX, IMAGE_TOKEN, PAD_TOKEN
from llava_conversation_lib import conv_templates
from helper_functions import preprocess_audio, preprocess_animation

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

def tokenizer_image_token(prompt, tokenizer, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
    image_token_index = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def preprocess_llama_2(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False):
    conv = conv_templates["llama_2"].copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    tokenized = tokenizer(conversations, return_tensors='pt', padding=True, truncation=True)

    input_ids = tokenized.input_ids
    targets = input_ids.clone()

    # Masking for loss computation
    for target, source, conversation in zip(targets, sources, conversations):
        sep = conversation.split('\n')[0]  # Extract the separator from the conversation
        rounds = conversation.split(sep)[1:-1]
        cur_len = 0
        total_len = len(tokenizer(conversation).input_ids)
        logging.info(f"Total length of the conversation: {total_len}")
        for rou in rounds:
            parts = rou.split("ASSISTANT:")
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not getattr(tokenizer, "legacy", False):
                # The legacy and non-legacy modes handle special tokens differently
                round_len -= 1
                instruction_len -= 1

            logging.info(f"Round length: {round_len}, Instruction length: {instruction_len}")

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        logging.info(f"Current length after processing rounds: {cur_len}")

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                logging.warning(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_multimodal(sources, is_multimodal=True):
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(IMAGE_TOKEN, '').strip()
                sentence['value'] = IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
            replace_token = IMAGE_TOKEN
            sentence["value"] = sentence["value"].replace(IMAGE_TOKEN, replace_token)

    return sources

def get_preprocess_func(model_id):
    lower_model_id = model_id.lower()
    if "llama-3" in lower_model_id:
        return preprocess_llama_2
    elif "llama-2" in lower_model_id:
        return preprocess_llama_2
    else:
        return preprocess_multimodal
