[![Click to Watch the Video](https://img.youtube.com/vi/d00dspatedA/0.jpg)](https://www.youtube.com/watch?v=d00dspatedA)

# B-Llama3-o: A Multimodal LLaMA Model by B-Bot

B-Llama3-o is a multimodal Language Model Adaptation (LLaMA) developed by B-Bot. This model supports text, audio, and video inputs, and produces text, audio, and animation outputs. The repository includes data preprocessing scripts, dataset handling, and training scripts that leverage the transformers library.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
   - [Training](#training)
   - [Evaluation](#evaluation)
- [Data Preparation](#data-preparation)
- [Files and Directories](#files-and-directories)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/beyondExp/B-Llama3-o.git
    cd b-llama3o
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the pretrained models** (if needed):
    ```bash
    # Example commands to download pretrained models
    wget https://path-to-pretrained-model/llama-3-8B.zip
    unzip llama-3-8B.zip -d pretrained_models/llama-3-8B
    ```

## Usage

### Training

1. **Prepare your dataset**:
   - Ensure your data is in the appropriate format (JSONL) and place audio, video, text, and animation files in the specified directories.
   - Example structure:
     ```
     data/
     ├── input/
     │   ├── audio/
     │   │   ├── audio1.wav
     │   │   ├── audio2.wav
     │   │   └── ...
     │   ├── videos/
     │   │   ├── video1.mp4
     │   │   ├── video2.mp4
     │   │   └── ...
     │   └── data.jsonl
     ├── output/
     │   ├── audio/
     │   │   ├── audio1.wav
     │   │   ├── audio2.wav
     │   │   └── ...
     │   ├── animations/
     │   │   ├── animation1.fbx
     │   │   ├── animation2.fbx
     │   │   └── ...
     ```

2. **Run the training script**:
    ```bash
    python main.py
    ```

### Evaluation

1. **Evaluate the model**:
    ```bash
    python evaluation.py --model_path <path_to_trained_model>
    ```

## Data Preparation

### Data Format

Your dataset should be in JSON Lines (JSONL) format and structured to support multimodal inputs and outputs. Each entry should include text, audio, and video inputs, and optionally include reasoning and the expected output format. Here is an example structure for the JSONL file:

```
{"conversations":[{"from":"human","value":"<video>\nDescribe what is happening in this video."},{"from":"gpt","value":"The video shows a person performing a complex dance routine with high energy and precision."}],"video":"example_video.mp4","audio":"example_audio.wav","reasoning":"The model is analyzing the dance routine shown in the video.","output_audio":"output_audio.wav","output_animation":"output_animation.fbx"}
...
```

### Directories

Ensure your data directories are structured as follows:

```
data/
├── input/
│   ├── audio/
│   │   ├── example_audio.wav
│   │   └── ...
│   ├── videos/
│   │   ├── example_video.mp4
│   │   └── ...
│   └── data.jsonl
├── output/
│   ├── audio/
│   │   ├── output_audio.wav
│   │   └── ...
│   ├── animations/
│   │   ├── output_animation.fbx
│   │   └── ...
```

### Reasoning

The dataset should include a reasoning field if applicable. This field provides insights into the model's thought process or decision-making steps during training.

### Preprocessing

The preprocessing scripts handle tokenization, feature extraction, and formatting of text, audio, and video data to ensure compatibility with the B-Llama3o model.

## Files and Directories

- `main.py`: Main script to fine-tune the B-Llama3o model.
- `training.py`: Script containing functions for fine-tuning and evaluating the model.
- `data_handling.py`: Contains functions for creating datasets and data collators.
- `data_utils.py`: Utility functions for preprocessing data.
- `data_classes.py`: Defines dataset and data collator classes.
- `llava_conversation_lib.py`: Manages conversation templates and generates prompts.
- `constants.py`: Contains constants used across the project.
- `trainer_llama.py`: Custom trainer class for the B-Llama3o model.
- `src/`: Directory containing model configurations and processing scripts.
- `requirements.txt`: Lists required Python packages.

## Contributing

We welcome contributions! Please fork the repository and submit pull requests for any enhancements or bug fixes. Make sure to follow the coding style and include appropriate tests.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
