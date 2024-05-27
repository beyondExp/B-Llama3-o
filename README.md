[![Click to Watch the Video](https://img.youtube.com/vi/d00dspatedA/0.jpg)](https://www.youtube.com/watch?v=d00dspatedA)

# 🎉 B-Llama3-o: A Multimodal LLaMA Model by B-Bot 🎉

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

## 📋 Table of Contents


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

## 💻 Usage


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
     │   └── ...
     ├── output/
     │   ├── audio/
     │   │   ├── audio1.wav
     │   │   ├── audio2.wav
     │   │   └── ...
     │   ├── animations/
     │   │   ├── animation1.fbx
     │   │   ├── animation2.fbx
     │   │   └── ...
     │   └── ...
     └── data.jsonl
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

## 📂 Data Preparation


### Data Format

[![Click to Watch the Video](https://i9.ytimg.com/vi/DaYp_by0rqA/mq1.jpg?sqp=CJTK07IG-oaymwEmCMACELQB8quKqQMa8AEB-AHUBoAC4AOKAgwIABABGCcgGih_MA8=&rs=AOn4CLD4tN0FzyEZIA2GTvfNg-RHEMsZLA)](https://www.youtube.com/watch?v=DaYp_by0rqA)


Your dataset should be in JSON Lines (JSONL) format and structured to support multimodal inputs and outputs. Each entry should include text, audio, and video inputs, and optionally include reasoning and the expected output format. Here is an example structure for the JSONL file:

### Example 1: Only Text

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "What is the capital of France?"
    },
    {
      "from": "gpt",
      "value": "The capital of France is Paris."
    }
  ],
  "reasoning": "The model is providing a direct answer to a factual question."
}
```

### Example 2: Text and Video

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "What is happening in this video?",
      "video": "street_performance.mp4"
    },
    {
      "from": "gpt",
      "value": "The video shows a street performer juggling flaming torches."
    }
  ],
  "reasoning": "The model is describing the actions visible in the provided video."
}
```
### Example 3: Text and Audio
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "What does the sound in this audio clip represent?",
      "audio": "bird_song.wav"
    },
    {
      "from": "gpt",
      "value": "The audio clip features the song of a nightingale."
    }
  ],
  "reasoning": "The model is identifying the source of the sounds in the provided audio clip."
}
```
### Example 4: Text, Video, and Audio
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Describe what you see and hear in this video.",
      "video": "concert.mp4",
      "audio": "concert_audio.wav"
    },
    {
      "from": "gpt",
      "value": "The video shows a live concert where the band is playing rock music. The crowd is cheering and clapping along with the music."
    }
  ],
  "reasoning": "The model is providing a description of both the visual and auditory elements in the video."
}
```
### Example 5: Text with Output Audio
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Can you tell me a joke?"
    },
    {
      "from": "gpt",
      "value": "Why don't scientists trust atoms? Because they make up everything!",
      "output_audio": "joke_audio.wav"
    }
  ],
  "reasoning": "The model is generating a text joke and also providing an audio version of it."
}
```
### Example 6: Text with Output Animation
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Describe how to perform a push-up."
    },
    {
      "from": "gpt",
      "value": "To perform a push-up, start in a plank position with your hands shoulder-width apart. Lower your body until your chest nearly touches the floor, then push yourself back up to the starting position.",
      "output_animation": "push_up_animation.fbx"
    }
  ],
  "reasoning": "The model is describing the steps to perform a push-up and providing an animation to illustrate the movement."
}
```
### Example 7: Text, Video, and Output Audio
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Can you describe the wildlife in this video and also provide an audio summary?",
      "video": "wildlife_clip.mp4"
    },
    {
      "from": "gpt",
      "value": "The video shows a variety of wildlife including deer, birds, and rabbits in a forest setting.",
      "output_audio": "wildlife_summary_audio.wav"
    }
  ],
  "reasoning": "The model is providing a textual description of the wildlife in the video and an audio summary."
}
```
### Example 8: Text, Audio, and Output Animation
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "What do you think about the speech in this audio clip?",
      "audio": "motivational_speech.wav"
    },
    {
      "from": "gpt",
      "value": "The speech is very motivational, encouraging people to pursue their dreams and never give up.",
      "output_animation": "motivational_speech_animation.fbx"
    }
  ],
  "reasoning": "The model is analyzing the content of the speech and providing an animation that represents the motivational theme."
}
```
### Example 9: Only Text with Reasoning
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "What is the boiling point of water?"
    },
    {
      "from": "gpt",
      "value": "The boiling point of water is 100 degrees Celsius at standard atmospheric pressure."
    }
  ],
  "reasoning": "The model is providing a factual answer to a straightforward scientific question."
}
```
### Example 10: Text, Video, and Output Animation
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Can you explain the dance steps shown in this video?",
      "video": "dance_steps.mp4"
    },
    {
      "from": "gpt",
      "value": "The dance steps involve a sequence of quick foot movements followed by a spin and a jump.",
      "output_animation": "dance_steps_animation.fbx"
    }
  ],
  "reasoning": "The model is explaining the dance steps visible in the video and providing an animation for better understanding."
}
```
These examples cover a range of scenarios, demonstrating how to handle different combinations of text, audio, video, and animations within the JSONL structure.


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

## 📁 Files and Directories


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

## 🤝 Contributing

We welcome contributions! Please fork the repository and submit pull requests for any enhancements or bug fixes. Make sure to follow the coding style and include appropriate tests.

## 📜 License


This project is licensed under the GPLv3 License. See the LICENSE here: [link](https://www.gnu.org/licenses/gpl-3.0.de.html) for details.


### Thanks to
    
```
https://github.com/AdrianBZG/llama-multimodal-vqa
For providing the base code for the multimodal model tuning. 
Without you this idea would not have been possible.
```
