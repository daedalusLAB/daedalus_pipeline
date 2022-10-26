# Pipeline

Pipeline is a simple and easy tools to createvertical files and corpus from video files.

## Installation

### Requirements

#### **ffmpeg**


Requires the command-line tool [`ffmpeg`](https://ffmpeg.org/) to be installed on your system, which is available from most package managers:

```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

#### **Spacy models**
Requires spacy models:
<code>
python -m spacy download en_core_web_lg
</code>

#### **Some python packages**
<code >
  pip install -r requirements.txt
</code>


#### **GENTLE**

Requires to install gentle from [here](https://lowerquality.com/gentle/) and be in your path.



## Usage

<code >
  python pipeline.py --file <input_file>
</code>



## External tools
 - [Whisper](https://github.com/openai/whisper): is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification.
 - [stable_whisper](https://github.com/jianfch/stable-ts):  modifies methods of Whisper's model to gain access to the predicted timestamp tokens of each word (token) without needing additional inference. It also stabilizes the timestamps down to the word (token) level to ensure chronology.
  - [Spacy] (https://spacy.io/): is a free, open-source library for Natural Language Processing in Python. It features NER, POS tagging, dependency parsing, word vectors and more.

