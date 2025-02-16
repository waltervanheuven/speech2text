# Speech2Text

Speech2Text provides a simple and easy-to-use graphical user interface (GUI) for different automatic speech recognition (ASR) systems and services based on [OpenAI's Whisper](https://openai.com/research/whisper): [whisper.cpp](https://github.com/ggerganov/whisper.cpp), [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), [faster-whisper](https://github.com/guillaumekln/faster-whisper), [Whisper ASR webservice](https://github.com/ahmetoner/whisper-asr-webservice), and the [Whisper API](https://openai.com/blog/introducing-chatgpt-and-whisper-apis). The application transcribes or translates the speech in audio and video files. The output is a text file or a subtitle file (.vtt or .srt). When you select openai-whisper, mlx-whisper, whisper.cpp, or faster-whisper, the ASR runs locally on your computer.

Please note that mlx-whisper (only available on Macs with an M1, M2, or later) and whisper.cpp are much faster than OpenAI's whisper. Speech2Text can also send the audio/video file to a remote computer running the whisper ASR webservice or use OpenAI's whisper API, which performs ASR on OpenAI's servers.

To achieve the best accuracy, select one of the 'large' models in the Settings (e.g. `large-v2` or `large-v3-turbo`).

## Download and install binaries

Binaries for macOS and Windows can be downloaded at [https://waltervanheuven.net/s2t/](https://waltervanheuven.net/s2t/)

## Run on macOS

Use [brew](https://brew.sh) to install latest Python and other apps.

```sh
brew install python@3.12
brew install uv
brew install ffmpeg
```

Clone speech2text.

```sh
git clone https://github.com/waltervanheuven/speech2text.git
cd speech2text
```

Set up venv and install packages using [uv](https://github.com/astral-sh/uv).

```sh
# venv
uv venv --python 3.12.9
source .venv/bin/activate

# install packages
uv pip install -U pip setuptools wheel
uv pip install -r requirements.txt
```

### Build and install whisper.cpp on macOS

```sh
# create folder for whisper.cpp
mkdir bin
mkdir bin/metal

# Further build instructions: https://github.com/ggerganov/whisper.cpp
mkdir tmp
cd tmp
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
cmake -B build
cmake --build build --config Release
cp build/bin/whisper-cli ../../bin/metal/whisper-cli
cd ../..
```

## Run on Windows

Use [scoop](https://scoop.sh) to install latest Python and other required apps

```powershell
scoop update
scoop bucket add versions
scoop install python312
scoop install main/uv
scoop install ffmpeg
```

Clone speech2text.

```powershell
git clone https://github.com/waltervanheuven/speech2text.git
cd speech2text
```

Set up venv and install packages using [uv](https://github.com/astral-sh/uv).

```powershell
uv venv --python 3.12.9
source .venv/bin/activate

uv pip install -U pip setuptools wheel
uv pip install -r requirements.txt
```

### Build and install whisper.cpp on Windows

```powershell
# create folder for whisper.cpp
mkdir bin
mkdir bin/cuda

# build instructions: https://github.com/ggerganov/whisper.cpp
# or download binaries and place `whisper-cli.exe` and `*.dll` in folder `bin`
```

## Start app in venv

```sh
python src/Speech2Text.py
```
