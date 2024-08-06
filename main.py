import ChatTTS
import os
from IPython.display import Audio
import torch
import torchaudio
import json
def generateAudio(text, local_path="./chatTTS", output_name="output.wav"):
    chat = ChatTTS.Chat()
    print("Loading models...")
    config_path = os.path.join(local_path, 'config', 'path.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    chat.load_models(source="local", local_path=local_path)
    print("Models loaded!")

    spk_vector = json.load(open("speaker.json", "r"))
    girl_vector = spk_vector["girl"]
    speaker_emb = torch.tensor([float(x) for x in spk_vector.split(",")])
    texts = [text, ]
    params_infer_code = {
        "use_decoder": True,
        "speaker_emb": speaker_emb
    }
    wavs = chat.infer_waveform(texts, params_infer_code=params_infer_code)

    torchaudio.save("output.wav", torch.from_numpy(wavs[0]), 24000)

generateAudio("Hello, my name is ChatTTS. I am a text-to-speech model.")