import ChatTTS
import os
from IPython.display import Audio
import torch
import torchaudio
import json

# Function to get text from a file
def getTextCase(fileName):
    with open(fileName, 'r') as file:
        data = file.read().replace('\n', "")
    return data


# Function to generate audio from text
def generateAudio(text, local_path="./chatTTS", speaker="girl", output_name="girl_chinese1"):
    chat = ChatTTS.Chat()
    print("Loading models...")
    config_path = os.path.join(local_path, 'config', 'path.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    chat.load_models(source="local", local_path=local_path)
    print("Models loaded!")

    output_path = "./output"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    spk_vector = json.load(open("speaker.json", "r"))
    sok_vector = spk_vector[speaker]
    speaker_emb = torch.tensor([float(x) for x in spk_vector.split(",")])
    texts = [text, ]
    params_infer_code = {
        "spk_emb": speaker_emb
    }
    params_refine_text = {
        "prompt": "[oral_0][laugh_0][break_2]"
    }
    wavs = chat.infer(texts,
                      params_refine_text=params_refine_text,
                      params_infer_code=params_infer_code
                      )
    torchaudio.save(f"{output_path}/{output_name}_break2.wav", torch.from_numpy(wavs[0]), 24000)

    params_refine_text = {
        "prompt": "[oral_0][laugh_0][break_6]"
    }
    wavs = chat.infer(texts,
                      params_refine_text=params_refine_text,
                      params_infer_code=params_infer_code
                      )
    torchaudio.save(f"{output_path}/{output_name}_break6.wav", torch.from_numpy(wavs[0]), 24000)


if __name__ == "__main__":
    speakers = ["girl", "boy"]
    # 找到testcase文件夹下的所有文件
    testCases = os.listdir("testcase")
    for testCase in testCases:
        text = getTextCase(f"testcase/{testCase}")
        for speaker in speakers:
            testCase = testCase.split(".")[0]
            generateAudio(text, speaker=speaker, output_name=f"{speaker}_{testCase}")
            print(f"Audio for {speaker} generated for {testCase}")
