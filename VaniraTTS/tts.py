import json
import numpy as np
import soundfile as sf
import onnxruntime as ort
from vachana_g2p import th2ipa
from vaniratts.text.eng2tha import transliterator

class VaniraTTS:
    def __init__(self, model_id:str = "VIZINTZOR/VaniraTTS", local_path:str = None, device: str = "cpu"):

        providers = self._get_providers(device) 

        if local_path == None:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id=model_id, filename="tts.onnx")
            vocab_path = hf_hub_download(repo_id=model_id, filename="vocab.json")
        else:
            import os
            model_path = os.path.join(local_path, "tts.onnx")
            vocab_path = os.path.join(local_path, "vocab.json")

        self.ort_session = ort.InferenceSession(model_path, providers=providers)
        
        with open(vocab_path, 'r') as file:
            self.symbols_to_id = json.load(file)

    def _get_providers(self, device: str):
        if device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def text_to_ids(self, text: str):
        phonemes_ipa = " " + th2ipa(text) + " "
        phonemes_ids = [self.symbols_to_id[s] for s in phonemes_ipa if s in self.symbols_to_id]
        return phonemes_ids

    def infer(self, text:str = "", speed: float = 1.0, speaker: int = 1, pitch_mul: float = 1.0, volume: float = 1.0, output: str = "tts.wav"):

        clean_text = transliterator(text)
        ids_batch = np.array([self.text_to_ids(clean_text)], dtype=np.int64)
        speed = max(0.3, min(speed, 2.0))
        speaker = max(1, min(speaker, 4)) - 1
        volume = max(0.1, min(volume, 1.5))
        
        inputs = {
            "token_ids": ids_batch, 
            "pace": np.array([speed], dtype=np.float32),  
            "speaker": np.array([speaker], dtype=np.int32),      
            "emotion": np.array([0], dtype=np.int32),       
            "pitch_mul": np.array([pitch_mul], dtype=np.float32),
            "pitch_add": np.array([0.0], dtype=np.float32),
        }

        mel_outputs = self.ort_session.run(None, inputs)

        audio = np.squeeze(mel_outputs[0]).astype(np.float32)
        peak = np.maximum(np.abs(audio).max(), 1e-6)
        audio = audio / (peak / volume)
        sample_rate = 44100
        sf.write(output, audio, sample_rate)

        print(f"Audio saved successfully to {output}!")
