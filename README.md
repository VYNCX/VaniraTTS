# Vanira TTS

VaniraTTS โมเดล **Text-to-Speech (TTS)** สำหรับภาษาไทย  
สร้างเสียงพูดจากข้อความอย่างรวดเร็ว รองรับการใช้งานทั้ง **CPU** และ **GPU** ผ่าน `onnxruntime`  

- 🔥 สถาปัตยกรรม : [MixerTTS](https://arxiv.org/abs/2110.03584)  
- ⚡ โค้ดหลัก : [nipponjo/mixer-tts-pytorch](https://github.com/nipponjo/mixer-tts-pytorch)


## 🚀 เริ่มต้นใช้งาน  

### ติดตั้ง

```
pip install vaniratts
```

 ### การใช้งาน

```
from VaniraTTS import VaniraTTS

tts = VaniraTTS()
text = "สวัสดีครับ/ค่ะ นี่คือเสียงพูดภาษาไทย"

# เสียงพูดที่รองรับ 
# - 1: เสียงผู้หญิง 1
# - 2: เสียงผู้หญิง 2
# - 3: เสียงผู้ชาย 1
# - 4: เสียงผู้ชาย 2

tts.infer(text,
    speaker=1, # 1-4
    output="output.wav",
    volume=1.0,
    speed=1.0
)
```
