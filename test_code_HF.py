from transformers import MgpstrProcessor, MgpstrForSceneTextRecognition
import requests
from PIL import Image

processor = MgpstrProcessor.from_pretrained('alibaba-damo/mgp-str-base')
model = MgpstrForSceneTextRecognition.from_pretrained('alibaba-damo/mgp-str-base')

# load image from the IIIT-5k dataset
url = "car1.jpeg"
image = Image.open(url).convert("RGB")

pixel_values = processor(images=image, return_tensors="pt").pixel_values
outputs = model(pixel_values)

generated_text = processor.batch_decode(outputs.logits)['generated_text']
print(generated_text)