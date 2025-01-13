import onnxruntime as ort
import os
import numpy as np
from PIL import Image

folders = [f"data/{f.name}" for f in os.scandir('data') if f.is_dir()]
print(folders)

# Test vae decoder

options_our = ort.SessionOptions()
#options_our.add_session_config_entry("session.disable_cpu_ep_fallback", "0")
options_our.add_session_config_entry("ep.context_enable", "1")
options_our.log_severity_level = 1

vae_decoder = ort.InferenceSession("./vaedecoder.onnx",#.qnn.onnx_ctx
                                    sess_options=options_our,
                                    #providers=["QNNExecutionProvider"],
                                    provider_options=[{"backend_path": "QnnHtp.dll"}])

for f in folders:
    latent_dq = np.fromfile(f + '/latent.raw', dtype=np.float32).reshape(1, 4, 64, 64)
    outputs = vae_decoder.run(None, { 'latent': latent_dq})
    output_image = outputs[0].transpose(0, 2, 3, 1)
    output_image = (output_image / 2 + 0.5)
    image_size = 512
    output_image = np.clip(output_image * 255.0, 0.0, 255.0).astype(np.uint8)
    output_image = output_image.reshape(image_size, image_size, -1)
    image = Image.fromarray(output_image, mode="RGB")  # .save(image_path)
    image.show()
