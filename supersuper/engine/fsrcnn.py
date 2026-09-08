from PIL import Image 
import numpy as np 
import onnxruntime as ort 

class EngineFSRCNN:
    def __init__(self):
        self.session = ort.InferenceSession(
            "models/fsrcnn.onnx",
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def execute(self,
                input_image,
                output_image,
                ):
        image = Image.open(input_image).convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = np.transpose(image_np, (2, 0, 1))
        image_np = np.expand_dims(image_np, axis=0)
        image_np = image_np.astype(np.float32)

        output = self.session.run(
                    [self.output_name],
                    {
                        self.input_name: image_np
                    }
                )[0]
        output = np.squeeze(output, axis=0)
        output = np.transpose(output, (1, 2, 0))
        output = np.clip(output, 0, 1)
        output = (output * 255.0).astype(np.uint8)
        output_image_pil = Image.fromarray(output)
        output_image_pil.save(output_image)
        
        return {
                    "status": "success",
                    "output_path": output_image
                }