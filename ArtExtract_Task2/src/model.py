import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
from torchvision.transforms.functional import to_pil_image
from torchvision.models import VGG16_Weights, ResNet50_Weights

class ResNetCompressor():
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', weights=ResNet50_Weights.DEFAULT)
        self.extractor = torch.nn.Sequential(*list(self.model.children())[:-2]).to(device)

    def extract(self, image_tensor):
        with torch.no_grad():
            features = self.extractor(image_tensor.to(self.device))
        return features.view(features.size(0), -1)

class FaceCropper():
    def __init__(self, compressor, device='cpu'):
        self.compressor = compressor
        self.device = device
        self.face_detector = MTCNN(min_face_size=2,margin=10,thresholds = [0.6, 0.6, 0.6],device = device,keep_all=True,post_process=True)
        self.face_detector.eval()

    def crop_faces(self, image_tensor):
        cropped_images = []
        for img in image_tensor:
            pil_img = to_pil_image(img)
            boxes, _ = self.face_detector.detect(pil_img)

            if boxes is not None and len(boxes) > 0:
                valid_boxes = []
                for box in boxes:
                    x_min, y_min, x_max, y_max = map(int, box)
                    x_min, y_min, x_max, y_max = [max(0, val) for val in [x_min, y_min, x_max, y_max]]

                    if x_max > x_min and y_max > y_min:
                        valid_boxes.append((x_min, y_min, x_max, y_max))

                if len(valid_boxes) > 0:
                    x_min = min(box[0] for box in valid_boxes)
                    y_min = min(box[1] for box in valid_boxes)
                    x_max = max(box[2] for box in valid_boxes)
                    y_max = max(box[3] for box in valid_boxes)

                    cropped_img = img[:, y_min:y_max, x_min:x_max]

                    resized_img = torch.nn.functional.interpolate(cropped_img.unsqueeze(0), size=(200, 200), mode='bilinear', align_corners=False)
                    cropped_images.append(resized_img.squeeze(0))
            else:
                x_min, y_min, x_max, y_max = 0, 0, 200, 200
                cropped_img = img[:, y_min:y_max, x_min:x_max]
                cropped_images.append(cropped_img)

        stacked_tensor = torch.stack(cropped_images).to(self.device)

        extracted_features = self.compressor.extract(stacked_tensor)

        return extracted_features