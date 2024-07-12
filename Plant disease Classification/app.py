from flask import Flask, request, render_template, redirect, url_for
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the trained model
model = torch.load('models/plant_disease_classification.pth', map_location=torch.device('cpu'))
model.eval()

# Define the image transformations
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define a mapping from class indices to class names
class_names = [
    "Apple_scab", "Apple_black_rot", "Apple_cedar_apple_rust", "Apple_healthy",
    "Background_without_leaves", "Blueberry_healthy", "Cherry_powdery_mildew",
    "Cherry_healthy", "Corn_gray_leaf_spot", "Corn_common_rust", "Corn_northern_leaf_blight",
    "Corn_healthy", "Grape_black_rot", "Grape_black_measles", "Grape_leaf_blight",
    "Grape_healthy", "Orange_haunglongbing", "Peach_bacterial_spot", "Peach_healthy",
    "Pepper_bacterial_spot", "Pepper_healthy", "Potato_early_blight", "Potato_healthy",
    "Potato_late_blight", "Raspberry_healthy", "Soybean_healthy", "Squash_powdery_mildew",
    "Strawberry_healthy", "Strawberry_leaf_scorch", "Tomato_bacterial_spot",
    "Tomato_early_blight", "Tomato_healthy", "Tomato_late_blight", "Tomato_leaf_mold",
    "Tomato_septoria_leaf_spot", "Tomato_spider_mites_two-spotted_spider_mite",
    "Tomato_target_spot", "Tomato_mosaic_virus", "Tomato_yellow_leaf_curl_virus"
]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            img = Image.open(io.BytesIO(file.read()))
            img_tensor = data_transforms(img).unsqueeze(0)  # Apply transformations and add batch dimension
            outputs = model(img_tensor)
            _, preds = torch.max(outputs, 1)
            predicted_class = class_names[preds[0]]
            img_stream = io.BytesIO()
            img = transforms.ToPILImage()(img_tensor.squeeze(0).cpu())  # Convert tensor back to PIL image
            img.save(img_stream, format='JPEG')
            img_stream.seek(0)
            img_data = base64.b64encode(img_stream.getvalue()).decode()
            return render_template('result.html', class_name=predicted_class, img_data=img_data)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
