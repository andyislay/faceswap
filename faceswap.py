from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import io
import traceback
from deepface import DeepFace
import insightface
from gfpgan import GFPGANer
from skimage.exposure import match_histograms

app = Flask(__name__)

# Load InsightFace models
face_analyzer = insightface.app.FaceAnalysis(name='buffalo_l')
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
face_swapper = insightface.model_zoo.get_model('models/inswapper_128.onnx', providers=['CPUExecutionProvider'])

# Load GFPGAN model
gfpgan = GFPGANer(
    model_path='models/checkpoints/GFPGANv1.4.pth',
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)

def decode_image(file_bytes):
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to decode image")
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def expand_bbox(bbox, scale=1.6, image_shape=None):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w // 2, y1 + h // 2
    new_w, new_h = int(w * scale), int(h * scale)
    new_x1 = max(cx - new_w // 2, 0)
    new_y1 = max(cy - new_h // 2, 0)
    new_x2 = min(cx + new_w // 2, image_shape[1])
    new_y2 = min(cy + new_h // 2, image_shape[0])
    return np.array([new_x1, new_y1, new_x2, new_y2])

def create_feathered_mask(bbox, shape, feather=45):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    mask = cv2.GaussianBlur(mask, (feather, feather), 0)
    return mask

def blend_faces(original, swapped, mask):
    mask = mask.astype(np.float32) / 255.0
    mask = np.expand_dims(mask, axis=2)
    blended = swapped * mask + original * (1 - mask)
    return blended.astype(np.uint8)

@app.route('/swap', methods=['POST'])
def generate_card():
    try:
        print("Incoming size:", request.content_length)

        if 'source' not in request.files or 'target' not in request.files:
            return jsonify({'error': 'Missing source or target file'}), 400

        face_bytes = request.files['source'].read()
        template_bytes = request.files['target'].read()

        face_img = decode_image(face_bytes)
        template_img = decode_image(template_bytes)

        # Step 1: Estimate attributes
        analysis = DeepFace.analyze(face_img, actions=["age", "gender", "race"], enforce_detection=False)
        attributes = analysis[0]
        age = attributes["age"]
        gender = attributes["gender"]
        ethnicity = attributes["dominant_race"]

        print(f"Detected attributes - Age: {age}, Gender: {gender}, Ethnicity: {ethnicity}")

        # Step 2: Detect faces
        face_faces = face_analyzer.get(face_img)
        template_faces = face_analyzer.get(template_img)

        if not face_faces or not template_faces:
            return jsonify({'error': 'No faces detected in one of the images'}), 400

        # Step 3: Swap face
        swapped_img = face_swapper.get(template_img, template_faces[0], face_faces[0], paste_back=True)

        # Step 4: Expand bounding box and blend
        bbox = expand_bbox(template_faces[0].bbox.astype(int), scale=1.6, image_shape=template_img.shape)
        mask = create_feathered_mask(bbox, template_img.shape)
        blended_img = blend_faces(template_img, swapped_img, mask)

        # Step 5: Match skin tone
        blended_img = match_histograms(blended_img, template_img, channel_axis=-1)

        # Step 6: Enhance with GFPGAN
        _, _, enhanced_img = gfpgan.enhance(blended_img, has_aligned=False, only_center_face=False, paste_back=True)

        # Encode result image
        success, buffer = cv2.imencode('.jpg', enhanced_img)
        if not success:
            raise ValueError("Failed to encode result image")

        img_io = io.BytesIO(buffer)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7861, debug=True)
   