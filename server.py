# server.py
# Server Flask untuk prediksi multi-head captcha yang disesuaikan untuk Hugging Face Spaces.

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2
import os
import json
import base64
from io import BytesIO
import sys
import logging

# ==============================================================================
#  BAGIAN 1: PENGATURAN DASAR FLASK & LOGGING
# ==============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
app = Flask(__name__)
CORS(app)

MODEL = None
MAPPINGS = None
DEVICE = None
TRANSFORMS = None

# ==============================================================================
#  BAGIAN 2: DEFINISI MODEL (TIDAK ADA PERUBAHAN)
# ==============================================================================

class TextHeadCTC(nn.Module):
    def __init__(self, input_dim, hidden_dim, ctc_vocab_size):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, ctc_vocab_size)
    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        output_logits = self.fc(rnn_out)
        return nn.functional.log_softmax(output_logits, dim=2).permute(1, 0, 2)

class MultiHeadModel(nn.Module):
    def __init__(self, backbone_name, ctc_vocab_size, num_object_classes, num_types):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            drop_path_rate=0.1
        )
        backbone_features_dim = self.backbone.num_features
        rnn_hidden_dim, projected_embed_dim = 256, 256
        self.type_head = nn.Linear(backbone_features_dim, num_types)
        self.object_head = nn.Linear(backbone_features_dim, num_object_classes)
        self.input_proj = nn.Conv2d(backbone_features_dim, projected_embed_dim, kernel_size=1)
        self.text_head_ctc = TextHeadCTC(projected_embed_dim, rnn_hidden_dim, ctc_vocab_size)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        features = self.backbone.forward_features(x)
        pooled_features = self.pool(features).flatten(1)
        type_logits = self.type_head(pooled_features)
        object_logits = self.object_head(pooled_features)
        proj_features = self.input_proj(features)
        bs, c_proj, h_feat, w_feat = proj_features.size()
        image_features_seq = proj_features.view(bs, c_proj, h_feat * w_feat).permute(0, 2, 1)
        text_log_probs = self.text_head_ctc(image_features_seq)
        return type_logits, object_logits, text_log_probs

# ==============================================================================
#  BAGIAN 3: FUNGSI HELPER (TIDAK ADA PERUBAHAN)
# ==============================================================================

def get_transforms(img_height, img_width):
    interpolation_method = cv2.INTER_AREA
    return A.Compose([
        A.Resize(height=img_height, width=img_width, interpolation=interpolation_method),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def ctc_decoder_with_confidence(log_probs, idx_to_char_map, blank_idx):
    probs = torch.exp(log_probs)
    max_probs, pred_indices = torch.max(probs, dim=-1)
    max_probs = max_probs.squeeze(1).cpu().numpy()
    pred_indices = pred_indices.squeeze(1).cpu().numpy()
    
    decoded_sequence = []
    confidence_values = []
    last_idx = -1

    for i, idx in enumerate(pred_indices):
        if idx == blank_idx or idx == last_idx:
            last_idx = blank_idx if idx == blank_idx else last_idx
            continue
        
        decoded_sequence.append(idx_to_char_map.get(str(idx), '?'))
        confidence_values.append(max_probs[i])
        last_idx = idx
    
    final_text = "".join(decoded_sequence)
    avg_confidence = np.mean(confidence_values) if confidence_values else 0.0
    
    return final_text, avg_confidence

# ==============================================================================
#  BAGIAN 4: INISIALISASI SERVER (TIDAK ADA PERUBAHAN)
# ==============================================================================

def initialize_server(model_path, mappings_path):
    global MODEL, MAPPINGS, DEVICE, TRANSFORMS
    logging.info("Memulai inisialisasi server...")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Menggunakan device: {DEVICE}")

    try:
        if not os.path.exists(mappings_path):
            raise FileNotFoundError(f"File mappings tidak ditemukan di: {mappings_path}")
        with open(mappings_path, 'r', encoding='utf-8') as f:
            MAPPINGS = json.load(f)
        logging.info("File mappings berhasil dimuat.")
    except Exception as e:
        logging.error(f"FATAL: Gagal memuat file mappings: {e}")
        sys.exit(1)

    TRANSFORMS = get_transforms(MAPPINGS['img_height'], MAPPINGS['img_width'])
    logging.info(f"Pipeline transformasi diatur untuk input ukuran: {MAPPINGS['img_width']}x{MAPPINGS['img_height']}")

    try:
        m = MAPPINGS
        num_obj_classes = len(m['object_to_idx']) if m.get('object_to_idx') else 1
        ctc_vocab_size = len(m['ctc_char_to_idx'])
        num_types_val = len(m['type_to_idx'])
        
        logging.info(f"Parameter Model: backbone='{m['backbone']}', ctc_vocab={ctc_vocab_size}, obj_classes={num_obj_classes}, types={num_types_val}")
        
        MODEL = MultiHeadModel(
            backbone_name=m['backbone'],
            ctc_vocab_size=ctc_vocab_size,
            num_object_classes=num_obj_classes,
            num_types=num_types_val
        )
        logging.info(f"Instance model '{MAPPINGS['backbone']}' berhasil dibuat.")
    except Exception as e:
        logging.error(f"FATAL: Gagal membuat instance model. Error: {e}")
        sys.exit(1)
        
    try:
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"File model tidak ditemukan di: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        saved_state_dict = checkpoint.get('model_state_dict', checkpoint)
        new_state_dict = {key.replace('_orig_mod.', ''): value for key, value in saved_state_dict.items()}
        
        MODEL.load_state_dict(new_state_dict)
        MODEL.to(DEVICE)
        MODEL.eval()
        logging.info("Model weights berhasil dimuat dan siap digunakan.")

    except Exception as e:
        logging.error(f"FATAL: Gagal memuat model weights. Error: {e}")
        sys.exit(1)
    
    logging.info("Inisialisasi server selesai. Siap menerima permintaan.")

# ==============================================================================
#  BAGIAN 5: ENDPOINT FLASK (TIDAK ADA PERUBAHAN)
# ==============================================================================
@app.route('/', methods=['GET'])
def home():
    """Endpoint dasar untuk memeriksa apakah server berjalan."""
    return "<h1>Captcha Prediction Server is running.</h1><p>Gunakan endpoint /predict untuk melakukan prediksi.</p>", 200

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """Endpoint untuk menerima gambar base64 dan mengembalikan prediksi."""
    if not request.is_json:
        return jsonify({"error": "Request harus berupa JSON"}), 400
    
    data = request.get_json()
    base64_string = data.get('image_base64')

    if not base64_string:
        return jsonify({"error": "Key 'image_base64' tidak ditemukan atau kosong"}), 400

    try:
        if ',' in base64_string:
            _, encoded = base64_string.split(',', 1)
        else:
            encoded = base64_string
        image_data = base64.b64decode(encoded)
        
        img_pil_original = Image.open(BytesIO(image_data))

        if img_pil_original.mode == 'RGBA' or 'A' in img_pil_original.info.get('transparency', ()):
            background = Image.new("RGB", img_pil_original.size, (255, 255, 255))
            background.paste(img_pil_original, mask=img_pil_original.split()[3])
            img_pil = background
        else:
            img_pil = img_pil_original.convert("RGB")
            
    except (base64.binascii.Error, UnidentifiedImageError) as e:
        logging.error(f"Error memproses gambar base64: {e}")
        return jsonify({"error": f"Data base64 tidak valid atau format gambar tidak didukung."}), 400
    except Exception as e:
        logging.error(f"Error tak terduga saat memproses gambar: {e}")
        return jsonify({"error": "Gagal memproses gambar."}), 500

    try:
        image_rgb = np.array(img_pil)
        img_tensor = TRANSFORMS(image=image_rgb)['image'].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            type_logits, object_logits, text_log_probs = MODEL(img_tensor)

        type_prob = torch.softmax(type_logits, dim=1)
        type_conf, type_pred_idx = torch.max(type_prob, dim=1)
        pred_type = MAPPINGS['idx_to_type'].get(str(type_pred_idx.item()), 'Tipe Tidak Dikenal')
        
        response = {
            "predicted_type": pred_type,
            "type_confidence": f"{type_conf.item():.2%}",
            "prediction": None,
            "prediction_confidence": None,
            "error": None
        }

        if pred_type == 'object':
            obj_prob = torch.softmax(object_logits, dim=1)
            obj_conf, obj_pred_idx = torch.max(obj_prob, dim=1)
            pred_obj = MAPPINGS['idx_to_object'].get(str(obj_pred_idx.item()), 'Objek Tidak Dikenal')
            response["prediction"] = pred_obj
            response["prediction_confidence"] = f"{obj_conf.item():.2%}"
        elif pred_type == 'text':
            pred_text, confidence = ctc_decoder_with_confidence(text_log_probs, MAPPINGS['ctc_idx_to_char'], MAPPINGS['ctc_blank_idx'])
            response["prediction"] = pred_text
            response["prediction_confidence"] = f"{confidence:.2%}"
        
        logging.info(f"Prediksi berhasil: Tipe='{response['predicted_type']}', Hasil='{response['prediction']}', Conf='{response['prediction_confidence']}'")
        return jsonify(response), 200

    except Exception as e:
        logging.error(f"Error saat inferensi model: {e}", exc_info=True)
        return jsonify({"error": "Terjadi kesalahan pada server saat melakukan prediksi."}), 500

# ==============================================================================
#  BAGIAN 6: MENJALANKAN SERVER (UNTUK HUGGING FACE SPACES)
# ==============================================================================

# Path file akan relatif terhadap root repositori di Hugging Face Spaces
MODEL_FILE_PATH = "best_model.pth"
MAPPINGS_FILE_PATH = "mappings.json"

# Inisialisasi server saat aplikasi dimulai
initialize_server(MODEL_FILE_PATH, MAPPINGS_FILE_PATH)

# Bagian if __name__ == '__main__' tidak akan dieksekusi saat dijalankan dengan Gunicorn di cloud,
# namun tetap berguna untuk pengujian lokal.
if __name__ == '__main__':
    # Untuk pengujian lokal, jalankan seperti biasa
    app.run(host='0.0.0.0', port=5111, debug=True)
