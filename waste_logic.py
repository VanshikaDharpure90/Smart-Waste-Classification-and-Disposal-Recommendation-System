import os, json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers

APP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(APP_DIR, 'class_names.json')) as f:
    CLASS_NAMES = json.load(f)

CONFIDENCE_THRESHOLD = 0.70

DISPOSAL_KNOWLEDGE_BASE = {
    'glass': {
        'bin': 'Dry waste (blue) bin - glass',
        'prep': 'Empty it, rinse it, and remove metal caps or lids. Wrap broken glass in paper before disposal.',
        'env_note': 'Glass can be recycled again and again without losing quality.'},
    'metal': {
        'bin': 'Dry waste (blue) bin - metal',
        'prep': 'Empty and rinse cans or tins. Crushing cans saves space.',
        'env_note': 'Recycling aluminium saves up to about 95% of the energy needed to make it from raw ore.'},
    'paper': {
        'bin': 'Dry waste (blue) bin - paper',
        'prep': 'Keep it dry and free from grease, oil and food.',
        'env_note': 'Greasy or wet paper cannot be recycled and can spoil a whole batch.'},
    'plastic': {
        'bin': 'Dry waste (blue) bin - plastic',
        'prep': 'Empty the contents, rinse clean and put the cap back on.',
        'env_note': 'Clean plastics are far more likely to be accepted and recycled at sorting facilities.'},
}

CONTAMINATION_RULES = {
    'glass': {'warning': 'Residue found: this glass is contaminated.',
              'prep': 'Rinse the bottle or jar thoroughly before putting it in the recycling bin.'},
    'metal': {'warning': 'Residue found: this metal item is contaminated.',
              'prep': 'Rinse out all food residue before recycling. If it cannot be cleaned, use general waste.'},
    'paper': {'warning': 'Greasy or food-soiled paper cannot be recycled.',
              'bin': 'Wet waste (green) bin or general waste - NOT the dry recycling bin',
              'prep': 'Tear off any clean, dry part and recycle that separately. Dispose of the soiled part as wet or general waste.'},
    'plastic': {'warning': 'Residue found: this plastic item is contaminated.',
                'prep': 'Rinse off food residue. If it cannot be cleaned (for example oily wrappers), put it in general waste.'},
}


def classify(model, pil_img):
    img = pil_img.convert('RGB').resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), 0)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs, arr


def get_recommendation(category, confidence, contaminated=False):
    if confidence < CONFIDENCE_THRESHOLD:
        return {'status': 'uncertain',
                'message': 'The model is not confident enough. Please retake the photo (plain background, good light).'}
    rec = dict(DISPOSAL_KNOWLEDGE_BASE[category])
    rec['warning'] = None
    if contaminated:
        rule = CONTAMINATION_RULES[category]
        rec['prep'] = rule['prep']
        rec['bin'] = rule.get('bin', rec['bin'])
        rec['warning'] = rule['warning']
    rec['status'] = 'ok'
    return rec


LAST_CONV = {'efficientnetb0': 'top_conv', 'vgg16': 'block5_conv3', 'resnet50': 'conv5_block3_out'}


def _find_base(model):
    for l in model.layers:
        if l.name in LAST_CONV:
            return l
    raise ValueError('Backbone not found in model')


def make_gradcam(model, img_array):
    base = _find_base(model)
    conv_model = tf.keras.Model(base.input, [base.get_layer(LAST_CONV[base.name]).output, base.output])
    gap = [l for l in model.layers if isinstance(l, layers.GlobalAveragePooling2D)][0]
    head = model.layers[-1]
    x = tf.cast(img_array, tf.float32)
    for l in model.layers:
        if l.name == 'preprocess':
            x = l(x)
    with tf.GradientTape() as tape:
        conv_out, feats = conv_model(x)
        preds = head(gap(feats))
        idx = tf.argmax(preds[0])
        score = preds[:, idx]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.squeeze(conv_out[0] @ weights[..., tf.newaxis])
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()


def overlay_gradcam(pil_img, cam, alpha=0.4):
    img = np.array(pil_img.convert('RGB').resize((224, 224)))
    cam = cv2.resize(cam, (224, 224))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 1 - alpha, heat, alpha, 0)
