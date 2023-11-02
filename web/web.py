from flask import Flask, render_template, request, jsonify
from autodistill_clip import CLIP
from autodistill_metaclip import MetaCLIP
from autodistill_altclip import AltCLIP
from autodistill_grounding_dino import GroundingDINO
from autodistill_owlv2 import OWLv2
from autodistill_fastvit import FastViT
from autodistill_fastsam import FastSAM
from autodistill.utils import plot
from autodistill.detection import CaptionOntology
import cv2
import base64
import tempfile

app = Flask(__name__)

clip = CLIP(None)
metaclip = MetaCLIP(None)
grounding_dino = GroundingDINO(None)
owlv2 = OWLv2(None)
fastvit = FastViT(None)
altclip = AltCLIP(None)
fastsam = FastSAM(None)

ENABLED_MODELS = {
    "CLIP": clip,
    "MetaCLIP": metaclip,
    "GroundingDINO": grounding_dino,
    "OWLv2": owlv2,
    "FastViT": fastvit,
    "AltCLIP": altclip,
    "FastSAM": fastsam,
}

CLASSIFICATION_MODELS = [
    "CLIP",
    "MetaCLIP",
    "FastViT",
    "AltCLIP",
]

DETECTION_MODELS = [
    "GroundingDINO",
    "OWLv2",
]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.form['image']
        model_name = request.form['model']
        classes = request.form['classes'].split(',')

        model = ENABLED_MODELS[model_name]

        model.ontology = CaptionOntology(
            {
                c: c
                for c in classes
            }
        )

        file = base64.b64decode(file.split(',')[1])

        with tempfile.NamedTemporaryFile() as temp:
            temp.write(file)

            image_url = temp.name
            result = model.predict(image_url)

            if model_name in DETECTION_MODELS:
                plotted_result = plot(
                    image=cv2.imread(image_url),
                    detections=result,
                    classes=classes,
                    raw=True
                )

                _, buffer = cv2.imencode('.jpg', plotted_result)

                image_base64 = base64.b64encode(buffer)

                return jsonify({
                    "image": image_base64.decode('utf-8'),
                    "class_confidences": [],
                    "class_names": [],
                })
            else:
                class_names = [classes[i] for i in result.class_id.tolist()]
                
                confidences = result.confidence.tolist()

                class_confidences = list(zip(class_names, confidences))

                return jsonify({
                    "class_confidences": class_confidences,
                })
    
    return render_template('index.html', models=ENABLED_MODELS.keys())

if __name__ == '__main__':
    app.run(debug=True)