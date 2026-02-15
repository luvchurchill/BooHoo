from pathlib import Path

import gradio as gr

from predict import InfantCryPredictor


MODEL_PATH = "deepinfant.pth"

TIPS = {
    "hungry": "Feed your baby if it is close to the usual feeding time.",
    "burping": "Hold your baby upright and gently pat their back.",
    "belly_pain": "Try gentle tummy massage and check for signs of colic.",
    "discomfort": "Check diaper, clothing fit, and room comfort.",
    "tired": "Reduce stimulation and help your baby settle for sleep.",
    "cold_hot": "Adjust clothing or room temperature to a comfortable level.",
    "lonely": "Provide comfort with holding, rocking, or soft reassurance.",
    "scared": "Reduce sudden noise/light and soothe with a calm voice.",
    "unknown": "Cry pattern is unclear. Check basics and monitor closely.",
}


_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = InfantCryPredictor(model_path=MODEL_PATH)
    return _predictor


def predict_from_audio(audio_path: str):
    if not audio_path:
        raise gr.Error("No audio provided. Use Record or Select File first.")
    if not Path(audio_path).exists():
        raise gr.Error("Audio path is invalid. Please record again or upload a file.")

    predictor = _get_predictor()
    label, confidence = predictor.predict(audio_path)
    confidence_text = f"{confidence:.2%}"
    tip = TIPS.get(label, "Check your baby and consult a healthcare professional if concerned.")
    return label, confidence_text, tip


with gr.Blocks(title="DeepInfant Windows GUI") as demo:
    gr.Markdown(
        "# DeepInfant\n"
        "Choose one option: `Record` from microphone or `Select File` to upload audio."
    )

    with gr.Tab("Record"):
        mic_audio = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Record Baby Cry",
        )
        mic_button = gr.Button("Predict")

    with gr.Tab("Select File"):
        upload_audio = gr.Audio(
            sources=["upload"],
            type="filepath",
            label="Upload Audio File",
        )
        upload_button = gr.Button("Predict")

    with gr.Row():
        out_label = gr.Textbox(label="Predicted Reason")
        out_confidence = gr.Textbox(label="Confidence")
    out_tip = gr.Textbox(label="Caregiver Tip")

    mic_button.click(
        fn=predict_from_audio,
        inputs=mic_audio,
        outputs=[out_label, out_confidence, out_tip],
    )
    upload_button.click(
        fn=predict_from_audio,
        inputs=upload_audio,
        outputs=[out_label, out_confidence, out_tip],
    )


if __name__ == "__main__":
    demo.launch()
