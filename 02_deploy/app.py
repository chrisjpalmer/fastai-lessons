#| export
from fastai.vision.all import load_learner
import gradio as gr

learn = load_learner('02_deploy/models/myModel.pkl')

def predict(img):
    category,_,probs = learn.predict(img)
    return category

demo = gr.Interface(
    fn=predict,
    inputs=["image"],
    outputs=["text"],
)

demo.launch()
