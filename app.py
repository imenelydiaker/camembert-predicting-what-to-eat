from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import pipeline 
import gradio as gr

import random


conjonctions = ['du', 'de la', 'au']
dates = ['ce soir', 'ce midi', 'à midi', 'pour le dîner']
prompts=[f"Je veux manger {conjonction} <mask> {date}" for conjonction in conjonctions for date in dates]
print(prompts)

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")

def predict(prompt: str):
    camembert_fill_mask  = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    results = camembert_fill_mask(prompt)
    idx = random.randint(0,4)
    print(idx)
    to_return = results[idx]['sequence']
    return to_return


if __name__ == "__main__":
    iface = gr.Interface(
        fn=predict, 
        inputs='text',
        outputs='text',
        examples=[[p] for p in prompts]
        )

    iface.launch()  
