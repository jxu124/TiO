from typing import Any, Optional, Union
from PIL import Image
import gradio as gr
import argparse
import numpy as np
import json

# from chatbot_interface import VDG, bbox_to_sbbox, sbbox_to_bbox, MapFunc


# get_name = gr.Interface(lambda name: name, inputs="textbox", outputs="textbox")
# prepend_hello = gr.Interface(lambda name: f"Hello {name}!", inputs="textbox", outputs="textbox")
# append_nice = gr.Interface(lambda greeting: f"{greeting} Nice to meet you!",
#                            inputs="textbox", outputs=gr.Textbox(label="Greeting"))
# demo = gr.Series(get_name, prepend_hello, append_nice)



# model = VDG("/mnt/bn/ckpt-lq/vldd/invig_huge_grounding_checkpoints/10_3e-5_512_20230501-0232/checkpoint.best_score_0.7630.pt")
model = argparse.Namespace(**{'chat': lambda *args: ["Hello, this is a test message.", ]})


# demo = gr.TabbedInterface([chatbot, history], ["Chatbot", "Flagged List"])
# demo = tab_chatbot

def pipo_fn(index, image, image_anns, chatbot, bbox_gt, text):
    output = "Hello! This is a responce."
    chatbot += [(text, output)]
    image_anns = (image, [(bbox_gt, "bbox_gt")])
    return index, image, image_anns, chatbot, bbox_gt, text

examples = [
    [i, "example/792915ff00cee3c0_Drink_Bottle_Wine_Wine glass_2.jpg", 
     ("example/792915ff00cee3c0_Drink_Bottle_Wine_Wine glass_2.jpg", []), [], '[0.2, 0.2, 0.8, 0.8]', "what does the image describe?"]
for i in range(50)]

index = gr.Number(label="index", info="default value is '-1'.", value=-1, visible=False)
image = gr.Image(label="image", type='pil', visible=False)
image_anns = gr.AnnotatedImage(label="image_anns").style(color_map={"bbox_gt": "#aaaaff", "bbox_pred": "#aaffaa"}, width=400, height=200)
bbox_gt = gr.Textbox(label="bbox_gt", value='[0, 0, 0, 0]')
text = gr.Textbox(label="input", value='')
chatbot = gr.Chatbot(label='chat', value=[])

tab_chatbot = gr.Interface(
    pipo_fn,
    inputs=[index, image, image_anns, chatbot, bbox_gt, text],
    outputs=[index, image, image_anns, chatbot, bbox_gt, text],
    examples=examples,
    analytics_enabled=False
)

# def load_flagged():
#     shows = []
#     csv = "/mnt/bn/hri-lq/projects/VLDD/OFA-Invig/flagged/log.csv"
#     with open(csv) as f:
#         line = f.readline().strip().split(",")
#         for i, v in enumerate(line):
#             if v.endswith()
#             line[i] = v
#         shows += [line]


with gr.Blocks() as tab_flag:
    from functools import partial
    def load_txt(key, path):
        with open(path) as f:
            data = f.read()
        return {key: data}

    import datasets
    ds = datasets.Dataset.from_csv("/mnt/bn/hri-lq/projects/VLDD/OFA-Invig/flagged/log.csv")
    ds = ds.map(partial(load_txt, 'image_anns'), input_columns=['image_anns']).cast_column('image_anns', datasets.Value('string'))
    ds = ds.map(partial(load_txt, "chat"), input_columns=['chat']).cast_column('chat', datasets.Value('string'))
    ds[0]
    # 'index', 'image', 'image_anns', 'chat', 'bbox_gt', 'input', 'flag', 'username', 'timestamp'

    ds = gr.Dataset(label="flagged data", 
                    components=["text", "image", "text"], 
                    samples=[
        [
            "1", "example/792915ff00cee3c0_Drink_Bottle_Wine_Wine glass_2.jpg", '[["#instruction: what does the image describe?", "Hello! This is a responce."], ["#instruction: what does the image describe?", "Hello! This is a responce."], ["#instruction: what does the image describe?", "Hello! This is a responce."]]'
        ]])

# tab_flag = gr.Interface(
#     lamdba :,
#     inputs=[]
# )



# demo = tab_chatbot
demo = gr.TabbedInterface([tab_chatbot, tab_flag])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", "--server-name", default="0.0.0.0", type=str)
    args = parser.parse_args()
    demo.queue(4).launch(server_name=args.ip)
