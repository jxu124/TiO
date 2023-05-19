from typing import Any, Optional, Union
from PIL import Image
import gradio as gr
import argparse
import numpy as np

from chatbot_interface import VDG, bbox_to_sbbox, sbbox_to_bbox, MapFunc


model = VDG("/mnt/bn/ckpt-lq/vldd/invig_huge_grounding_checkpoints/10_3e-5_512_20230501-0232/checkpoint.best_score_0.7630.pt")


def vischatg_fn(image, input, bbox, options, dialog):
    oracle_prompt, questioner_prompt, anns = fill_propmt_blocks(image, input, bbox, options, dialog)
    if 0 in options and 2 not in options:
        a = model.chat([image], [oracle_prompt])[0]
        questioner_prompt = questioner_prompt.replace("<text_what_oracle_speak>", a)
    else:
        a = input
    q = model.chat([image], [questioner_prompt])[0]
    if q.startswith(" not yet."):
        q = q[9:]
    if q.startswith(" sure."):
        q = q[6:]
        pred_bbox = (i for i in sbbox_to_bbox(q, *image.size).reshape(4).astype(int))
        anns = (anns[0], anns[1] + [(pred_bbox, "predict")])
    dialog += [(a, q)]
    return oracle_prompt, questioner_prompt, anns, dialog


def fill_propmt_blocks(image, input, bbox, options, dialog):
    _dialog = [j for i in dialog for j in i] if dialog else []
    if 2 in options:
        _dialog = _dialog + [input, "<responce>"]
        oracle_prompt = ""
        questioner_prompt = MapFunc.make_text_pair([input, *_dialog], human_first=True)[0]
    else:
        if 0 in options:
            _dialog = _dialog + ["<text_what_oracle_speak>", "<responce>"]
            sbbox = bbox_to_sbbox(np.array(bbox[0], dtype=float))
            oracle_prompt = "state your query about the region." if len(dialog) == 0 else "answer the question based on the region."
            oracle_prompt = MapFunc.make_text_pair([oracle_prompt, *_dialog], src_sbbox=sbbox)[0]
            questioner_prompt = "which region does the context describe?" if 1 in options else "can you specify which region the context describes?"
            questioner_prompt = MapFunc.make_text_pair([questioner_prompt, *_dialog], human_first=True)[0]
        else:
            _dialog = _dialog + [input, "<responce>"]
            oracle_prompt = ""
            questioner_prompt = "which region does the context describe?" if 1 in options else "can you specify which region the context describes?"
            questioner_prompt = MapFunc.make_text_pair([questioner_prompt, *_dialog], human_first=True)[0]
    oracle_prompt = oracle_prompt.lstrip(" \n")
    questioner_prompt = questioner_prompt.lstrip(" \n")

    w, h = image.size
    bbox = np.array(bbox[0], dtype=float) * np.asarray([w, h, w, h])
    bbox = (i for i in bbox.astype(int))
    anns = (image, [(bbox, "oracle")])
    return oracle_prompt, questioner_prompt, anns


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=0.8):
            blk_image = gr.Image(label="image", type='pil')
            with gr.Row():
                blk_bbox = gr.List(
                    label="#oracle# bbox (relative, with values from 0 to 1)",
                    headers=["x1", "y1", "x2", "y2"], 
                    datatype="number",
                    row_count=(1, "fixed"),
                    col_count=(4, "fixed"),
                    interactive=True
                )
            blk_input = gr.Textbox(label="input", lines=5, value="I love this picture.")
            cgb_option = gr.CheckboxGroup(label="options", choices=["#oracle# speak for me", "force predict bbox", "disable prompt"], type="index")
            with gr.Row():
                btn_clear = gr.Button("Clear")
                btn_check = gr.Button("Check")
                btn_submit = gr.Button("Submit")
        with gr.Column(scale=2.4):
            blk_chatbot = gr.Chatbot(label="Chatbot", value=[])
        with gr.Column(scale=0.8):
                blk_image_bbox = gr.AnnotatedImage(label="image with bbox").style(color_map={"oracle": "#aaaaff", "predict": "#aaffaa"})
                blk_a_prompt = gr.Textbox(label="#oracle# prompt (real input)", lines=3, value="")
                blk_q_prompt = gr.Textbox(label="questioner/guesser prompt (real input)", lines=3, value="")

    btn_clear.click(lambda: None, None, outputs=blk_chatbot, queue=False)

    # blk_input.change(fill_propmt_blocks, inputs=[blk_input, blk_bbox, cgb_option, blk_chatbot], outputs=[blk_a_prompt, blk_q_prompt])
    btn_check.click(fill_propmt_blocks, inputs=[blk_image, blk_input, blk_bbox, cgb_option, blk_chatbot], outputs=[blk_a_prompt, blk_q_prompt, blk_image_bbox],
                    api_name="/check")
    btn_submit.click(vischatg_fn, inputs=[blk_image, blk_input, blk_bbox, cgb_option, blk_chatbot], outputs=[blk_a_prompt, blk_q_prompt, blk_image_bbox, blk_chatbot],
                     api_name="/predict")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", "--server-name", default="0.0.0.0", type=str)
    args = parser.parse_args()
    demo.launch(server_name=args.ip)
