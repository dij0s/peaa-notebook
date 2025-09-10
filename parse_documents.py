import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""This noteboook addresses the parsing of unstructured PDF documents to a structured datastructure.""")
    return


@app.cell
def _():
    import os
    from PIL import Image
    return Image, os


@app.cell
def _():
    images_directory = "./images"
    return (images_directory,)


@app.cell
def _(Image, images_directory, os):
    images = [
        Image.open(os.path.join(images_directory, filename))
        for filename in os.listdir(images_directory)
        if filename.lower().endswith(".png")
    ]
    return (images,)


@app.cell
def _(images):
    images[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Visual Language Models""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Qwen2.5-VL-7B (barely fits on GPU), very slow""")
    return


@app.cell
def _():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    from qwen_vl_utils import process_vision_info
    return (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        process_vision_info,
    )


@app.cell
def _():
    from pydantic import BaseModel, Field
    from typing import Optional
    return BaseModel, Field, Optional


@app.cell
def _():
    import base64
    from io import BytesIO
    return BytesIO, base64


@app.cell
def _(AutoProcessor, Qwen2_5_VLForConditionalGeneration):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    return model, processor


@app.cell
def _(BaseModel, Field, Optional):
    class PageData(BaseModel):
        options: Optional[list[tuple[str, str]]] = Field(
            default=[],
            description="List of tuples of checked data from the form"
        )
        tables: Optional[list[tuple[str, list]]] = Field(
            default=[],
            description="List of tuples of tabular data from the form, mapping each table key to its information"
        )
    return (PageData,)


@app.cell
def _(PageData):
    example_page_data: PageData = PageData(
        options=[
            ('Justificatif "Besoins de chaleur - solutions standards"', "oui"),
            ('Justificatif "Installations de ventilation"', "non")
        ],
        tables=[
            ('Justificatif "Besoins de chaleur - solutions standards"', ["oui", "EN-VS-101a", "-"]),
            ('Justificatif "Production propre d\'électricité"', ["oui", "EN-VS-104", "-"])
        ]
    )
    return (example_page_data,)


@app.cell
def _(PageData, example_page_data: "PageData"):
    system_prompt = f"""
    You are a professional verifier for legal proof of energy measure forms.
    You will be given french documents, often displayed as checkboxes/options and tables.

    **Your goal is to retrieve the information of what is checked or not and the full content of the tables, in the following schema:**

    {PageData.model_json_schema()}

    Here's an example output:

    {example_page_data.model_dump_json()}
    """
    return (system_prompt,)


@app.cell
def _(
    BytesIO,
    Image,
    base64,
    model,
    process_vision_info,
    processor,
    system_prompt,
):
    def infer_image(image: Image):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        base64_image = f"data:image/png;base64,{base64_image}"

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": base64_image,
                    },
                    {"type": "text", "text": "Fill in the values from this image."},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        print(f"Sending inputs to {model.device}")
        inputs = inputs.to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        return output
    return (infer_image,)


@app.cell
def _(images, infer_image):
    infer_image(images[28])
    return


@app.cell
def _(images):
    images[28]
    return


@app.cell
def _(mo):
    mo.md(r"""Qwen2.5-VL-7B-AWQ (4bit aware-quantized)""")
    return


@app.cell
def _(model, torch):
    del model
    torch.cuda.empty_cache()
    return


if __name__ == "__main__":
    app.run()
