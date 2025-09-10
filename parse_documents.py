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
    from PIL import Image
    return (Image,)


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
    import torch
    return (
        AutoProcessor,
        AutoTokenizer,
        Qwen2_5_VLForConditionalGeneration,
        process_vision_info,
        torch,
    )


@app.cell
def _():
    from pydantic import BaseModel, Field
    from typing import Optional
    return BaseModel, Field


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
def _(BaseModel, Field):
    class Checkbox(BaseModel):
        label: str
        checked: bool

    class TableRow(BaseModel):
        label: str
        values: list[str]

    class Table(BaseModel):
        title: str
        rows: list[TableRow]

    class PageData(BaseModel):
        metadata: dict[str, str] = Field(
            default_factory=dict,
            description="General key-value metadata from the page header (e.g., Title, Object, etc.)"
        )
        checkboxes: list[Checkbox] = Field(
            default_factory=list,
            description="List of checkboxes and their status"
        )
        tables: list[Table] = Field(
            default_factory=list,
            description="List of tables with labeled rows and values"
        )
    return Checkbox, PageData, Table, TableRow


@app.cell
def _(Checkbox, PageData, Table, TableRow):
    example_page_data_json = [
        page.model_dump_json()
        for page in [PageData(
            metadata={
                "Commune": "EN-VS",
                "Object": "Justificatif"
            },
            checkboxes=[
                Checkbox(label="Nouveau bâtiment", checked=True),
                Checkbox(label="Transformation", checked=True),
                Checkbox(label="Projet d'intérêt cantonal", checked=False),
            ],
            tables=[
                Table(
                    title="Justificatifs",
                    rows=[
                        TableRow(label="Justificatif \"Besoins de chaleur - solutions standards\"", values=["oui", "EN-VS-101a", "-"]),
                        TableRow(label="Justificatif \"Protection thermique, Performances ponctuelles\"", values=["oui", "EN-VS-102a", "-"]),
                    ]
                )
            ]
        ),
            PageData(
        metadata={
            "Commune": "Sierre",
            "Parcel": "8812",
            "Object": "Transformation d’un immeuble résidentiel"
        },
        checkboxes=[
            Checkbox(label="Transformation", checked=True),
            Checkbox(label="Changement d’affectation", checked=False),
            Checkbox(label="Protection solaire extérieure", checked=True),
        ],
        tables=[
            Table(
                title="Éléments d’enveloppe",
                rows=[
                    TableRow(label="Mur", values=["26 cm", "0.16", "0.17"]),
                    TableRow(label="Sol", values=["31 cm", "0.17", "0.25"]),
                    TableRow(label="Fenêtre", values=["Uw 1.00", "-", "1.30"]),
                ]
            )
        ]
    ),
        PageData(
        metadata={
            "Commune": "Monthey",
            "Parcel": "1203",
            "Object": "Agrandissement d’une école"
        },
        checkboxes=[
            Checkbox(label="Agrandissement", checked=True),
            Checkbox(label="Rénovation de toiture", checked=False),
            Checkbox(label="Installations de ventilation", checked=True),
        ],
        tables=[
            Table(
                title="Installations spéciales",
                rows=[
                    TableRow(label="Justificatif \"Énergie électrique, éclairage\"", values=["non", "EN-VS-111", "-"]),
                    TableRow(label="Justificatif \"Piscines chauffées\"", values=["oui", "EN-VS-135", "-"]),
                ]
            )
        ]
    )

        ]
    ]
    return (example_page_data_json,)


@app.cell
def _(PageData, example_page_data_json):
    system_prompt = f"""
    You are a professional verifier for official French legal forms related to energy measures.
    These documents include checkboxes, options, and tabular data. 
    Your task is to extract the data into a **structured JSON object** following the schema below.

    Always follow these rules:
    1. Respect the given schema exactly: {PageData.model_json_schema()}.
    2. Preserve the original French labels (do not translate).
    3. For checkboxes or binary options, set the value to "oui" or "non".
       - If the box is clearly marked (✓, x, •, checked, etc.), return "oui".
       - If the box is empty, return "non".
       - If ambiguous, return "non" and note uncertainty in a comment field if available.
    4. For tables:
       - Use the row label as the key (string).
       - Collect all corresponding values in a list, keeping order as seen in the document.
       - If a cell is empty, use "-" as placeholder.
    5. Do not hallucinate missing rows or values. Only include what is present.
    6. If the document has multiple tables, capture each separately in the "tables" field.

    Example output:

    {example_page_data_json}

    Output should be **only valid JSON** that conforms to the schema.
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


@app.cell(hide_code=True)
def _(images):
    ([
        {
            "metadata": {
                "Commune": "",
                "Parcel": "",
                "Object": ""
            },
            "checkboxes": [
                {"label": "Label Minergie existant", "checked": False},
                {"label": "Label Minergie P/A, Quartier ou Rénovation, certificat CECB A/A", "checked": False},
                {"label": "Certificat CECB", "checked": False},
                {"label": "Couverture des besoins de chaleur - neuf / agrandissement", "checked": True},
                {"label": "Protection thermique - enveloppe du bâtiment", "checked": True},
                {"label": "Installation de chauffage et de production d'eau chaude", "checked": True},
                {"label": "Chaîne renouvelable - remplacement du producteur de chaleur", "checked": False},
                {"label": "Production propre d'électricité", "checked": True},
                {"label": "Installations de ventilation", "checked": False},
                {"label": "Refraichissement, humidification et/ou déshumidification", "checked": False},
                {"label": "Installations et bâtiment spéciaux", "checked": False}
            ],
            "tables": [
                {
                    "title": "Justificatifs",
                    "rows": [
                        {"label": "Justificatif \"Besoins de chaleur - solutions standards\"", "values": ["non", "-", "-"]},
                        {"label": "Justificatif \"Besoins de chaleur - preuve calculée\"", "values": ["non", "-", "-"]},
                        {"label": "Justificatif \"Besoins de chaleur - bâtiments simples ENtеб\"", "values": ["non", "-", "-"]},
                        {"label": "Justificatif \"Protection thermique, Performances ponctuelles\"", "values": ["non", "-", "-"]},
                        {"label": "Justificatif \"Protection thermique, Performance globale\"", "values": ["non", "-", "-"]},
                        {"label": "Justificatif \"Chauffage et eau chaude sanitaire\"", "values": ["non", "-", "-"]},
                        {"label": "Justificatif \"Chaîne renouvelable, remplacement du producteur de chaleur\"", "values": ["non", "-", "-"]},
                        {"label": "Justificatif \"Production propre d'électricité\"", "values": ["non", "-", "-"]},
                        {"label": "Justificatif \"Installations de ventilation\"", "values": ["non", "-", "-"]},
                        {"label": "Justificatif \"Refraichissement, humidification/déshumidification\"", "values": ["non", "-", "-"]},
                        {"label": "Justificatif \"Énergie électrique, éclairage\"", "values": ["non", "-", "-"]},
                        {"label": "Justificatif \"Locaux frigorifiques\"", "values": ["non", "-", "-"]},
                        {"label": "Justificatif \"Résidences secondaires/Occupation intermittente\"", "values": ["non", "-", "-"]},
                        {"label": "Justificatif \"Chauffage de plein air\"", "values": ["non", "-", "-"]},
                        {"label": "Justificatif \"Piscines chauffées\"", "values": ["non", "-", "-"]},
                        {"label": "*Optimisation de l'exploitation*", "values": ["-", "-", "-"]}
                    ]
                }
            ]
        }
    ],images[28])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Qwen2.5-VL-7B-AWQ (4bit aware-quantized), ~2 times smaller on GPU, 2min.""")
    return


@app.cell
def _(torch):
    torch.cuda.empty_cache()
    return


@app.cell
def _(AutoProcessor, Qwen2_5_VLForConditionalGeneration, torch):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct-AWQ", dtype=torch.float16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct-AWQ", min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    return model, processor


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
    def infer_image_quantized(image: Image):
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
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output
    return (infer_image_quantized,)


@app.cell
def _(images):
    test_image = images[28]
    test_image
    return (test_image,)


@app.cell
def _(infer_image_quantized, test_image):
    infer_image_quantized(test_image)
    return


@app.cell
def _(test_image):
    ([
        {
            "metadata": {
                "Commune": "Monthey",
                "Parcel": "1203",
                "Object": "Agrandissement d’une école"
            },
            "checkboxes": [
                {"label": "Agrandissement", "checked": True},
                {"label": "Rénovation de toiture", "checked": False},
                {"label": "Installations de ventilation", "checked": True}
            ],
            "tables": [
                {
                    "title": "Installations spéciales",
                    "rows": [
                        {"label": "Justificatif \"Énergie électrique, éclairage\"", "values": ["non", "EN-VS-111", "-"]},
                        {"label": "Justificatif \"Locaux frigorifiques\"", "values": ["non", "EN-VS-112", "-"]},
                        {"label": "Justificatif \"Résidences secondaires/Occupation intermittente\"", "values": ["non", "EN-VS-130", "-"]},
                        {"label": "Justificatif \"Habitations\"", "values": ["non", "EN-VS-131", "-"]},
                        {"label": "Justificatif \"Halles gonflables\"", "values": ["non", "EN-VS-132", "-"]},
                        {"label": "Justificatif \"Installation de production d'électricité\"", "values": ["non", "EN-VS-133", "-"]},
                        {"label": "Justificatif \"Chauffage de plein air\"", "values": ["non", "EN-VS-134", "-"]},
                        {"label": "Justificatif \"Piscines chauffées\"", "values": ["non", "EN-VS-135", "-"]},
                        {"label": "*Optimisation de l'exploitation*", "values": ["-", "-", "-"]}
                    ]
                }
            ]
        }
    ], test_image)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### OCR""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""nanonets/Nanonets-OCR-s""")
    return


@app.cell
def _():
    from transformers import AutoModelForImageTextToText
    return (AutoModelForImageTextToText,)


@app.cell
def _(AutoModelForImageTextToText, AutoProcessor, AutoTokenizer):
    model_path = "nanonets/Nanonets-OCR-s"
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, 
        dtype="auto",
        device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


@app.cell
def _(BytesIO, Image, base64, model, processor, system_prompt):
    def infer_ocr(image: Image):
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

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)

        output_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]

        output = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output
    return (infer_ocr,)


@app.cell
def _(infer_ocr, test_image):
    infer_ocr(test_image)
    return


@app.cell(hide_code=True)
def _(test_image):
    ({
        "metadata": {
            "Commune": "Crans-Montana",
            "Parcel": "3945",
            "Object": "Construction d'une villa - Chermignon"
        },
        "checkboxes": [
            {"label": "Nouveau bâtiment", "checked": True},
            {"label": "Transformation", "checked": True},
            {"label": "Projet d'intérêt cantonal", "checked": False}
        ],
        "tables": [
            {
                "title": "Justificatifs",
                "rows": [
                    {"label": "Justificatif \"Besoins de chaleur - preuve calculée\"", "values": ["oui", "EN-VS-101a", "-"]},
                    {"label": "Justificatif \"Besoins de chaleur - bâtiments empilés ENteb\"", "values": ["-", "-", "-"]}
                ]
            }
        ]
    },test_image)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""numind/NuMarkdown-8B-Thinking""")
    return


@app.cell
def _(AutoProcessor, Qwen2_5_VLForConditionalGeneration, torch):
    model_id = "numind/NuMarkdown-8B-reasoning"       

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        min_pixels=100*28*28, max_pixels=5000*28*28   
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, processor


@app.cell
def _(Image, model, processor, torch):
    def infer_numd(image: Image):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
            ],
        }]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_input = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)

        with torch.no_grad():
            model_output = model.generate(**model_input, temperature = 0.7, max_new_tokens=5000)

        result = processor.decode(model_output[0])
        reasoning = result.split("<think>")[1].split("</think>")[0]
        answer  = result.split("<answer>")[1].split("</answer>")[0]

        return answer
    return (infer_numd,)


@app.cell
def _(test_image):
    test_image
    return


@app.cell
def _(infer_numd, test_image):
    infer_numd(test_image)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Preprocessing documents""")
    return


@app.cell
def _():
    import PIL
    return (PIL,)


@app.cell
def _(Image):
    example_mask = Image.open("./mask/example_mask.png")
    example_mask
    return (example_mask,)


@app.cell
def _(PIL):
    def diff_images(img1, img2):
        # Open images
        img1 = img1.convert("RGB")
        img2 = img2.convert("RGB")

        common_size = img1.size
        img2_resized = img2.resize(common_size)

        # Compute absolute difference
        diff = PIL.ImageChops.difference(img1, img2_resized)
        return diff
    return (diff_images,)


@app.cell
def _(PIL, diff_images, example_mask, test_image):
    PIL.ImageChops.invert(diff_images(test_image, example_mask))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Simulating an already masked image""")
    return


@app.cell
def _(Image):
    masked_image = Image.open("./examples/masked_image.png")
    masked_image
    return (masked_image,)


@app.cell
def _(infer_image_quantized, masked_image):
    infer_image_quantized(masked_image)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""après ~1m. d'inférence, avec Qwen3:7b-AQW""")
    return


@app.cell(hide_code=True)
def _(masked_image):
    ([
        {
            "metadata": {},
            "checkboxes": [
                {"label": "Annexe", "checked": True},
                {"label": "EN-VS-101a", "checked": True},
                {"label": "EN-VS-101b", "checked": False},
                {"label": "EN-VS-101c", "checked": False},
                {"label": "EN-VS-102a", "checked": True},
                {"label": "EN-VS-102b", "checked": False},
                {"label": "EN-VS-103", "checked": True},
                {"label": "EN-VS-120", "checked": False},
                {"label": "EN-VS-104", "checked": True},
                {"label": "EN-VS-105", "checked": False},
                {"label": "EN-VS-110", "checked": True},
                {"label": "EN-VS-111", "checked": False},
                {"label": "EN-VS-112", "checked": False},
                {"label": "EN-VS-130", "checked": False},
                {"label": "EN-VS-131", "checked": False},
                {"label": "EN-VS-132", "checked": False},
                {"label": "EN-VS-133", "checked": False},
                {"label": "EN-VS-134", "checked": False},
                {"label": "EN-VS-135", "checked": False}
            ],
            "tables": []
        }
    ], masked_image)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""avec modèle OCR (plus petit)""")
    return


@app.cell
def _():
    from transformers import pipeline
    from dotenv import load_dotenv
    import os
    return load_dotenv, os, pipeline


@app.cell
def _(load_dotenv, pipeline, torch):
    load_dotenv()

    pipe = pipeline(
        "image-text-to-text",
        model="google/gemma-3-4b-it",
        device="cuda",
        torch_dtype=torch.bfloat16
    )
    return (pipe,)


@app.cell
def _(BytesIO, Image, base64, pipe, system_prompt):
    def infer_ocr_gemma(image: Image):
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
                        "image": "https://i.imgur.com/WXKeer3.png",
                    },
                    {"type": "text", "text": "Fill in the values from this image."},
                ],
            }
        ]
    
        output = pipe(text=messages, max_new_tokens=2048)
        print(output[0]["generated_text"][-1]["content"])
    return (infer_ocr_gemma,)


@app.cell
def _(system_prompt):
    system_prompt
    return


@app.cell
def _(infer_ocr_gemma, test_image):
    infer_ocr_gemma(test_image)
    return


if __name__ == "__main__":
    app.run()
