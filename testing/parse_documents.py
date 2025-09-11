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
                "Title": "EN-VS",
                "Object": "Justificatif des mesures énergétiques"
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
            "Title": "EN-VS-101a",
            "Object": "Couverture des besoins de chaleur"
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
            "Title": "EN-VS-102a",
            "Object": "Performance thermique Performances ponctuelles"
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
    model_awq = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct-AWQ", dtype=torch.float16, device_map="auto"
    )
    processor_awq = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct-AWQ", min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    return model_awq, processor_awq


@app.cell
def _(
    BytesIO,
    Image,
    base64,
    model_awq,
    process_vision_info,
    processor_awq,
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

        text = processor_awq.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor_awq(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = model_awq.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = processor_awq.batch_decode(
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
    mo.md(r"""numind/NuExtract-2.0-8B-GPTQ (quantized finetuned Qwen2.5-VL:7b)""")
    return


@app.cell
def _(AutoProcessor, Qwen2_5_VLForConditionalGeneration, model_id, torch):
    model_name = "numind/NuExtract-2.0-8B-GPTQ"

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
    mo.md(r"""après ~1m. d'inférence, avec Qwen2.5-VL:7b-AQW, toutes les coches sont reportées sauf pour le doc. EN-VS-104""")
    return


@app.cell(hide_code=True)
def _(masked_image):
    ([
        {
            "metadata": {
                "Title": "EN-VS",
                "Object": "Justificatif des mesures énergétiques"
            },
            "checkboxes": [
                {"label": "Annexe", "checked": True},
                {"label": "EN-VS-101a", "checked": True},
                {"label": "EN-VS-101b", "checked": False},
                {"label": "EN-VS-101c", "checked": False},
                {"label": "EN-VS-102a", "checked": True},
                {"label": "EN-VS-102b", "checked": False},
                {"label": "EN-VS-103", "checked": True},
                {"label": "EN-VS-120", "checked": False},
                {"label": "EN-VS-104", "checked": False},
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


@app.cell
def _(mo):
    mo.md(r"""avec modèle OCR (plus petit), gemma fonctionne pas..""")
    return


@app.cell
def _():
    from transformers import Gemma3ForConditionalGeneration
    from dotenv import load_dotenv
    import os
    return Gemma3ForConditionalGeneration, load_dotenv, os


@app.cell
def _(AutoProcessor, Gemma3ForConditionalGeneration, load_dotenv):
    load_dotenv()

    gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
        "google/gemma-3-4b-it", device_map="auto"
    ).eval()
    gemma_processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
    return gemma_model, gemma_processor


@app.cell
def _(gemma_model, gemma_processor, processor, torch):
    def infer_ocr_gemma():
        #buffer = BytesIO()
        #image.save(buffer, format="PNG")
        #buffer.seek(0)
        #base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        #base64_image = f"data:image/png;base64,{base64_image}"

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
                    {"type": "text", "text": "Describe this image in detail."}
                ]
            }
        ]

        inputs = gemma_processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(gemma_model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = gemma_model.generate(**inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]

        decoded = processor.decode(generation, skip_special_tokens=True)
        return decoded
    return (infer_ocr_gemma,)


@app.cell
def _(infer_ocr_gemma):
    infer_ocr_gemma()
    return


@app.cell
def _(Image):
    masked_table = Image.open("./examples/masked_table.png")
    masked_table
    return (masked_table,)


@app.cell
def _(infer_image_quantized, masked_table):
    infer_image_quantized(masked_table)
    return


@app.cell(hide_code=True)
def _(masked_table):
    ([
        {
            "metadata": {
                "Title": "EN-VS-101a",
                "Object": "Couverture des besoins de chaleur"
            },
            "checkboxes": [
                {"label": "Habitat collectif", "checked": False},
                {"label": "Habitat individuel", "checked": True},
                {"label": "autre", "checked": False}
            ],
            "tables": [
                {
                    "title": "Exigences",
                    "rows": [
                        {
                            "label": "Eléments de construction opaques contre l'extérieur",
                            "values": ["0,17", "W/m²K"]
                        },
                        {
                            "label": "Fenêtres",
                            "values": ["Uw max. 1,00", "W/m²K"]
                        },
                        {
                            "label": "Ventilation contrôlée",
                            "values": ["≥ 80%", "rendement RC"]
                        },
                        {
                            "label": "Installation solaire th.",
                            "values": [
                                "pour l'eau chaude sanitaire avec surface d'eau min. 2% de la SRE",
                                "min. 4 m² projeté"
                            ]
                        },
                        {
                            "label": "Installation solaire th.",
                            "values": [
                                "pour l'eau chaude sanitaire avec surface d'eau min. 2% de la SRE",
                                "min. 6 m² projeté"
                            ]
                        },
                        {
                            "label": "Eléments de construction opaques contre l'extérieur",
                            "values": ["0,15", "W/m²K"]
                        },
                        {
                            "label": "Fenêtres",
                            "values": ["Uw max. 1,00", "W/m²K"]
                        },
                        {
                            "label": "Eléments de construction opaques contre l'extérieur",
                            "values": ["0,15", "W/m²K"]
                        },
                        {
                            "label": "Fenêtres",
                            "values": ["Uw max. 0,80", "W/m²K"]
                        }
                    ]
                }
            ]
        }
    ], masked_table)
    return


@app.cell
def _(Image, infer_image_quantized):
    masked_table_bis = Image.open("./examples/masked_table_bis.png")
    infer_image_quantized(masked_table_bis)
    return (masked_table_bis,)


@app.cell(hide_code=True)
def _(masked_table_bis):
    ([
        {
            "metadata": {
                "Title": "EN-VS-102a",
                "Object": "Performance thermique Performances ponctuelles"
            },
            "checkboxes": [
                {"label": "Agrandissement", "checked": True},
                {"label": "Transformation", "checked": False},
                {"label": "Changement d'affectation", "checked": False}
            ],
            "tables": [
                {
                    "title": "Protection thermique en été",
                    "rows": [
                        {"label": "Valeur g", "values": ["oui", "non", "non"]},
                        {"label": "Rafrichissement", "values": ["non", "oui", "non"]}
                    ]
                },
                {
                    "title": "Éléments d'enveloppe et exigences",
                    "rows": [
                        {"label": "Toit/plafond", "values": ["tt1", "20.00", "0.16", "0.17", "20.00", "0.17", "0.25"]},
                        {"label": "Mur", "values": ["me1", "20.00", "0.15", "0.17", "mt1", "20.00", "0.17", "0.25"]},
                        {"label": "Mur", "values": ["me2", "26.00", "0.16", "0.17", "mi1", "20.00", "0.15", "0.25"]},
                        {"label": "Mur", "values": ["", "", "0.17", "", "", "", "0.25"]},
                        {"label": "Mur", "values": ["", "", "0.17", "", "", "", "0.25"]},
                        {"label": "Sol", "values": ["d1", "31.00", "0.17", "0.17", "", "", "", "0.25"]},
                        {"label": "Portes", "values": ["", "", "", "", "", "", "1.70", "2.00"]},
                        {"label": "Caisson de store", "values": ["cs1", "6.00", "0.45", "0.50", "", "", "", "0.50"]},
                        {"label": "Fenêtre, porte-fenêtre, Velux", "values": ["f1", "0.60", "0.93", "Uw max. 1.00", "", "", "", "Uw max. 1.30"]},
                        {"label": "Fenêtre, porte-fenêtre, Velux", "values": ["", "", "", "Uw max. 1.00", "", "", "", "Uw max. 1.30"]},
                        {"label": "Porte", "values": ["", "", "", "", "", "", "1.20", "1.50"]},
                        {"label": "Fenêtre avec corps de chauffe", "values": ["", "", "", "Uw max. 1.00", "", "", "", "Uw max. 1.30"]}
                    ]
                }
            ]
        }
    ], masked_table_bis)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Tesseract OCR

    ![Tesseract](https://i.imgur.com/Ft1ETx1.png)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    NuMarkdown-8B-Thinking

    EN-VS-102a | Justificatif énergétique
    -----------|---------------------------
    | Protection thermique
    | Performances ponctuelles

    Nature des travaux :
    [ ] Nouveau bâtiment
    [ ] Transformation
    [ ] Agrandissement
    [ ] Changement d'affectation

    ---

    **Hygiène de l'air intérieur**

    Concept de
    ventilation:
    * Système de ventilation avec air fourni et air repris
    * Installation simple d'air repris avec entrées d'air neuf définies
    * Aération par fenêtres avec commande automatique
    * Aération par ouverture manuelle des fenêtres
    * Autre:

    ---

    **Protection thermique en été**

    Valeur g
    * Protection solaire extérieure
    * Justificatif de la valeur g du vitrage et de la protection solaire
    * Valeur g non respectée ;
    motif: Choisir s.v.p. :

    Rafraîchissement
    * Non, ni "nécessaire" ni "souhaitable"
    * Oui [ ] Commande automatique des protections solaires
    [ ] Pas automatique ;
    motif: freecooling

    ---

    **Eléments d'enveloppe et exigences** NEUF (bâtiment à construire ou agrandissement)
    Il = habitat individuel
    Neuf - Norme SIA 380/1;2016 ou solution standard 1 ou 2
    rendement de récup ventil mini 80% requis pour sol.std. 1 / part énergie fossile max 25%

    | Elément | Epaisseur de l'isolant en cm | N° (2) | épaisseur cm | Valeur U W/m²K | Valeur limite W/m²K | N° (2) | épaisseur cm | Valeur U W/m²K | Valeur limite W/m²K |
    |---|---|---|---|---|---|---|---|---|---|
    | Toit/plafond | tt1 | 20.00 | 0.16 | 0.17 | | | | 0.25 |
    | Toit/plafond | | | | | | | | 0.25 |
    | Mur | me1 | 20.00 | 0.15 | 0.17 | mt1 | 20.00 | 0.17 | 0.25 |
    | Mur | me2 | 26.00 | 0.16 | 0.17 | mi1 | 20.00 | 0.15 | 0.25 |
    | Mur | | | | | | | | 0.25 |
    | Mur | | | | | | | | 0.25 |
    | Sol | d1 | 31.00 | 0.17 | 0.17 | | | | 0.25 |
    | Sol | | | | | | | | 0.25 |
    | Portes (SIA 343) | | | | 1.70 | | | | 2.00 |
    | Caisson de store | cs1 | 6.00 | 0.45 | 0.50 | | | | 0.50 |
    | | N° (2) | Uvitrage W/m²K | Ufenêtre W/m²K | Valeur limite W/m²K | N° (2) | Uvitrage W/m²K | Ufenêtre W/m²K | Valeur limite W/m²K |
    | Fenêtre, porte-fenêtre, Velux | f1 | 0.60 | 0.93 | Uw max. 1.00 | | | | Uw max. 1.30 |
    | Fenêtre, porte-fenêtre, Velux | | | | Uw max. 1.00 | | | | Uw max. 1.30 |
    | Porte | | | | 1.20 | | | | 1.50 |
    | Fenêtre avec corps de chauffe (3) | | | | Uw max. 1.00 | | | | Uw max. 1.30 |

    ---
    """
    )
    return


@app.cell
def _(system_prompt):
    system_prompt
    return


if __name__ == "__main__":
    app.run()
