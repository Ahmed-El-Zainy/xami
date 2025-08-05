import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from dotsOCR.dots_ocr.utils import dict_promptmode_to_prompt
import warnings
import os
import glob
import json

warnings.filterwarnings("ignore")

# Model setup
model_path = "/content/dots.ocr/weights/DotsOCR"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    # local_files_only=True
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# Folder path
input_folder = "/content/drive/MyDrive/test_data"
output_folder = input_folder  # Save to the same folder

# Get model name for output files (clean version)
clean_model_name = "DotsOCR"

# Different prompts for different output formats
prompts = {
    "txt": """Please extract all text content from the PDF image and return it as plain text. Follow these rules:
    - Extract text in reading order
    - Use simple line breaks for paragraphs
    - Convert tables to simple text format
    - Convert formulas to plain text representation
    - Include all textual content without formatting markup
    - Do not include layout or positioning information""",

    "md": """Please extract the content from the PDF image and format it as clean Markdown. Follow these rules:
    - Use proper Markdown syntax for headers (# ## ###)
    - Format tables using Markdown table syntax
    - Use **bold** and *italic* for emphasis where appropriate
    - Convert formulas to LaTeX format wrapped in $ or $$
    - Use proper list formatting (- or 1.)
    - Preserve document hierarchy and structure
    - Use code blocks for any code content""",

    "json": """Please output the layout information from the PDF image as a structured JSON object, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object."""
}

# Supported image extensions
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']

# Get all image files in the folder
image_files = []
for extension in image_extensions:
    image_files.extend(glob.glob(os.path.join(input_folder, extension)))
    image_files.extend(glob.glob(os.path.join(input_folder, extension.upper())))

print(f"Found {len(image_files)} image files to process")

# Process each image with all three formats
for i, image_path in enumerate(image_files):
    try:
        print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        base_filename = os.path.splitext(os.path.basename(image_path))[0]

        # Process for each output format
        for format_type, prompt in prompts.items():
            try:
                print(f"  Generating {format_type.upper()} format...")

                # Prepare messages for the model
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_path
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]

                # Preparation for inference
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
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

                # Inference: Generation of the output
                generated_ids = model.generate(**inputs, max_new_tokens=24000)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                # Extract the output text (it's a list, get first element)
                result = output_text[0] if output_text else ""

                # Create output filename
                output_filename = f"{base_filename}_{clean_model_name}.{format_type}"
                output_path = os.path.join(output_folder, output_filename)

                # Save based on format type
                if format_type == "json":
                    try:
                        # Try to parse as JSON and save with proper formatting
                        parsed_json = json.loads(result)
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(parsed_json, f, indent=2, ensure_ascii=False)
                        print(f"    ✓ Saved as formatted JSON: {output_filename}")
                    except json.JSONDecodeError:
                        # If not valid JSON, save as text file with .json extension
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(result)
                        print(f"    ✓ Saved as text (invalid JSON): {output_filename}")
                else:
                    # Save as plain text or markdown
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(result)
                    print(f"    ✓ Saved as {format_type.upper()}: {output_filename}")

            except Exception as format_error:
                print(f"    ✗ Error generating {format_type.upper()} format: {str(format_error)}")
                continue

        # Show preview of one format (txt)
        if 'result' in locals():
            preview = result[:150] if len(result) > 150 else result
            print(f"Preview: {preview}...")

        print("-" * 60)

    except Exception as e:
        print(f"✗ Error processing {os.path.basename(image_path)}: {str(e)}")
        continue

print(f"\nProcessing complete! Processed {len(image_files)} images.")
print(f"Output files saved in: {output_folder}")
print("\nFile naming convention:")
print(f"- Plain text: {{filename}}_{clean_model_name}.txt")
print(f"- Markdown: {{filename}}_{clean_model_name}.md")
print(f"- JSON: {{filename}}_{clean_model_name}.json")