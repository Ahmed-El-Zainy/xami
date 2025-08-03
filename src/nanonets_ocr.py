import time
from gradio_client import Client, handle_file

def convert_pdf_to_markdown(
    client_url: str,
    username: str,
    password: str,
    file_paths: list[str],
    model_name: str = "hosted_vllm/nanonets/Nanonets-OCR-s"
):
    """
    Convert PDF/images to markdown using the API

    Args:
        client_url: URL of the docext server
        username: Authentication username
        password: Authentication password
        file_paths: List of file paths to convert
        model_name: Model to use for conversion

    Returns:
        str: Converted markdown content
    """
    client = Client(client_url, auth=(username, password))

    # Prepare file inputs
    file_inputs = [{"image": handle_file(file_path)} for file_path in file_paths]

    # Convert to markdown (non-streaming)
    result = client.predict(
        images=file_inputs,
        api_name="/process_markdown_streaming"
    )

    return result

# Example usage
# client url can be the local host or the public url like `https://6986bdd23daef6f7eb.gradio.live/`
CLIENT_URL = "http://localhost:7860"

# Single image conversion
markdown_content = convert_pdf_to_markdown(
    CLIENT_URL,
    "admin",
    "admin",
    ["assets/invoice_test.pdf"]
)
print(markdown_content)

# Multiple files conversion
markdown_content = convert_pdf_to_markdown(
    CLIENT_URL,
    "admin",
    "admin",
    ["assets/invoice_test.jpeg", "assets/invoice_test.pdf"]
)
print(markdown_content)