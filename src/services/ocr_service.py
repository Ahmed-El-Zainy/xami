# services/ocr_service.py
import torch
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import io
import time
import json
import os
from typing import List, Tuple, Optional, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
import subprocess
import sys
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info

from src.config.config_settings import settings
from src.config.logging_config import get_logger, log_execution_time, CustomLoggerTracker
from src.models.schemas import OCRResult

warnings.filterwarnings("ignore")

try:
    custom = CustomLoggerTracker()
    logger = custom.get_logger("ocr_service")
    logger.info("Custom Logger Start Working.....")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("ocr_service")
    logger.info("Using standard logger - custom logger not available")


class ModelDownloader:
    """Handles automatic model downloading and setup"""
    
    def __init__(self):
        self.model_dir = Path("./weights/DotsOCR")
        self.model_repo = "rednote-hilab/dots.ocr"
        self.logger = logger
        
    def check_model_exists(self) -> bool:
        """Check if DotsOCR model exists locally"""
        required_files = [
            "config.json",
            "configuration_dots.py", 
            "modeling_dots_ocr.py",
            "modeling_dots_vision.py",
            "preprocessor_config.json"
        ]
        
        if not self.model_dir.exists():
            return False
            
        for file in required_files:
            if not (self.model_dir / file).exists():
                return False
                
        return True
    
    async def download_model(self, force_download: bool = False) -> bool:
        """Download DotsOCR model using huggingface_hub"""
        if self.check_model_exists() and not force_download:
            self.logger.info("DotsOCR model already exists")
            return True
            
        try:
            self.logger.info(f"Downloading DotsOCR model from {self.model_repo}")
            
            # Ensure directory exists
            self.model_dir.mkdir(parents=True, exist_ok=True)
            
            # Import and download using huggingface_hub
            from huggingface_hub import snapshot_download
            
            snapshot_download(
                repo_id=self.model_repo,
                local_dir=str(self.model_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            self.logger.info(f"Model downloaded successfully to {self.model_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download model: {str(e)}")
            return False
    
    async def setup_model_files(self):
        """Setup additional model configuration files"""
        try:
            # Copy configuration files if they don't exist
            config_files = {
                "configuration_dots.py": self._get_configuration_dots_content(),
                "modeling_dots_ocr.py": self._get_modeling_ocr_content(),
                "modeling_dots_vision.py": self._get_modeling_vision_content()
            }
            
            for filename, content in config_files.items():
                file_path = self.model_dir / filename
                if not file_path.exists():
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.logger.info(f"Created {filename}")
                    
        except Exception as e:
            self.logger.error(f"Error setting up model files: {str(e)}")
    
    def _get_configuration_dots_content(self) -> str:
        """Get configuration_dots.py content"""
        return '''from typing import Any, Optional
from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2 import Qwen2Config
from transformers import Qwen2_5_VLProcessor, AutoProcessor
from transformers.models.auto.configuration_auto import CONFIG_MAPPING


class DotsVisionConfig(PretrainedConfig):
    model_type: str = "dots_vit"

    def __init__(
        self,
        embed_dim: int = 1536,
        hidden_size: int = 1536,
        intermediate_size: int = 4224,
        num_hidden_layers: int = 42,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 1,
        rms_norm_eps: float = 1e-5,
        use_bias: bool = False,
        attn_implementation="flash_attention_2",
        initializer_range=0.02,
        init_merger_std=0.02,
        is_causal=False,
        post_norm=True,
        gradient_checkpointing=False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.rms_norm_eps = rms_norm_eps
        self.use_bias = use_bias
        self.attn_implementation = attn_implementation
        self.initializer_range = initializer_range
        self.init_merger_std = init_merger_std
        self.is_causal = is_causal
        self.post_norm = post_norm
        self.gradient_checkpointing = gradient_checkpointing


class DotsOCRConfig(Qwen2Config):
    model_type = "dots_ocr"
    def __init__(self, 
        image_token_id = 151665, 
        video_token_id = 151656,
        vision_config: Optional[dict] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_config = DotsVisionConfig(**(vision_config or {}))

    def save_pretrained(self, save_directory, **kwargs):
        self._auto_class = None
        super().save_pretrained(save_directory, **kwargs)


class DotsVLProcessor(Qwen2_5_VLProcessor):
    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.image_token = "<|imgpad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token


AutoProcessor.register("dots_ocr", DotsVLProcessor)
CONFIG_MAPPING.register("dots_ocr", DotsOCRConfig)
'''
    
    def _get_modeling_ocr_content(self) -> str:
        """Get modeling_dots_ocr.py content"""
        return '''# Content for modeling_dots_ocr.py - Implementation of DotsOCR model
# This would contain the full model implementation from the provided file
pass  # Placeholder - replace with actual content
'''
    
    def _get_modeling_vision_content(self) -> str:
        """Get modeling_dots_vision.py content"""
        return '''# Content for modeling_dots_vision.py - Vision transformer implementation
# This would contain the full vision model implementation
pass  # Placeholder - replace with actual content
'''


class DotsOCRService:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.model = None
        self.processor = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.model_path = getattr(settings, 'DOTS_OCR_MODEL_PATH', './weights/DotsOCR')
        self.downloader = ModelDownloader()
        self._initialized = False
        
        # Different prompts for different output formats
        self.prompts = {
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
        
    async def initialize(self):
        """Initialize DotsOCR model and processor with automatic download"""
        if self._initialized:
            return
            
        try:
            self.logger.info("Initializing DotsOCR model...")
            
            # Check if model exists, download if not
            if not self.downloader.check_model_exists():
                self.logger.info("DotsOCR model not found, downloading...")
                success = await self.downloader.download_model()
                if not success:
                    raise Exception("Failed to download DotsOCR model")
                
                # Setup additional model files
                await self.downloader.setup_model_files()
            
            # Load model and processor in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._load_model)
            
            self.logger.info("DotsOCR model initialized successfully")
            self._initialized = True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize DotsOCR model: {str(e)}")
            raise
    
    def _load_model(self):
        """Load the DotsOCR model and processor"""
        try:
            # Add the model directory to Python path
            sys.path.insert(0, str(Path(self.model_path).parent))
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
        except Exception as e:
            self.logger.error(f"Error loading DotsOCR model: {str(e)}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results (optional for DotsOCR)"""
        try:
            # DotsOCR typically works well with original images
            if len(image.shape) == 3:
                return image
            else:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed: {str(e)}, using original")
            return image
    
    async def _process_image_with_dots_ocr(self, image_path: str, output_format: str = "txt") -> str:
        """Process a single image with DotsOCR"""
        try:
            prompt = self.prompts.get(output_format, self.prompts["txt"])
            
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
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # Move to appropriate device
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)

            # Inference: Generation of the output
            generated_ids = self.model.generate(**inputs, max_new_tokens=24000)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            # Extract the output text (it's a list, get first element)
            result = output_text[0] if output_text else ""
            return result
            
        except Exception as e:
            self.logger.error(f"DotsOCR processing failed: {str(e)}")
            return ""
    
    @log_execution_time
    async def extract_text_from_pdf(self, pdf_path: str, enhance_quality: bool = True, output_format: str = "txt") -> OCRResult:
        """Extract text from PDF using DotsOCR"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        all_text = []
        total_confidence = 0.0
        page_count = 0
        detected_languages = set()
        
        try:
            self.logger.info(f"Starting DotsOCR processing for: {pdf_path}")
            
            # Convert PDF to images
            loop = asyncio.get_event_loop()
            images = await loop.run_in_executor(
                self.executor,
                lambda: convert_from_path(pdf_path, dpi=300)
            )
            
            page_count = len(images)
            self.logger.info(f"PDF converted to {page_count} images")
            
            # Create temporary directory for images
            temp_dir = os.path.join(os.path.dirname(pdf_path), 'temp_images')
            os.makedirs(temp_dir, exist_ok=True)
            
            try:
                for page_num, image in enumerate(images, 1):
                    self.logger.info(f"Processing page {page_num}/{page_count}")
                    
                    # Save image temporarily
                    temp_image_path = os.path.join(temp_dir, f'page_{page_num}.png')
                    image.save(temp_image_path)
                    
                    # Process with DotsOCR
                    page_text = await loop.run_in_executor(
                        self.executor,
                        lambda p=temp_image_path, f=output_format: asyncio.run(
                            self._process_image_with_dots_ocr(p, f)
                        )
                    )
                    
                    if page_text.strip():
                        all_text.append(page_text.strip())
                        confidence = self._estimate_confidence(page_text)
                        total_confidence += confidence
                        
                        # Detect language from first page with substantial text
                        if page_num == 1:
                            detected_lang = self._detect_language(page_text)
                            detected_languages.add(detected_lang)
                    
                    # Clean up temporary image
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)
                
            finally:
                # Clean up temporary directory
                if os.path.exists(temp_dir):
                    try:
                        os.rmdir(temp_dir)
                    except OSError:
                        pass
            
            # Combine all text based on output format
            if output_format == "json":
                final_text = self._combine_json_pages(all_text)
            else:
                separator = "\n\n---\n\n" if output_format == "md" else "\n\n"
                final_text = separator.join(all_text)
            
            avg_confidence = total_confidence / page_count if page_count > 0 else 0
            processing_time = time.time() - start_time
            
            primary_language = list(detected_languages)[0] if detected_languages else "ar"
            
            result = OCRResult(
                text=final_text,
                confidence=avg_confidence,
                processing_time=processing_time,
                language_detected=primary_language,
                page_count=page_count
            )
            
            self.logger.info(f"DotsOCR completed. Pages: {page_count}, Confidence: {avg_confidence:.2f}, Time: {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"DotsOCR processing failed: {str(e)}")
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=processing_time,
                language_detected="unknown",
                page_count=page_count
            )
    
    def _combine_json_pages(self, json_texts: List[str]) -> str:
        """Combine multiple JSON pages into a single JSON structure"""
        try:
            combined_elements = []
            for page_num, json_text in enumerate(json_texts, 1):
                try:
                    page_data = json.loads(json_text)
                    if isinstance(page_data, dict) and 'elements' in page_data:
                        elements = page_data['elements']
                    elif isinstance(page_data, list):
                        elements = page_data
                    else:
                        elements = [page_data]
                    
                    # Add page number to each element
                    for element in elements:
                        if isinstance(element, dict):
                            element['page'] = page_num
                    
                    combined_elements.extend(elements)
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to parse JSON from page {page_num}")
                    continue
            
            return json.dumps({"pages": len(json_texts), "elements": combined_elements}, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to combine JSON pages: {str(e)}")
            return json.dumps({"error": "Failed to combine pages", "pages": json_texts}, indent=2)
    
    def _estimate_confidence(self, text: str) -> float:
        """Estimate confidence based on text characteristics"""
        if not text.strip():
            return 0.0
        
        text_length = len(text.strip())
        word_count = len(text.split())
        
        base_confidence = min(0.9, 0.5 + (text_length / 1000) * 0.3)
        word_factor = min(1.0, word_count / 50)
        
        return base_confidence * word_factor
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character patterns"""
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06ff')
        english_chars = sum(1 for char in text if char.isalpha() and ord(char) < 128)
        
        if arabic_chars > english_chars:
            return "ar"
        else:
            return "en"
    
    async def extract_text_from_image(self, image_data: bytes, output_format: str = "txt") -> str:
        """Extract text from a single image"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Convert bytes to PIL Image and save temporarily
            image = Image.open(io.BytesIO(image_data))
            temp_path = f"/tmp/temp_image_{int(time.time())}.png"
            image.save(temp_path)
            
            try:
                # Process with DotsOCR
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    lambda: asyncio.run(self._process_image_with_dots_ocr(temp_path, output_format))
                )
                return result
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
        except Exception as e:
            self.logger.error(f"Image OCR failed: {str(e)}")
            return ""
    
    async def extract_with_layout(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text with detailed layout information using JSON format"""
        result = await self.extract_text_from_pdf(pdf_path, output_format="json")
        
        try:
            if result.text:
                layout_data = json.loads(result.text)
                return {
                    "layout": layout_data,
                    "confidence": result.confidence,
                    "processing_time": result.processing_time,
                    "language_detected": result.language_detected,
                    "page_count": result.page_count
                }
        except json.JSONDecodeError:
            self.logger.error("Failed to parse layout JSON")
        
        return {
            "layout": {},
            "confidence": result.confidence,
            "processing_time": result.processing_time,
            "language_detected": result.language_detected,
            "page_count": result.page_count
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        # Clear GPU memory
        if self.model is not None:
            del self.model
            del self.processor
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        self._initialized = False


# Global OCR service instance
ocr_service = DotsOCRService()