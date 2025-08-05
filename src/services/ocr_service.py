# # services/ocr_service.py
# import easyocr
# import cv2
# import numpy as np
# from pdf2image import convert_from_path
# from PIL import Image
# import io
# import time
# from typing import List, Tuple, Optional
# import asyncio
# from concurrent.futures import ThreadPoolExecutor

# from config.config_settings import settings
# from config.custom_logger import get_logger, log_execution_time, CustomLoggerTracker

# from models.schemas import OCRResult

# class OCRService:
#     def __init__(self):
#         self.logger = get_logger(__name__)
#         self.reader = None
#         self.executor = ThreadPoolExecutor(max_workers=2)
        
#     async def initialize(self):
#         """Initialize EasyOCR reader"""
#         try:
#             self.logger.info("Initializing EasyOCR reader...")
#             loop = asyncio.get_event_loop()
#             self.reader = await loop.run_in_executor(
#                 self.executor,
#                 lambda: easyocr.Reader(
#                     settings.OCR_LANGUAGES,
#                     gpu=settings.OCR_GPU
#                 )
#             )
#             self.logger.info("EasyOCR reader initialized successfully")
#         except Exception as e:
#             self.logger.error(f"Failed to initialize OCR reader: {str(e)}")
#             raise
    
#     def preprocess_image(self, image: np.ndarray) -> np.ndarray:
#         """Preprocess image for better OCR results"""
#         try:
#             # Convert to grayscale
#             if len(image.shape) == 3:
#                 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             else:
#                 gray = image
            
#             # Apply Gaussian blur to reduce noise
#             blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
#             # Apply adaptive thresholding
#             binary = cv2.adaptiveThreshold(
#                 blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                 cv2.THRESH_BINARY, 11, 2
#             )
            
#             # Morphological operations to clean up
#             kernel = np.ones((2, 2), np.uint8)
#             cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
#             return cleaned
#         except Exception as e:
#             self.logger.warning(f"Image preprocessing failed: {str(e)}, using original")
#             return image
    
#     @log_execution_time
#     async def extract_text_from_pdf(self, pdf_path: str, enhance_quality: bool = True) -> OCRResult:
#         """Extract text from PDF using OCR"""
#         if not self.reader:
#             await self.initialize()
        
#         start_time = time.time()
#         all_text = []
#         total_confidence = 0
#         page_count = 0
#         detected_languages = set()
        
#         try:
#             self.logger.info(f"Starting OCR processing for: {pdf_path}")
            
#             # Convert PDF to images
#             loop = asyncio.get_event_loop()
#             images = await loop.run_in_executor(
#                 self.executor,
#                 lambda: convert_from_path(pdf_path, dpi=300)
#             )
            
#             page_count = len(images)
#             self.logger.info(f"PDF converted to {page_count} images")
            
#             for page_num, image in enumerate(images, 1):
#                 self.logger.info(f"Processing page {page_num}/{page_count}")
                
#                 # Convert PIL image to numpy array
#                 img_array = np.array(image)
                
#                 # Preprocess image if enhancement is enabled
#                 if enhance_quality:
#                     img_array = self.preprocess_image(img_array)
                
#                 # Perform OCR
#                 results = await loop.run_in_executor(
#                     self.executor,
#                     lambda: self.reader.readtext(img_array, detail=1)
#                 )
                
#                 page_text = []
#                 page_confidence = []
                
#                 for (bbox, text, confidence) in results:
#                     if confidence > 0.5:  # Filter low confidence text
#                         page_text.append(text.strip())
#                         page_confidence.append(confidence)
                
#                 if page_text:
#                     all_text.append(" ".join(page_text))
#                     total_confidence += sum(page_confidence) / len(page_confidence)
                
#                 # Detect language from first page with substantial text
#                 if page_num == 1 and page_text:
#                     detected_lang = self._detect_language(" ".join(page_text))
#                     detected_languages.add(detected_lang)
            
#             # Combine all text
#             final_text = "\n\n".join(all_text)
#             avg_confidence = total_confidence / page_count if page_count > 0 else 0
#             processing_time = time.time() - start_time
            
#             # Determine primary language
#             primary_language = list(detected_languages)[0] if detected_languages else "ar"
            
#             result = OCRResult(
#                 text=final_text,
#                 confidence=avg_confidence,
#                 processing_time=processing_time,
#                 language_detected=primary_language,
#                 page_count=page_count
#             )
            
#             self.logger.info(f"OCR completed. Pages: {page_count}, Confidence: {avg_confidence:.2f}, Time: {processing_time:.2f}s")
#             return result
            
#         except Exception as e:
#             processing_time = time.time() - start_time
#             self.logger.error(f"OCR processing failed: {str(e)}")
#             return OCRResult(
#                 text="",
#                 confidence=0.0,
#                 processing_time=processing_time,
#                 language_detected="unknown",
#                 page_count=page_count
#             )
    
#     def _detect_language(self, text: str) -> str:
#         """Simple language detection based on character patterns"""
#         arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06ff')
#         english_chars = sum(1 for char in text if char.isalpha() and ord(char) < 128)
        
#         if arabic_chars > english_chars:
#             return "ar"
#         else:
#             return "en"
    
#     async def extract_text_from_image(self, image_data: bytes) -> str:
#         """Extract text from a single image"""
#         if not self.reader:
#             await self.initialize()
        
#         try:
#             # Convert bytes to PIL Image
#             image = Image.open(io.BytesIO(image_data))
#             img_array = np.array(image)
            
#             # Preprocess
#             img_array = self.preprocess_image(img_array)
            
#             # OCR
#             loop = asyncio.get_event_loop()
#             results = await loop.run_in_executor(
#                 self.executor,
#                 lambda: self.reader.readtext(img_array, detail=1)
#             )
            
#             # Extract text with confidence > 0.5
#             text_parts = [text for (_, text, conf) in results if conf > 0.5]
#             return " ".join(text_parts)
            
#         except Exception as e:
#             self.logger.error(f"Image OCR failed: {str(e)}")
#             return ""
    
#     def cleanup(self):
#         """Cleanup resources"""
#         if self.executor:
#             self.executor.shutdown(wait=True)

# # Global OCR service instance
# ocr_service = OCRService()


