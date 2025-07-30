#!/usr/bin/env python3
"""
PDF Image Extractor
Extracts all images from a PDF file and saves them to a new folder.
"""

import os
import sys
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io

def extract_images_from_pdf(pdf_path, output_folder=None):
    """
    Extract all images from a PDF file and save them to a folder.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_folder (str, optional): Output folder path. If None, creates folder based on PDF name.
    
    Returns:
        tuple: (success_count, total_images, output_folder_path)
    """
    
    # Validate PDF file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Create output folder
    if output_folder is None:
        pdf_name = Path(pdf_path).stem
        output_folder = f"{pdf_name}_images"
    
    # Create the output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the PDF
    pdf_document = fitz.open(pdf_path)
    
    image_count = 0
    success_count = 0
    
    print(f"Processing PDF: {pdf_path}")
    print(f"Output folder: {output_folder}")
    print("-" * 50)
    
    # Single loop through pages - no nested loops
    for page_num in range(1):
        page = pdf_document[page_num]
        
        # Get all images from this page and process them immediately
        page_images = page.get_images(full=True)
        
        if page_images:
            print(f"Found {len(page_images)} image(s)")
            
            # Save all images from this page with one operation
            for img_data in page_images:
                try:
                    xref = img_data[0]
                    pix = fitz.Pixmap(pdf_document, xref)
                    
                    # Skip tiny images
                    if pix.width < 50 or pix.height < 50:
                        pix = None
                        continue
                    
                    image_count += 1
                    filename = f"page_{image_count:03d}.png"
                    img_path = os.path.join(output_folder, filename)
                    
                    # Convert and save
                    if pix.n - pix.alpha >= 4:  # CMYK
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    
                    pix.save(img_path)
                    success_count += 1
                    print(f"  Saved: {filename} ({pix.width}x{pix.height})")
                    pix = None
                    
                except Exception as e:
                    print(f"  Error on page {page_num + 1}: {e}")
        else:
            print(f"Page {page_num + 1}: No images found")
    
    pdf_document.close()
    
    print("-" * 50)
    print(f"Extraction complete!")
    print(f"Successfully extracted: {success_count}/{image_count} images")
    print(f"Images saved to: {os.path.abspath(output_folder)}")
    
    return success_count, image_count, output_folder

def main():
    """Main function to handle command line arguments and run the extraction."""
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_image_extractor.py <pdf_file> [output_folder]")
        print("Example: python pdf_image_extractor.py document.pdf")
        print("Example: python pdf_image_extractor.py document.pdf my_images")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        success_count, total_images, output_path = extract_images_from_pdf(pdf_path, output_folder)
        
        if success_count == 0:
            print("\nNo images were found or extracted from the PDF.")
        else:
            print(f"\n✅ Successfully extracted {success_count} images!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()