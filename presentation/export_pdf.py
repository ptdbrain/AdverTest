import time
import os
from pathlib import Path
from playwright.sync_api import sync_playwright
from PIL import Image
import io

def export_to_pdf():
    html_path = Path(__file__).parent.resolve() / "advertest-demo.html"
    pdf_path = Path(__file__).parent.resolve() / "AdverTest-PitchDeck.pdf"
    
    screenshots = []
    
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=True)
        # Set viewport exactly to 1920x1080 for perfect scale
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        
        # Load local HTML file
        # Convert path to valid file URI format
        file_uri = html_path.as_posix()
        if not file_uri.startswith('/'):
            file_uri = '/' + file_uri
            
        print(f"Loading: file://{file_uri}")
        page.goto(f"file://{file_uri}", wait_until="networkidle")
        
        # Give fonts and images some time to load visually
        page.wait_for_timeout(3000)
        
        # The presentation has 9 slides
        for i in range(9):
            # Allow animations to settle
            page.wait_for_timeout(1000)
            
            # Take screenshot
            screenshot_bytes = page.screenshot(type="jpeg", quality=95)
            
            # Convert bytes to PIL Image
            img = Image.open(io.BytesIO(screenshot_bytes))
            img = img.convert('RGB')
            screenshots.append(img)
            
            # Go to next slide
            page.keyboard.press("ArrowRight")
            
        browser.close()
        
    if screenshots:
        print(f"Exporting {len(screenshots)} slides to {pdf_path.name}...")
        screenshots[0].save(
            pdf_path,
            save_all=True,
            append_images=screenshots[1:],
            resolution=100.0
        )
        print(f"Successfully saved to {pdf_path}")

if __name__ == "__main__":
    export_to_pdf()
