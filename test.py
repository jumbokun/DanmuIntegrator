from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 初始化OCR
ocr_result = ocr.ocr("你那张图的路径.png", cls=True)

for line_group in ocr_result:
    for box, (text, confidence) in line_group:
        if confidence > 0.5:
            print(f"{text} ({confidence:.2f})")
