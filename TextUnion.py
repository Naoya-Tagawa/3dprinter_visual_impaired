import pytesseract
from PIL import Image, ImageDraw

text1 = "GGGGHHHko"
text2 = "Hko.code"

# 重複を排除して2つのテキストを結合
result = text1 + text2.lstrip(text1)

print(result)

texts = ["GGGGHHHko", "Hko.code", "code.is", "is.awesome"]

# 重複を排除して複数のテキストを結合
result = texts[0]
for text in texts[1:]:
    result += text.lstrip(result)

print(result)

img_path = "./ProcessingDisplay/mask_frame_1699191587.5858126.jpg"
img = Image.open(img_path)
box = pytesseract.image_to_boxes(img)
print(box)
# 画像のコピーを作成
image_with_boxes = img.copy()
draw = ImageDraw.Draw(image_with_boxes)

# テキスト位置情報を使用して枠を描画
for b in box.splitlines():
    b = b.split()
    char = b[0]
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])

    # 枠を描画
    draw.rectangle([x, img.height - y, w, img.height - h], outline="red", width=2)

# 枠を描画した画像を保存または表示
image_with_boxes.show()
