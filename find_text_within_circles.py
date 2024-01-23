import cv2
import numpy as np
import pytesseract

# 再度画像を読み込む
image_path = '/Users/kimuratoshiyuki/Dropbox/Python/TextCircleDetection/text_within_circles.png'
image = cv2.imread(image_path)

# グレースケール画像に変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# メディアンフィルタを適用してノイズを減らす
median = cv2.medianBlur(gray, 5)

# 二値化を行い、円をより明確にする
_, binary = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 円を検出
circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                           param1=50, param2=15, minRadius=10, maxRadius=50)

# 検出された円がある場合のみ処理を行う
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    circle_count = 1  # 円のカウンター

    for (x, y, r) in circles:
        # cv2.circle(image, (x, y), r, (0, 255, 0), 4)
        # 円の中心に矩形を描画しROIを作成
        roi = binary[y - r:y + r, x - r:x + r]
        
        # ROIを使用してTesseractによるテキスト認識を行う
        text = pytesseract.image_to_string(roi, config='--psm 6')

        # テキストの有無に応じてメッセージを出力
        if text.strip() != "":  # テキストが検出された場合
            print(f"円{circle_count}: 文字列が検出されました。")
        else:  # テキストが検出されなかった場合
            print(f"円{circle_count}: 文字列がありませんでした。")

        circle_count += 1
        
        # テキストが検出された場合のみ、円を緑色で描画
        if text.strip() != "":  # 任意のテキストが検出された場合
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)

# 処理結果の画像を保存
output_path = '/Users/kimuratoshiyuki/Dropbox/Python/TextCircleDetection/detected_circles_with_text.png'
cv2.imwrite(output_path, image)

output_path
