import cv2
import numpy as np

# 再度画像を読み込む
image_path = '/Users/kimuratoshiyuki/Dropbox/Python/TextCircleDetection/text_within_circles.png'
image = cv2.imread(image_path)

# グレースケール画像に変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ノイズを低減しながらエッジを強調するためのメディアンフィルタを適用
median_filtered = cv2.medianBlur(gray, 5)

# HoughCirclesのパラメータを再調整して円を検出
# dpの値を下げて解像度を上げ、minDistを減らして円同士が近い場合でも検出できるようにする
# param1 (Cannyエッジ検出器の高い閾値) とparam2 (中心検出の閾値) を調整する
circles = cv2.HoughCircles(median_filtered, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                           param1=50, param2=15, minRadius=10, maxRadius=50)

# 検出した円を描画
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)

# 処理結果の画像を保存
output_path = '/Users/kimuratoshiyuki/Dropbox/Python/TextCircleDetection/detected_circles.png'
cv2.imwrite(output_path, image)

# 検出された円の数と画像のパスを返す
detected_circles = circles.shape[0] if circles is not None else 0
output_path, detected_circles
