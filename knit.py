import cv2
import numpy as np
import os
from google.colab.patches import cv2_imshow

# 画像の読み込み
image_path = '/content/null.png'
image = cv2.imread(image_path)

# 前処理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# 輪郭の検出
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# セル(四角)を探す
cell_contours = []
min_area = 10
max_area = 10000

for cnt in contours:
    area = cv2.contourArea(cnt)
    if min_area < area < max_area:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        if len(approx) == 4:
            cell_contours.append(cnt)

# セルの数を数える
num_cells = len(cell_contours)
print(f"検出されたセルの総数: {num_cells}")

# セルの中心座標とバウンディングボックス検出
cell_info = []
for cnt in cell_contours:
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(cnt)
        cell_info.append({'center': (cx, cy), 'bbox': (x, y, w, h)})

# 座標のクラスタリング関数
def coords_counter(coords, threshold):
    if not coords:
        return []

    sorted_coords = sorted(list(set(coords)))
    clusters = []
    current_cluster = [sorted_coords[0]]

    for i in range(1, len(sorted_coords)):
        if sorted_coords[i] - current_cluster[-1] < threshold:
            current_cluster.append(sorted_coords[i])
        else:
            clusters.append(int(np.mean(current_cluster)))
            current_cluster = [sorted_coords[i]]
    clusters.append(int(np.mean(current_cluster)))
    return clusters

x_coords = [info['center'][0] for info in cell_info]
y_coords = [info['center'][1] for info in cell_info]

# クラスタリングの閾値
x_clusters = coords_counter(x_coords, threshold=10)
y_clusters = coords_counter(y_coords, threshold=10)

num_cols = len(x_clusters)
num_rows = len(y_clusters)

print(f"検出されたグリッドの横の数: {num_cols}")
print(f"検出されたグリッドの縦の数: {num_rows}")

# num_rowsとnum_colsで配列を作成
amizu = [[None for _ in range(num_cols)] for _ in range(num_rows)]

for info in cell_info:
    cx, cy = info['center']
    x_idx = np.argmin([abs(cx - x_c) for x_c in x_clusters])
    y_idx = np.argmin([abs(cy - y_c) for y_c in y_clusters])
    # セルが存在する位置にバウンディングボックスを格納
    amizu[y_idx][x_idx] = info['bbox']

print("\nグリッドの配列:")

for row in amizu:
    print([ '-' if cell is None else cell for cell in row ])

# --- セルごとに画像を分割して保存 ---
output_dir = '/content/cells'
os.makedirs(output_dir, exist_ok=True)

for row_idx, row in enumerate(amizu):
    for col_idx, bbox in enumerate(row):
        if bbox is not None:
            x, y, w, h = bbox
            cell_image = image[y:y+h, x:x+w]
            filename = os.path.join(output_dir, f'cell_{row_idx+1}_{col_idx+1}.png')
            cv2.imwrite(filename, cell_image)

print(f"\n検出されたセル画像を '{output_dir}' に保存しました。")
# --- セルごとに画像を分割して保存 ---

# 結果の可視化
output_image = image.copy()
cv2.drawContours(output_image, cell_contours, -1, (0, 255, 0), 2)

if x_clusters:
    for x in x_clusters:
        cv2.line(output_image, (x, 0), (x, output_image.shape[0]), (255, 0, 0), 2)

if y_clusters:
    for y in y_clusters:
        cv2.line(output_image, (0, y), (output_image.shape[1], y), (0, 0, 255), 2)

cv2_imshow(output_image)

import pandas as pd
from tensorflow import keras

model_path = '/content/drive/MyDrive/knit/knit3.keras'
load_model = keras.models.load_model(model_path)

class_label = sorted(os.listdir('/content/drive/MyDrive/knit/knit'))
print("モデルが認識するクラス:", class_label)

# 2. 分割した画像を1枚ずつ読み込み、推論する
amizu_predict = [[None for _ in range(num_cols)] for _ in range(num_rows)]

for row_idx, row in enumerate(amizu):
    for col_idx, bbox in enumerate(row):
        if bbox is not None:
            # 画像読み込み
            image_filename = os.path.join(output_dir, f'cell_{row_idx+1}_{col_idx+1}.png')
            
            # 画像の前処理
            img = cv2.imread(image_filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img_array = img / 255.0
            img_array = np.expand_dims(img_array, axis=0) # バッチ次元を追加

            # 推論を実行
            predictions = load_model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            predicted_class_name = class_label[predicted_class_index]
            
            amizu_predict[row_idx][col_idx] = predicted_class_name
            
            print(f"cell ({row_idx+1}, {col_idx+1}): {predicted_class_name} (確率: {predictions[0][predicted_class_index]:.2f})")

# 結果の配列を表示
print("\n結果:")
for row in amizu_predict:
    print(row)

df_amizu = pd.DataFrame(amizu_predict)
df_amizu.fillna('-', inplace=True)

header_info = [f"#rows,{num_rows}", f"#cols,{num_cols}"]
header_df = pd.DataFrame(header_info)

# ヘッダーとデータを結合
final_df = pd.concat([header_df, df_amizu], ignore_index=True)

output_csv_path = '/content/amizu_predicted.csv'
final_df.to_csv(output_csv_path, index=False, header=False, encoding='utf-8')

print(f"\n結果が '{output_csv_path}' に保存されました。")