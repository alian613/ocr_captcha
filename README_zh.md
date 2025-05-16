# OCR CAPTCHA

本專案是基於 PyTorch 撰寫的 CAPTCHA 驗證碼光學辨識(OCR)工具，可用於 **訓練** 及 **驗證** CAPTCHA 辨識模型，
並提供一個簡單的 CLI 工具，用於執行產生圖片、模型訓練、辨識與評估等功能。

📖 [English Documentation](README.md)

## ✨ Features | 功能特色

- ✅ 支援產生驗證碼圖片
- ✅ 支援 CUDA/GPU (若系統支援 CUDA，將自動使用 GPU 加速模型訓練與預測)
- ✅ 支援使用訓練好的模型辨識驗證碼
- ✅ 支援自訂字元集、字元長度與圖片大小及各類訓練用參數
- ✅ 支援單一圖片或資料夾進行批次預測
- ✅ 支援模型評估功能以計算準確率
- ✅ 提供簡易的 CLI 操作

---

## 💾 安裝

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start | 快速開始

### 1. 產生驗證碼圖片
   ```bash
   python generate.py
   ```

或使用 CLI 工具
   ```base
   python main.py generate
   ```

> 查看參數說明 使用 ```--help```

### 2. 訓練模型
   ```bash
   python train.py
   ```

或使用 CLI 工具
   ```base
   python main.py train
   ```

> 查看參數說明 使用 ```--help```


### 3. 辨識 CAPTCHA
將要辨識的圖片放到 `./data/pred`
   ```bash
   python predict.py ./data/pred
   ```

或使用 CLI 工具
   ```base
   python main.py predict ./data/pred
   ```

> 查看參數說明 使用 ```--help```

---

## 📂 Project Structure | 專案結構
ocr-captcha/  
├── cli.py                    # CLI 主程式  
├── config.py                 # 讀取參數設定檔  
├── config.yaml               # 預設參數設定檔  
├── train.py                  # 模型訓練  
├── predict.py                # CAPTCHA辨識  
├── evaluate.py               # 評估模型準確率  
├── codec.py                  # 編碼與解碼工具  
├── model.py                  # CNN 模型  
├── dataset.py                # 資料集與預處理(可將資料集存在記憶體或資料夾)  
├── generate.py               # CAPTCHA產生器  
├── requirements.txt          # 專案所需套件清單  
├── data/                     # CAPTCHA資料夾(預設)  
│   ├── eval/                 # 評估用圖片(預設)  
│   ├── pred/                 # 預測用圖片(預設)  
│   └── raw/                  # 訓練用圖片(預設)  
└── model/                    # 儲存模型的資料夾(預設)  


---

## 💡小貼士 | Tips
- 使用 --cache 加速訓練效率
> 執行 `train.py` 可加上 `--cache` 參數，將資料集載入至記憶體（RAM），避免每次從磁碟讀取資料，提升訓練效率，且不會占用磁碟空間，
但需確保系統擁有足夠的記憶體供使用

- 模型準確率參考值
> `train.py --dataset-size 100000 --epochs 30`，訓練出來的模型在相同資料集上的評估準確率可達 **99.5%** 以上，
但若使用該模型來預測不同資料集的驗證碼圖片，準確率可能下降至約 80% 左右。

- 留意`--eval-acc-threshold`參數
> 若未特別修改 config.yaml，其預設值為 75，
在訓練過程中，若某次評估結果低於此準確率門檻，將自動中止該次訓練並進入下一個 epoch，以節省時間並加快模型收斂。

- 透過 `config.yaml` 設定預設參數：
> 包含模型儲存與載入路徑、圖片尺寸、驗證碼長度、字元集等皆可在 config.yaml 中進行設定。
若執行 CLI 時未手動指定相關參數的值時，將自動套用 config.yaml 中的值。詳細說明請參考 config.yaml 說明。


---


## 🙋‍♀️ 聯絡方式
若有任何問題或建議，歡迎透過 GitHub Issues 提出！


---


## ⭐ Support My Open Source Project
If you appreciate my work, consider ⭐ starring this repository or buying me a coffee to support development.
Your support means a lot to me — thank you!

### [Ko-fi Support](https://ko-fi.com/alian613)

如果你覺得這個專案對你有幫助，歡迎給個 ⭐，也歡迎請我喝杯咖啡，非常感謝 ~


---


## 📄 License | 授權條款
本專案採用 MIT 授權條款。
This project is licensed under the MIT License.



