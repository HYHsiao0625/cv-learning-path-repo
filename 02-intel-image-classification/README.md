# 專案 02: Intel 自然場景分類

## 專案目標

應用進階的「遷移學習」(Transfer Learning) 技術，來解決一個真實世界的多類別場景分類問題。目標是載入一個在 ImageNet 上預訓練好的強大模型，並對其進行微調 (Fine-tuning)，以適應我們新的、相對較小的資料集。

## 使用資料集

- **Intel Image Classification**: 一個包含約 17,000 張 150x150 彩色圖片的自然場景資料集。
- **6 個類別**: `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`。
- **資料結構**: 圖片已按類別存放在不同的資料夾中，非常適合使用 `torchvision.datasets.ImageFolder` 進行載入。

## 學習到的關鍵技術

1.  **遷移學習 (Transfer Learning)**
    - 深刻理解了遷移學習的核心思想：不從零開始，而是站在巨人的肩膀上，利用預訓練模型已經學到的通用視覺知識。
    - 學習了如何從 `torchvision.models` 載入一個著名的預訓練模型 (ResNet-18)。

2.  **模型微調 (Fine-tuning)**
    - 掌握了遷移學習的標準流程：
        a. **凍結參數**：透過設定 `param.requires_grad = False` 來凍結預訓練模型的特徵提取層，保護其學到的知識不被破壞。
        b. **替換分類頭**：將模型原有的、為 ImageNet 1000 類設計的 `fc` 層，替換成一個符合我們 6 個類別的新全連接層。
    - 在設定優化器時，只將需要訓練的新分類頭的參數 (`model.fc.parameters()`) 傳遞給它，實現更高效的訓練。

3.  **處理真實世界圖片**
    - 學習了針對預訓練模型的標準圖片預處理流程 (`transforms.Compose`)，包括：
        a. **尺寸調整**: `Resize` 和 `Crop` 至模型需要的 224x224 尺寸。
        b. **標準化**: 使用 ImageNet 的平均值和標準差 (`transforms.Normalize`) 對圖片進行標準化，這是使用預訓練模型的關鍵一步。

## 最終成果

- 透過遷移學習，僅用 5 個 Epoch 的訓練，模型就在 3000 張測試圖片上達到了 **90.60%** 的準確率。

## 核心想法筆記

- Conda 與 Pip/Poetry 的環境管理是深度學習專案的常見痛點。建立穩定環境的黃金法則是：**讓 Conda 負責管理所有需要 C/C++ 編譯的複雜套件 (PyTorch, Pillow, Matplotlib 等)，避免混用**。
- 評估模型時，**絕對不能**使用訓練集 (`train_loader`)，否則無法得知模型的泛化能力。必須使用獨立的測試集 (`test_loader`)。