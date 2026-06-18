# 專案行為與選擇紀錄 (Project Action & Decision Log)

本文件紀錄了在開發「SVM Kernel Trick 教學專案」過程中的所有關鍵行為與設計選擇。

## 1. 需求分析與初始規劃
- **行為**: 讀取並深入分析 `claude.md` 與 `design.md`。
- **選擇**: 確定專案核心目標為「以視覺化方式解釋 SVM 的 Kernel Trick」，並決定採用**雙軌並行策略**：
  1. **Manim 動畫指令碼**: 撰寫高畫質的 3D 數學教學動畫，用於概念講解。
  2. **Streamlit 互動網頁**: 建立能讓使用者親自調整 SVM 超參數，並即時觀察 3D 決策曲面變化的互動儀表板。

## 2. 核心邏輯與資料集設計
- **行為**: 實作 `svm_kernel_demo.py`，將機器學習邏輯與 Manim 動畫封裝於同一個腳本中。
- **選擇**: 
  - **資料集**: 選擇 `sklearn.datasets.make_circles` 生成內外兩層的同心圓資料。這是因為線性不可分的同心圓最適合用來展示 RBF 與 Poly 核函數將空間扭曲、升維的特性。
  - **升維函數 (Explicit Lifting)**: 為了在動畫中具象化「映射到高維空間」，刻意選擇了直觀的數學映射 $z = x^2 + y^2$，讓內圈與外圈在 $Z$ 軸上產生高度落差，以便用 2D 平面 ($z=0$) 進行切割。

## 3. 網頁互動儀表板 (Streamlit + Plotly)
- **行為**: 建立 `streamlit_app.py` 應用程式並設計多個 UI 分頁。
- **選擇**: 
  - **網頁框架**: 選擇 `Streamlit`。相較於 React 等傳統前端框架，Streamlit 允許完全使用 Python 撰寫，能無縫接合 `scikit-learn` 與 `numpy`，極大化開發效率。
  - **3D 視覺化工具**: 選擇 `Plotly`（而非 Matplotlib），因為它支援網頁端的原生 3D 互動體驗（平移、旋轉、縮放），非常適合展示 3D 資料點與決策曲面。
  - **決策曲面渲染**: 為了呈現真實的 SVM 模型行為，選擇使用 `numpy.meshgrid` 建立底層網格，並呼叫模型的 `decision_function(X)` 來計算 $Z$ 軸高度，繪製出連續的 3D 地形圖（Landscape），而不僅僅是資料點而已。

## 4. 環境建置與服務啟動
- **行為**: 生成並整理 `requirements.txt`、撰寫 `README.md` 教學，並在背景執行套件安裝（`pip install plotly`）與啟動網頁伺服器（`streamlit run streamlit_app.py`）。
- **選擇**: 明確隔離依賴套件（包含 `manim`, `plotly`, `streamlit`, `scikit-learn` 等），確保專案在不同環境下皆能順利重現與啟動。

## 5. 自動化驗證與品質測試
- **行為**: 啟動 Antigravity 內建的 Browser Subagent，模擬真實使用者存取 `http://localhost:8501/`。
- **選擇**:
  - 不僅檢查伺服器是否啟動，還要求 Agent 點擊所有分頁（2D Boundary, 3D Lifting, 3D Surface）。
  - 刻意調整側邊欄參數（將 Kernel Type 更改為 `poly`），驗證 UI 的連動性與後端模型的即時重新訓練，最終確認所有互動功能與 3D 渲染皆完美運作。
