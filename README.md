# SVM Kernel Trick 3D Manim Animation Demo

🔗 **Live Web App**: [https://l13-svm-nrhrc5zu6cnhth333cd2f8.streamlit.app/](https://l13-svm-nrhrc5zu6cnhth333cd2f8.streamlit.app/)
🐙 **GitHub Repository**: [https://github.com/willest5555-cmyk/L13-SVM](https://github.com/willest5555-cmyk/L13-SVM)

This project provides a complete Manim Community Edition animation demonstrating the mathematical concept of the **Support Vector Machine (SVM) Kernel Trick**, alongside an interactive 3D web application.

It features an educational visualization in three distinct phases:
1. **Phase 1: 2D Non-linear Separability** — Demonstrating why concentric circle data cannot be separated by any single straight line.
2. **Phase 2: Explicit 3D Lifting** — Visualizing the geometric intuition of mapping 2D data into 3D space via the transformation $z = x^2 + y^2$ and showing how a horizontal hyperplane ($z = 0.55$) separates the data.
3. **Phase 3: Real RBF SVM Decision Surface** — Training a real Support Vector Classifier (SVC) using the Radial Basis Function (RBF) kernel and plotting its decision value $f(x, y)$ as a 3D landscape, highlighting support vectors on the decision boundary ($f(x, y) = 0$).

---

## 📐 Mathematical Concepts

### 1. Explicit 3D Lifting
We apply the mapping:
$$\phi(x, y) = (x, y, x^2 + y^2)$$

This maps points close to the origin to lower heights (near $z = 0$) and points far away to greater heights (near $z = 1$). In this lifted 3D space, a horizontal plane $z = c$ (where $0.12 < c < 1.0$) separates the two classes.

### 2. Radial Basis Function (RBF) Kernel
In practice, SVMs do not explicitly compute high-dimensional coordinates. Instead, they use the **kernel trick** to evaluate inner products directly in a high-dimensional (often infinite-dimensional) Hilbert space:
$$K(x, x_i) = \exp(-\gamma \|x - x_i\|^2)$$

Here, $\gamma$ is a hyperparameter controling the radius of influence of individual training samples.

### 3. SVM Decision Function
The trained classifier evaluates test points using the formula:
$$f(x) = \sum_{i \in \text{SVs}} \alpha_i y_i K(x, x_i) + b$$

The decision boundary is the level set where $f(x) = 0$. In 3D space, we plot $z = f(x, y)$ to observe the decision boundary landscape.

---

## 🚀 Installation & Setup

1. **Clone or Navigate** to this folder:
   ```bash
   cd "d:/AICourse26/ML/L13 SVM"
   ```

2. **Install dependencies**:
   Ensure you have Python 3.8+ installed, and run:
   ```bash
   pip install -r requirements.txt
   ```

3. **FFmpeg Installation (Required by Manim)**:
   Manim requires `ffmpeg` to encode animations.
   - **Windows**: Install using winget:
     ```powershell
     winget install Gyan.FFmpeg
     ```
     *(Restart your terminal after installation to update the PATH variable).*
   - **macOS**: Install using Homebrew:
     ```bash
     brew install ffmpeg
     ```
   - **Linux**: Install via package manager:
     ```bash
     sudo apt install ffmpeg
     ```

---

## 🌐 Running the Streamlit Web Application

We provide an interactive local Streamlit website to visualize the SVM kernel trick. You can adjust hyperparameters and visualize the results dynamically in your browser.

Run the following command to start the web app:
```bash
streamlit run streamlit_app.py
```
*(Once started, the website will open in your browser, usually at `http://localhost:8501`)*

---

## 🎬 Rendering the Animation

Use the following commands to render the video. The output files will be saved under `./media/videos/svm_kernel_demo/`.

### 1. Fast Preview (Low Resolution, 480p 15FPS)
Great for testing:
```bash
manim -pql svm_kernel_demo.py SVMKernelFullDemo
```

### 2. Production Render (High Quality, 1080p 60FPS)
For presentations and sharing:
```bash
manim -pqh svm_kernel_demo.py SVMKernelFullDemo
```

---

## 💡 Important Educational Note

> [!WARNING]
> The explicit lifting $z = x^2 + y^2$ shown in Phase 2 is an **educational visualization** to explain the geometric concept of high-dimensional separability.
>
> In reality, the RBF kernel maps data into an **infinite-dimensional** feature space, which cannot be directly graphed in 3D.
>
> Phase 3 resolves this by plotting the **decision function landscape $z = f(x, y)$** rather than raw feature coordinates, demonstrating the actual RBF SVM model output on the same 2D data.

---

## 📝 Project Development Log & Key Decisions

This project was built with a dual-track strategy to provide both high-quality educational animations and an interactive learning experience. Below is a summary of the key development decisions:

- **Dual-Track Pedagogy**: Designed a Manim script for narrative-driven 3D mathematical animations, and a Streamlit interactive web app for hands-on hyperparameter exploration.
- **Data & Mapping**: Chose `sklearn.datasets.make_circles` to generate non-linearly separable data. Used an explicit geometric lifting function $z = x^2 + y^2$ to visually demonstrate how data is mapped to higher dimensions.
- **Interactive 3D Rendering**: Selected `Plotly` combined with `Streamlit` to render the actual SVM decision landscape $z = f(x, y)$ using `numpy.meshgrid`, allowing users to dynamically visualize the mathematical surface of the fitted model.
- **Automated Verification**: The Streamlit interface, including 3D plots and UI widgets, was successfully verified using an AI browser subagent to ensure seamless interactivity across all kernel types (RBF, Poly, Linear).
- **Cloud Deployment**: Successfully deployed to Streamlit Community Cloud, handling underlying system dependencies (e.g., `ffmpeg`, `libcairo2-dev`) and pure ASCII encoding requirements for stable `apt-get` installation.

*For a detailed breakdown of all actions and choices, please refer to the full log in [`./resources/log.md`](./resources/log.md).*
