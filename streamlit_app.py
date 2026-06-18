import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="SVM Kernel Trick 3D Visualizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [data-testid="stSidebar"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF4B4B, #7928CA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #64717d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .card {
        background-color: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #7928CA;
    }
</style>
""", unsafe_allow_html=True)

# Main App Header
st.markdown('<div class="main-title">SVM Kernel Trick 3D Interactive Visualizer ⚡</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An interactive web application demonstrating Support Vector Machine kernel transformations in 2D and 3D.</div>', unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.markdown("### 🎛️ Dataset Generator Settings")
n_samples = st.sidebar.slider("Number of Samples", 50, 500, 120, step=10)
noise_level = st.sidebar.slider("Dataset Noise", 0.0, 0.5, 0.05, step=0.01)
inner_factor = st.sidebar.slider("Concentric Circle Factor", 0.1, 0.9, 0.35, step=0.05)
random_state = st.sidebar.number_input("Random Seed", value=7, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ SVM Classifier Hyperparameters")
kernel_type = st.sidebar.selectbox("SVM Kernel Type", ["rbf", "linear", "poly", "sigmoid"], index=0)
c_val = st.sidebar.slider("Regularization Parameter (C)", 0.01, 100.0, 10.0, step=0.1)

# Conditional hyperparameter displays
gamma_val = "scale"
degree_val = 3
coef0_val = 0.0

if kernel_type in ["rbf", "poly", "sigmoid"]:
    gamma_mode = st.sidebar.selectbox("Gamma Mode", ["scale", "auto", "custom"], index=2)
    if gamma_mode == "custom":
        gamma_val = st.sidebar.slider("Gamma (γ)", 0.01, 10.0, 3.0, step=0.05)
    else:
        gamma_val = gamma_mode

if kernel_type == "poly":
    degree_val = st.sidebar.slider("Polynomial Degree", 1, 5, 3, step=1)

if kernel_type in ["poly", "sigmoid"]:
    coef0_val = st.sidebar.slider("Independent Term (coef0)", 0.0, 10.0, 0.0, step=0.5)

# Generate Dataset
@st.cache_data
def load_data(n, noise, factor, seed):
    X, y = make_circles(n_samples=n, factor=factor, noise=noise, random_state=seed)
    return X, y

X, y = load_data(n_samples, noise_level, inner_factor, random_state)

# Fit SVM Model
clf = SVC(kernel=kernel_type, C=c_val, gamma=gamma_val, degree=degree_val, coef0=coef0_val)
clf.fit(X, y)

# Split classes for visualization
X_0 = X[y == 0]
X_1 = X[y == 1]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎮 2D Classification Boundary", 
    "🧬 3D Explicit Lifting", 
    "🎚️ 3D SVM Decision Surface", 
    "🎬 Manim Animation Player"
])

# Define grid for decision surface
x_grid = np.linspace(-1.8, 1.8, 100)
y_grid = np.linspace(-1.8, 1.8, 100)
xx, yy = np.meshgrid(x_grid, y_grid)
grid_points = np.c_[xx.ravel(), yy.ravel()]
zz = clf.decision_function(grid_points).reshape(xx.shape)

with tab1:
    st.markdown("### 2D Decision Boundary & Margin")
    st.write("Below is the interactive 2D representation of the SVM model. The contours show $f(x, y) = -1, 0, 1$. The solid contour line ($f(x,y)=0$) represents the decision boundary.")
    
    fig_2d = go.Figure()
    
    # 1. Background heatmap contour representing decision function values
    fig_2d.add_trace(go.Contour(
        x=x_grid,
        y=y_grid,
        z=zz,
        contours_coloring='heatmap',
        colorscale='RdBu',
        reversescale=True,
        zmin=-2,
        zmax=2,
        line_width=1,
        opacity=0.45,
        colorbar=dict(title="f(x, y) value"),
        contours=dict(
            start=-1,
            end=1,
            size=1,
            showlabels=True
        )
    ))
    
    # 2. Add class points
    fig_2d.add_trace(go.Scatter(
        x=X_0[:, 0], y=X_0[:, 1],
        mode='markers',
        marker=dict(color='#007BFF', size=8, line=dict(color='white', width=1)),
        name='Class 0 (Inner)'
    ))
    fig_2d.add_trace(go.Scatter(
        x=X_1[:, 0], y=X_1[:, 1],
        mode='markers',
        marker=dict(color='#FF4136', size=8, line=dict(color='white', width=1)),
        name='Class 1 (Outer)'
    ))
    
    # 3. Highlight support vectors
    if len(clf.support_vectors_) > 0:
        svs = clf.support_vectors_
        fig_2d.add_trace(go.Scatter(
            x=svs[:, 0], y=svs[:, 1],
            mode='markers',
            marker=dict(
                color='rgba(0,0,0,0)',
                size=14,
                line=dict(color='#FFDC00', width=2)
            ),
            name=f'Support Vectors ({len(svs)})'
        ))
        
    fig_2d.update_layout(
        xaxis=dict(range=[-1.8, 1.8], constrain='domain'),
        yaxis=dict(range=[-1.8, 1.8], scaleanchor="x", scaleratio=1),
        width=700,
        height=600,
        margin=dict(l=40, r=40, b=40, t=40)
    )
    
    st.plotly_chart(fig_2d, use_container_width=True)

with tab2:
    st.markdown("### 3D Explicit Feature Lifting ($z = x^2 + y^2$)")
    st.write("Points are transformed explicitly into 3D using $z = x^2 + y^2$. This mapping separates concentric circles into two height layers, making them linearly separable by a horizontal hyperplane.")
    
    # Slider to manually adjust the horizontal separating plane in 3D
    lift_threshold = st.slider("3D Hyperplane Height (z)", 0.0, 2.0, float(np.round((inner_factor**2 + 1.0)/2, 3)), step=0.01)
    
    fig_lift = go.Figure()
    
    # 1. Paraboloid Lifting Surface z = x^2 + y^2
    u_lift = np.linspace(-1.8, 1.8, 30)
    v_lift = np.linspace(-1.8, 1.8, 30)
    uu, vv = np.meshgrid(u_lift, v_lift)
    zz_paraboloid = uu**2 + vv**2
    
    fig_lift.add_trace(go.Surface(
        x=uu, y=vv, z=zz_paraboloid,
        opacity=0.25,
        colorscale='Greys',
        showscale=False,
        name='z = x² + y²'
    ))
    
    # 2. Horizontal separating hyperplane
    zz_plane = np.full_like(uu, lift_threshold)
    fig_lift.add_trace(go.Surface(
        x=uu, y=vv, z=zz_plane,
        opacity=0.35,
        colorscale=[[0, '#FFDC00'], [1, '#FFDC00']],
        showscale=False,
        name=f'Hyperplane z = {lift_threshold}'
    ))
    
    # 3. Lifted Points in 3D
    fig_lift.add_trace(go.Scatter3d(
        x=X_0[:, 0], y=X_0[:, 1], z=X_0[:, 0]**2 + X_0[:, 1]**2,
        mode='markers',
        marker=dict(color='#007BFF', size=4, line=dict(color='white', width=1)),
        name='Class 0 (Inner)'
    ))
    fig_lift.add_trace(go.Scatter3d(
        x=X_1[:, 0], y=X_1[:, 1], z=X_1[:, 0]**2 + X_1[:, 1]**2,
        mode='markers',
        marker=dict(color='#FF4136', size=4, line=dict(color='white', width=1)),
        name='Class 1 (Outer)'
    ))
    
    fig_lift.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=[-1.8, 1.8]),
            yaxis=dict(title='Y', range=[-1.8, 1.8]),
            zaxis=dict(title='Z', range=[0, 3.5]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.7)
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=650
    )
    
    st.plotly_chart(fig_lift, use_container_width=True)

with tab3:
    st.markdown("### RBF SVM Decision Surface Landscape ($z = f(x, y)$)")
    st.write("Plotting the actual decision function $z = f(x, y)$ as a 3D surface. The valley represents Class 0 (negative values) and the outer rim represents Class 1 (positive values). The yellow plane at $z=0$ shows the decision boundary.")
    
    fig_surface = go.Figure()
    
    # 1. SVM decision function surface
    # Clip decision values between -2.0 and 2.0 to maintain uniform visual boundaries
    zz_clipped = np.clip(zz, -2.0, 2.0)
    
    fig_surface.add_trace(go.Surface(
        x=x_grid, y=y_grid, z=zz_clipped,
        opacity=0.7,
        colorscale='RdBu',
        reversescale=True,
        colorbar=dict(title="f(x, y)"),
        name='Decision Surface'
    ))
    
    # 2. Decision boundary threshold plane (z = 0)
    zz_zero = np.zeros_like(xx)
    fig_surface.add_trace(go.Surface(
        x=x_grid, y=y_grid, z=zz_zero,
        opacity=0.3,
        colorscale=[[0, '#FFDC00'], [1, '#FFDC00']],
        showscale=False,
        name='z = 0 plane'
    ))
    
    # 3. Plot original training points on the z = 0 plane
    fig_surface.add_trace(go.Scatter3d(
        x=X_0[:, 0], y=X_0[:, 1], z=np.zeros(len(X_0)),
        mode='markers',
        marker=dict(color='#007BFF', size=3),
        name='Class 0 (Inner)'
    ))
    fig_surface.add_trace(go.Scatter3d(
        x=X_1[:, 0], y=X_1[:, 1], z=np.zeros(len(X_1)),
        mode='markers',
        marker=dict(color='#FF4136', size=3),
        name='Class 1 (Outer)'
    ))
    
    # 4. Highlight support vectors using yellow diamonds on the z = 0 plane
    if len(clf.support_) > 0:
        sv_coords = X[clf.support_]
        # Offset slightly in z to prevent z-fighting with the zero plane
        fig_surface.add_trace(go.Scatter3d(
            x=sv_coords[:, 0], y=sv_coords[:, 1], z=np.full(len(sv_coords), 0.02),
            mode='markers',
            marker=dict(color='#FFDC00', size=6, symbol='diamond'),
            name='Support Vectors'
        ))
        
    fig_surface.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=[-1.8, 1.8]),
            yaxis=dict(title='Y', range=[-1.8, 1.8]),
            zaxis=dict(title='f(x, y)', range=[-2.2, 2.2]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.7)
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=650
    )
    
    st.plotly_chart(fig_surface, use_container_width=True)

with tab4:
    st.markdown("### 🎬 Manim Video Animation Player")
    st.write("This tab serves the high-quality mathematical animation rendered by Manim Community Edition.")
    
    # Check for rendered videos in common manim output folders
    video_paths = [
        Path("media/videos/svm_kernel_demo/1080p60/SVMKernelFullDemo.mp4"),
        Path("media/videos/svm_kernel_demo/480p15/SVMKernelFullDemo.mp4")
    ]
    
    found_video = None
    for path in video_paths:
        if path.exists():
            found_video = path
            break
            
    if found_video:
        st.success(f"Found rendered video: `{found_video}`")
        video_bytes = found_video.read_bytes()
        st.video(video_bytes)
    else:
        st.info("No rendered Manim video found. You can render the animation locally by running:")
        st.code("manim -pql svm_kernel_demo.py SVMKernelFullDemo", language="bash")
        st.write("Once rendered, the video will automatically show up and play in this tab!")

# Footer details and explanations
st.markdown("---")
col_foo1, col_foo2 = st.columns(2)
with col_foo1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### ⚡ Quick SVM Summary")
    st.write(f"""
    - **Kernel Fitted**: `{kernel_type.upper()}`
    - **Regularization (C)**: `{c_val}`
    - **Total Support Vectors**: `{len(clf.support_)}` (out of {n_samples} samples)
    
    Support vectors are the crucial training points that lie closest to the decision boundary (or on the margin). Changing these coordinates modifies the decision boundary, while moving other points has no effect.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col_foo2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📖 How the Kernel Trick Works")
    st.write("""
    Concentric circles cannot be separated by any line in 2D. However:
    1. **Lifting**: We map points to 3D. Inner circle points map to lower heights, outer circle points map to higher heights. A flat 2D plane can now slice between them.
    2. **Kernel Trick**: Instead of explicitly calculating 3D positions, SVM uses a kernel function (like RBF) to implicitly measure similarities in a higher-dimensional space, fitting the decision function efficiently.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
