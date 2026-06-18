import os
import json
import numpy as np
from manim import *
from sklearn.svm import SVC
from sklearn.datasets import make_circles

# Load dynamic configuration from environment, or use defaults
config_str = os.environ.get("SVM_MANIM_CONFIG", None)
if config_str:
    user_cfg = json.loads(config_str)
else:
    user_cfg = {
        "n_samples": 120,
        "factor": 0.35,
        "noise": 0.05,
        "random_state": 7,
        "kernel": "rbf",
        "gamma": 3.0,
        "C": 10.0,
        "degree": 3,
        "coef0": 0.0
    }

# Define constants for colors and ranges
CLASS_0_COLOR = BLUE      # Inner circle
CLASS_1_COLOR = RED       # Outer circle
HYPERPLANE_COLOR = YELLOW
SV_HIGHLIGHT_COLOR = YELLOW

X_MIN, X_MAX = -1.6, 1.6
Y_MIN, Y_MAX = -1.6, 1.6

class SVMKernelFullDemo(ThreeDScene):
    def construct(self):
        # ----------------------------------------------------------------------
        # Phase 1: 2D Non-linearly Separable Data
        # ----------------------------------------------------------------------
        
        # Initialize title and description flat on screen
        title = Text("Phase 1: 2D Non-linearly Separable Data", font_size=24, color=WHITE)
        title.to_edge(UP)
        desc = Text("A straight line cannot separate center points from outer points.", font_size=16, color=LIGHT_GRAY)
        desc.next_to(title, DOWN)
        
        self.add_fixed_in_frame_mobjects(title, desc)
        
        # 2D orientation looking down positive z-axis
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)
        
        # Build axes (using ThreeDAxes so we can transition to 3D seamlessly)
        axes = ThreeDAxes(
            x_range=[-2.0, 2.0, 1.0],
            y_range=[-2.0, 2.0, 1.0],
            z_range=[-1.0, 3.0, 1.0],
            x_length=6.0,
            y_length=6.0,
            z_length=4.0,
        )
        x_label = axes.get_x_axis_label("x")
        y_label = axes.get_y_axis_label("y")
        z_label = axes.get_z_axis_label("z")
        
        # Generate Concentric Circle Data
        X, y = make_circles(
            n_samples=user_cfg["n_samples"], 
            factor=user_cfg["factor"], 
            noise=user_cfg["noise"], 
            random_state=user_cfg["random_state"]
        )
        
        # Generate 2D Dots
        dots = VGroup()
        for (x_val, y_val), label in zip(X, y):
            dot_color = CLASS_0_COLOR if label == 0 else CLASS_1_COLOR
            # Placed at z=0 plane initially
            dot = Dot3D(point=axes.c2p(x_val, y_val, 0), color=dot_color, radius=0.065)
            dots.add(dot)
            
        # Display Phase 1 Elements
        self.play(FadeIn(title), FadeIn(desc))
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(FadeIn(dots, lag_ratio=0.05))
        self.wait(1.5)
        
        # Show failed lines in 2D to emphasize non-linear separability
        line1 = DashedLine(axes.c2p(-1.8, 0, 0), axes.c2p(1.8, 0, 0), color=GRAY)
        line2 = DashedLine(axes.c2p(0, -1.8, 0), axes.c2p(0, 1.8, 0), color=GRAY)
        line3 = DashedLine(axes.c2p(-1.3, -1.3, 0), axes.c2p(1.3, 1.3, 0), color=GRAY)
        
        failed_text = Text("Attempting linear separation...", font_size=14, color=GRAY)
        failed_text.next_to(desc, DOWN)
        self.add_fixed_in_frame_mobjects(failed_text)
        
        self.play(FadeIn(failed_text))
        self.play(Create(line1), run_time=1.0)
        self.wait(0.5)
        self.play(ReplacementTransform(line1, line2), run_time=1.0)
        self.wait(0.5)
        self.play(ReplacementTransform(line2, line3), run_time=1.0)
        self.wait(1.0)
        self.play(FadeOut(line3), FadeOut(failed_text))
        self.remove_fixed_in_frame_mobjects(failed_text)
        self.wait(1.0)
        
        # Clean up Phase 1 annotations
        self.play(FadeOut(title), FadeOut(desc))
        self.remove_fixed_in_frame_mobjects(title, desc)
        
        # ----------------------------------------------------------------------
        # Phase 2: 3D Feature Lifting and Hyperplane
        # ----------------------------------------------------------------------
        
        # Setup Phase 2 Title and Text
        title = Text("Phase 2: Explicit 3D Feature Lifting", font_size=24, color=WHITE)
        title.to_edge(UP)
        desc = Text("Transform 2D points to 3D space using: z = x² + y²", font_size=16, color=LIGHT_GRAY)
        desc.next_to(title, DOWN)
        formula = MathTex("z = x^2 + y^2", color=YELLOW)
        formula.to_edge(DOWN)
        
        self.add_fixed_in_frame_mobjects(title, desc, formula)
        
        # Animate Camera transition to 3D Perspective
        self.move_camera(
            phi=65 * DEGREES, theta=-45 * DEGREES,
            added_anims=[
                Write(z_label),
                FadeIn(title),
                FadeIn(desc),
                Write(formula)
            ],
            run_time=2.0
        )
        self.wait(1.0)
        
        # Animate dot movement: lifting z coordinate to x^2 + y^2
        self.play(
            *[dot.animate.move_to(axes.c2p(x_val, y_val, x_val**2 + y_val**2))
              for dot, (x_val, y_val) in zip(dots, X)],
            run_time=3.0
        )
        self.wait(1.0)
        
        # Draw explicit lifting surface: paraboloid z = x^2 + y^2
        lifting_surface = Surface(
            lambda u, v: axes.c2p(u, v, u**2 + v**2),
            u_range=[X_MIN, X_MAX],
            v_range=[Y_MIN, Y_MAX],
            resolution=(20, 20),
            checkerboard_colors=[GRAY, LIGHT_GRAY]
        )
        lifting_surface.set_style(fill_opacity=0.25, stroke_color=GRAY, stroke_width=0.3)
        
        self.play(Create(lifting_surface), run_time=2.0)
        self.wait(1.0)
        
        # Draw decision hyperplane z = 0.55
        hyperplane_desc = Text("A horizontal hyperplane (z = 0.55) separates the classes in 3D.", font_size=16, color=LIGHT_GRAY)
        hyperplane_desc.next_to(title, DOWN)
        
        hyperplane = Surface(
            lambda u, v: axes.c2p(u, v, 0.55),
            u_range=[X_MIN, X_MAX],
            v_range=[Y_MIN, Y_MAX],
            resolution=(2, 2),
            checkerboard_colors=[HYPERPLANE_COLOR, HYPERPLANE_COLOR]
        )
        hyperplane.set_style(fill_opacity=0.35, stroke_color=HYPERPLANE_COLOR, stroke_width=1.0)
        
        hyperplane_label = MathTex("z = 0.55", color=HYPERPLANE_COLOR)
        hyperplane_label.next_to(formula, RIGHT, buff=1.5)
        
        self.add_fixed_in_frame_mobjects(hyperplane_label)
        self.play(
            Transform(desc, hyperplane_desc),
            Create(hyperplane),
            Write(hyperplane_label),
            run_time=2.0
        )
        
        # Rotate camera to show the 3D separation
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(4.0)
        self.stop_ambient_camera_rotation()
        self.wait(1.0)
        
        # Clean up Phase 2 elements
        self.play(
            FadeOut(title), FadeOut(desc), FadeOut(formula), FadeOut(hyperplane_label),
            FadeOut(lifting_surface), FadeOut(hyperplane)
        )
        self.remove_fixed_in_frame_mobjects(title, desc, formula, hyperplane_label)
        
        # ----------------------------------------------------------------------
        # Phase 3: Real RBF SVM Decision Surface
        # ----------------------------------------------------------------------
        
        # Setup Phase 3 Title and Text
        kernel_name = user_cfg["kernel"].upper()
        title = Text(f"Phase 3: Real {kernel_name} SVM Decision Surface", font_size=24, color=WHITE)
        title.to_edge(UP)
        desc = Text("Visualizing the SVM decision function f(x, y) as a 3D landscape.", font_size=16, color=LIGHT_GRAY)
        desc.next_to(title, DOWN)
        
        self.add_fixed_in_frame_mobjects(title, desc)
        
        # Animate dots moving back to z=0 plane
        self.play(
            *[dot.animate.move_to(axes.c2p(x_val, y_val, 0)) for dot, (x_val, y_val) in zip(dots, X)],
            FadeIn(title),
            FadeIn(desc),
            run_time=2.0
        )
        self.wait(1.0)
        
        # Train a real SVM using user configuration
        clf = SVC(
            kernel=user_cfg["kernel"], 
            gamma=user_cfg["gamma"], 
            C=user_cfg["C"],
            degree=user_cfg["degree"],
            coef0=user_cfg["coef0"]
        )
        clf.fit(X, y)
        
        # Build decision surface z = f(x, y)
        def get_decision_point(u, v):
            val = clf.decision_function(np.array([[u, v]]))[0]
            val_clipped = np.clip(val, -2.0, 2.0)
            return axes.c2p(u, v, val_clipped)
            
        decision_surface = Surface(
            get_decision_point,
            u_range=[X_MIN, X_MAX],
            v_range=[Y_MIN, Y_MAX],
            resolution=(30, 30),
            checkerboard_colors=[BLUE_D, RED_D]  # Checkerboard representing the binary classification values
        )
        decision_surface.set_style(fill_opacity=0.65, stroke_color=WHITE, stroke_width=0.3)
        
        # Build flat z = 0 decision threshold plane
        zero_plane = Surface(
            lambda u, v: axes.c2p(u, v, 0.0),
            u_range=[X_MIN, X_MAX],
            v_range=[Y_MIN, Y_MAX],
            resolution=(2, 2),
            checkerboard_colors=[YELLOW, YELLOW]
        )
        zero_plane.set_style(fill_opacity=0.2, stroke_color=YELLOW, stroke_width=0.0)
        
        self.play(Create(decision_surface), Create(zero_plane), run_time=3.0)
        self.wait(1.0)
        
        # Highlight support vectors on the z=0 plane with circles
        sv_rings = VGroup()
        for idx in clf.support_:
            x_val, y_val = X[idx]
            # Offset z slightly to prevent z-fighting with the zero plane
            ring = Circle(radius=0.1, color=SV_HIGHLIGHT_COLOR, stroke_width=3.0)
            ring.move_to(axes.c2p(x_val, y_val, 0.02))
            sv_rings.add(ring)
            
        sv_desc = Text("Support Vectors (yellow rings) define the SVM decision boundary.", font_size=16, color=LIGHT_GRAY)
        sv_desc.next_to(title, DOWN)
        
        self.play(
            Transform(desc, sv_desc),
            Create(sv_rings),
            run_time=2.0
        )
        self.wait(1.5)
        
        # Camera rotation and formula display
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(3.0)
        
        formula_label = MathTex("f(x, y) = \\sum_i \\alpha_i y_i K(x, x_i) + b", color=YELLOW)
        formula_label.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(formula_label)
        self.play(Write(formula_label), run_time=1.5)
        
        self.wait(5.0)
        self.stop_ambient_camera_rotation()
        self.wait(1.0)
        
        # Final concluding note
        final_desc = Text("The decision boundary is the contour f(x, y) = 0 (a circle).", font_size=16, color=LIGHT_GRAY)
        final_desc.next_to(title, DOWN)
        self.play(Transform(desc, final_desc))
        self.wait(3.0)
        
        # Fade out everything at the end
        self.play(
            FadeOut(title), FadeOut(desc), FadeOut(formula_label),
            FadeOut(axes), FadeOut(x_label), FadeOut(y_label), FadeOut(z_label),
            FadeOut(dots), FadeOut(decision_surface), FadeOut(zero_plane), FadeOut(sv_rings)
        )
        self.remove_fixed_in_frame_mobjects(title, desc, formula_label)
        self.wait(1.0)
