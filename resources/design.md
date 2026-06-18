project:
  name: "svm_kernel_trick_3d_manim_demo"
  goal: >
    Build a complete Manim animation project to demonstrate Support Vector Machine
    kernel trick using 2D non-linearly separable data, 3D feature lifting, and
    RBF SVM decision surface visualization.

  language: "Python"
  main_library:
    - "manim"
    - "numpy"
    - "scikit-learn"

  output_style:
    - "Educational animation"
    - "3D visualization"
    - "Suitable for classroom presentation"
    - "Clear labels, formulas, camera movement, and step-by-step explanation"

  expected_files:
    - "svm_kernel_demo.py"
    - "requirements.txt"
    - "README.md"

  install_command: "pip install -r requirements.txt"

  render_commands:
    preview_quality:
      command: "manim -pql svm_kernel_demo.py SVMKernelFullDemo"
    high_quality:
      command: "manim -pqh svm_kernel_demo.py SVMKernelFullDemo"

  requirements_txt:
    packages:
      - "manim"
      - "numpy"
      - "scikit-learn"

concept:
  title: "Visualizing SVM Kernel Trick in 3D"
  core_idea: >
    In the original 2D input space, the blue points in the center and red points
    outside cannot be separated by a straight line. After applying a feature
    transformation or kernel trick, the data can be viewed in a higher-dimensional
    space where a hyperplane can separate the two classes.

  important_warning: >
    The explicit 3D transformation z = x^2 + y^2 is a teaching visualization.
    The real RBF kernel maps data into a high-dimensional or infinite-dimensional
    feature space, not simply into 3D. Therefore, the project should clearly
    distinguish between the educational 3D lifting demo and the real RBF SVM
    decision function visualization.

  formulas:
    explicit_lifting:
      latex: "\\phi(x, y) = (x, y, x^2 + y^2)"
      explanation: >
        Points farther from the origin get larger z values. This turns concentric
        circle data into two height levels that can be separated by a horizontal
        plane.

    rbf_kernel:
      latex: "K(x, x_i) = \\exp(-\\gamma \\|x - x_i\\|^2)"
      explanation: >
        RBF kernel measures similarity between points. Gamma controls how local
        the influence of each training point is.

    svm_decision_function:
      latex: "f(x) = \\sum_i \\alpha_i y_i K(x, x_i) + b"
      explanation: >
        The decision boundary is where f(x) = 0. In the visualization, f(x, y)
        can be plotted as a 3D surface.

data:
  generator:
    function: "sklearn.datasets.make_circles"
    parameters:
      n_samples: 120
      factor: 0.35
      noise: 0.05
      random_state: 7

  class_design:
    class_0:
      color: "BLUE"
      meaning: "inner circle / center points"
    class_1:
      color: "RED"
      meaning: "outer circle / surrounding points"

  visual_range:
    x_range: [-1.6, 1.6]
    y_range: [-1.6, 1.6]

implementation:
  main_scene:
    class_name: "SVMKernelFullDemo"
    inherits_from: "ThreeDScene"
    description: >
      A full Manim animation containing three phases:
      1. 2D non-linear separability
      2. 3D feature lifting and hyperplane
      3. RBF SVM decision surface

  helper_functions:
    - name: "generate_data"
      purpose: "Generate concentric circle data using make_circles."

    - name: "lift_z"
      purpose: "Return z = x^2 + y^2 for explicit 3D lifting."

    - name: "train_rbf_svm"
      purpose: "Train sklearn.svm.SVC with RBF kernel."

    - name: "decision_value"
      purpose: "Return clipped SVM decision_function value for a given x, y."

    - name: "make_2d_dots"
      purpose: "Create 2D or z=0 class-colored dots."

    - name: "make_3d_lifted_dots"
      purpose: "Create Dot3D objects using lifted z coordinates."

    - name: "make_lifting_surface"
      purpose: "Create surface z = x^2 + y^2."

    - name: "make_decision_surface"
      purpose: "Create surface z = SVM decision_function(x, y)."

phase_1:
  name: "2D Non-linearly Separable Data"
  objective: >
    Show that the original 2D data cannot be separated by a single straight line.

  scene_behavior:
    - "Display 2D axes with x and y labels."
    - "Plot blue points in the center and red points in the outer ring."
    - "Show several attempted straight lines that fail to separate the data."
    - "Add text: 'Original 2D space: not linearly separable'."
    - "Use simple camera view from top or orthogonal 2D-like view."

  visual_elements:
    axes:
      type: "Axes or ThreeDAxes viewed from top"
      labels: ["x", "y"]

    points:
      blue_points: "inner circle"
      red_points: "outer circle"

    failed_lines:
      count: 3
      style: "thin gray dashed lines"
      purpose: "Demonstrate that no single straight line works."

    text:
      title: "Phase 1: 2D data is not linearly separable"
      explanation: "A straight line cannot separate center points from outer points."

  acceptance_criteria:
    - "The viewer can clearly see blue points inside and red points outside."
    - "At least 3 failed linear separators are shown."
    - "The scene clearly explains why a linear model in 2D fails."

phase_2:
  name: "3D Feature Lifting and Hyperplane"
  objective: >
    Demonstrate the kernel trick intuition by lifting the 2D points into 3D using
    z = x^2 + y^2, then separating them with a hyperplane.

  transformation:
    formula: "z = x^2 + y^2"
    interpretation: >
      Points near the origin stay low. Points far from the origin move higher.
      This makes inner and outer circles separable by a horizontal plane.

  scene_behavior:
    - "Transition from 2D axes to 3D axes."
    - "Animate each point moving upward according to z = x^2 + y^2."
    - "Draw the surface z = x^2 + y^2 with transparent opacity."
    - "Add a horizontal decision hyperplane z = c."
    - "Use camera rotation to show the separation from multiple angles."
    - "Add formula label: z = x^2 + y^2."
    - "Add hyperplane label: z = c."

  visual_elements:
    axes:
      type: "ThreeDAxes"
      labels: ["x", "y", "z"]

    lifted_points:
      blue_points: "low z values"
      red_points: "higher z values"

    lifting_surface:
      equation: "z = x^2 + y^2"
      opacity: 0.25
      color_style: "gray transparent surface"

    hyperplane:
      equation: "z = c"
      suggested_c_value: 0.55
      opacity: 0.35
      color: "YELLOW"

    camera:
      initial_phi: "65 degrees"
      initial_theta: "-45 degrees"
      rotation: true
      rotation_rate: 0.25

  acceptance_criteria:
    - "The animation clearly shows points moving from 2D to 3D."
    - "The blue points should appear below the hyperplane."
    - "The red points should appear above the hyperplane."
    - "The camera should rotate so the viewer can understand the 3D separation."
    - "The scene should explicitly say this is a teaching-friendly explicit lifting."

phase_3:
  name: "Real RBF SVM Decision Surface"
  objective: >
    Train a real RBF SVM model and visualize its decision function as a 3D surface.

  model:
    library: "sklearn.svm.SVC"
    parameters:
      kernel: "rbf"
      gamma: 3.0
      C: 10
    train_on: "same make_circles dataset"

  decision_surface:
    definition: "z = clf.decision_function([[x, y]])[0]"
    clipping_range: [-2, 2]
    resolution: [35, 35]

  scene_behavior:
    - "Train the RBF SVM model."
    - "Display 3D axes with z-axis labeled f(x, y)."
    - "Plot the SVM decision function as a 3D surface."
    - "Display z = 0 plane as the classification threshold."
    - "Plot original data points on the z = 0 plane."
    - "Highlight support vectors using yellow larger dots or rings."
    - "Add explanation: 'Decision boundary is where f(x, y) = 0'."
    - "Rotate camera to view the surface from multiple directions."

  visual_elements:
    axes:
      type: "ThreeDAxes"
      labels: ["x", "y", "f(x, y)"]

    decision_surface:
      opacity: 0.65
      color_style: "blue/red or gradient-like checkerboard"
      note: "Manim Surface can use checkerboard colors if gradient is difficult."

    zero_plane:
      equation: "f(x, y) = 0"
      opacity: 0.25
      color: "YELLOW"

    data_points:
      placement: "z = 0 plane"
      blue_points: "class 0"
      red_points: "class 1"

    support_vectors:
      source: "clf.support_"
      style:
        color: "YELLOW"
        radius: 0.065
      meaning: "Important training points that define the SVM boundary."

  acceptance_criteria:
    - "The RBF SVM model is actually trained with sklearn."
    - "The decision function surface is visible in 3D."
    - "The z = 0 plane is clearly shown."
    - "Support vectors are highlighted."
    - "The scene explains that the true RBF feature space is not simply 3D."

code_quality:
  requirements:
    - "Use clean class and helper function structure."
    - "Avoid putting all logic directly inside construct if possible."
    - "Use constants for colors, ranges, gamma, C, and camera settings."
    - "Include comments explaining each phase."
    - "Make the code runnable directly with Manim."
    - "Do not require external datasets."
    - "Use deterministic random_state for reproducibility."

  manim_guidelines:
    - "Use ThreeDScene for the full demo."
    - "Use ThreeDAxes for 3D visualization."
    - "Use Dot3D for 3D points."
    - "Use Surface for lifting surface, hyperplane, and decision surface."
    - "Use MathTex for formulas."
    - "Use Text for short explanations."
    - "Use FadeIn, FadeOut, Transform, Create, ReplacementTransform, and LaggedStart."
    - "Use begin_ambient_camera_rotation and stop_ambient_camera_rotation."
    - "Keep animations readable, not too fast."

readme:
  must_include:
    - "Project title"
    - "Concept explanation"
    - "Installation steps"
    - "Render commands"
    - "Explanation of each phase"
    - "Difference between explicit 3D lifting and real RBF kernel"
    - "Troubleshooting notes for Manim installation"

final_deliverable:
  description: >
    A complete Manim project that renders one educational animation showing:
    Phase 1: non-linear separability in 2D,
    Phase 2: explicit 3D lifting with hyperplane,
    Phase 3: real RBF SVM decision function surface.

  success_definition: >
    The animation should be understandable by students learning SVM for the first
    time. It should clearly show why kernel methods are useful and how a
    non-linear boundary in 2D can correspond to a linear hyperplane in a transformed
    feature space.