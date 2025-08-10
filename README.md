# BRDF Importance Sampling Visualizer

An interactive Python application for exploring and comparing different sampling strategies in Bidirectional Reflectance Distribution Function (BRDF) evaluation. This project demonstrates the effectiveness of importance sampling over uniform sampling for Monte Carlo integration in computer graphics rendering.

**Team Z_Buffer**: 2005076, 2005106, 2005110  
**Course**: CSE 409 - Computer Graphics  
**Institution**: BUET

## 🎯 Project Overview

This project implements and visualizes various BRDF models (Phong, Blinn-Phong, Cook-Torrance) with different sampling strategies to demonstrate the power of importance sampling in reducing variance and improving convergence in Monte Carlo integration.

### Key Features

- **Multiple BRDF Models**: Phong, Blinn-Phong, and Cook-Torrance BRDF implementations
- **Interactive Visualizations**: Real-time 3D and 2D plots with adjustable parameters
- **Sampling Comparison**: Side-by-side comparison of uniform vs. importance sampling
- **Convergence Analysis**: Detailed analysis of how different sampling methods converge
- **Professional UI**: Clean, intuitive interface with sliders and controls

## 📋 Requirements

### Dependencies
- Python 3.7+
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0

### System Requirements
- Windows, macOS, or Linux
- 4GB RAM (recommended)
- Display capable of 1920x1080 resolution

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Importance-Sampling-in-BRDF-Graphics--main
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv brdf_env
   ```

3. **Activate the virtual environment**:
   - **Windows**:
     ```bash
     brdf_env\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source brdf_env/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

### Quick Start

Run the main application:
```bash
python code/brdf_interactive.py
```

The application will start with a numerical comparison and then present an interactive menu with visualization options.

### Interactive Menu Options

1. **🎯 Interactive BRDF Explorer** (Recommended)
   - Real-time parameter adjustment with sliders
   - Switch between BRDF models
   - Observe immediate visual changes

2. **📊 Clean Static Comparison**
   - Side-by-side comparison of sampling methods
   - Static plots for documentation

3. **📈 Convergence Analysis**
   - Detailed convergence study across different sample sizes
   - Variance reduction analysis

4. **🔄 Interactive 3D Explorer**
   - Real-time 3D BRDF exploration
   - Multiple visualization views
   - Live parameter adjustments

5. **📊 BRDF Model Comparison**
   - Compare Phong, Blinn-Phong, and Cook-Torrance models
   - 3D and 1D visualizations

## 🔬 Technical Details

### BRDF Models Implemented

#### Phong BRDF
```python
f_r = max(R · V, 0)^n
```
- Simple and computationally efficient
- Good for glossy surfaces
- Adjustable specular power (n)
- Uses reflection direction R

#### Blinn-Phong BRDF
```python
f_r = max(N · H, 0)^n
```
- More physically plausible than Phong
- Uses half-angle vector H = normalize(L + V)
- Better for specular highlights
- More stable than Phong

#### Cook-Torrance BRDF
```python
f_r ∝ D(θ) · F(θ) · G(θ)
```
- Physically-based rendering model
- Includes distribution (D), Fresnel (F), and geometry (G) terms
- More realistic for rough surfaces
- Industry standard for PBR

### Sampling Strategies

#### Uniform Sampling
- Random samples distributed across the hemisphere
- PDF: `p(ω) = 1/(2π)`
- Simple implementation but inefficient
- High variance, slow convergence

#### Importance Sampling
- Samples concentrated where the BRDF is large
- **Phong/Blinn-Phong**: `θ = arccos(u₁^(1/(n+1)))`, `φ = 2πu₂`
- **Cook-Torrance**: `p(θ) ∝ D(θ)cos(θ)`
- Dramatically reduces variance
- Faster convergence

### Mathematical Foundation

The project demonstrates Monte Carlo integration of the rendering equation:

```
L_o(v̂) = ∫ f_r(l̂, v̂) L_i(l̂) (n̂ · l̂) dl̂
```

Where:
- `L_o`: Outgoing radiance
- `f_r`: BRDF function
- `L_i`: Incident radiance
- `n̂ · l̂`: Cosine term
- `l̂, v̂, n̂`: Light, view, and normal directions

## 📊 Results and Performance

### Typical Results (Phong BRDF, n=32, 1000 samples)

| Method | Estimate | Standard Deviation | Variance Reduction |
|--------|----------|-------------------|-------------------|
| Analytical | 0.184698 | - | - |
| Uniform | 0.161883 | ±0.023 | 1.0× |
| Importance | 0.184698 | ±0.008 | 8.3× |

### Key Findings

- **Correctness**: Both methods converge to the same analytical solution
- **Efficiency**: Importance sampling shows 8.3× variance reduction
- **Convergence**: Importance sampling requires fewer samples for the same accuracy
- **Scalability**: Performance improvement increases with higher specular powers

## 🎨 Visualization Features

### Interactive Controls
- **Specular Power Slider**: Adjust the sharpness of reflections (1-100)
- **Roughness Slider**: Control surface roughness for Cook-Torrance model
- **BRDF Model Selector**: Switch between different BRDF implementations
- **Sample Count**: Adjust the number of Monte Carlo samples

### Plot Types
- **3D BRDF Lobes**: Interactive 3D visualization of reflection patterns
- **2D Cross-sections**: Detailed 1D plots showing BRDF behavior
- **Sample Distributions**: Visual comparison of sampling strategies
- **Convergence Plots**: Analysis of estimation accuracy vs. sample count

## 📚 Educational Value

This project serves as an excellent learning tool for:

- **Computer Graphics Students**: Understanding BRDF theory and implementation
- **Monte Carlo Methods**: Learning importance sampling techniques
- **Scientific Visualization**: Creating interactive plots and animations
- **Python Programming**: Advanced matplotlib and numpy usage

## 🔧 Customization

### Adding New BRDF Models

To add a new BRDF model:

1. Implement the BRDF function following the existing pattern
2. Add sampling functions if needed
3. Update the visualization functions to include the new model
4. Add UI controls for any new parameters

### Modifying Parameters

Key configuration constants in `code/brdf_interactive.py`:
```python
N_SAMPLES = 1000              # Default number of samples
DEFAULT_SPECULAR_POWER = 32   # Default Phong exponent
L_i = 1.0                     # Incident light intensity
```

## 📁 Project Structure

```
Importance-Sampling-in-BRDF-Graphics--main/
├── code/
│   ├── brdf_interactive.py      # Main application
│   └── requirements.txt         # Python dependencies
├── slide/
│   ├── Z_Buffer_76_106_110.tex  # LaTeX presentation source
│   ├── Z_Buffer_76_106_110.pdf  # Compiled presentation
│   ├── brdf_cone_diagram.png    # BRDF visualization
│   └── brdf_lobes_clean.png     # BRDF lobes comparison
├── resources/
│   └── 409 Assignment.pdf       # Assignment document
└── README.md                    # This file
```

## 📖 Presentation Content

The project includes a comprehensive LaTeX presentation (`slide/Z_Buffer_76_106_110.pdf`) covering:

### Key Topics Covered:
1. **Rendering Challenge**: Why sampling is necessary for the rendering equation
2. **BRDF Basics**: Three specular models (Phong, Blinn-Phong, Cook-Torrance)
3. **Monte Carlo Integration**: Estimating integrals with random sampling
4. **Importance Sampling PDFs**: Mathematical foundation for each model
5. **Variable Definitions**: Complete guide to mathematical notation
6. **Implementation Hints**: Practical coding considerations
7. **Model Selection**: When to use which BRDF model

### Mathematical Formulas:
- **Phong**: `f_r = max(r̂ · v̂, 0)^n` with PDF `p(θ_R) = (n+1)/(2π)cos^n(θ_R)`
- **Blinn-Phong**: `f_r = max(n̂ · ĥ, 0)^n` with PDF `p(θ_H) = (n+1)/(2π)cos^n(θ_H)`
- **Cook-Torrance**: `f_r ∝ D(θ)F(θ)G(θ)` with PDF `p(θ) ∝ D(θ)cos(θ)`

## 🤝 Contributing

This is an educational project, but suggestions and improvements are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 🙏 Acknowledgments

- **CSE 409 Course Staff**: For providing the assignment framework
- **Computer Graphics Community**: For the mathematical foundations
- **Matplotlib and NumPy Teams**: For the excellent visualization and numerical libraries

---

**Team Z_Buffer** - *Happy Rendering! 🎨✨* 