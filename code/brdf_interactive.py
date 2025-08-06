import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.gridspec as gridspec
import time

# Set matplotlib style for better presentation
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16
})

# === CONFIGURATION ===
N_SAMPLES = 1000
DEFAULT_SPECULAR_POWER = 32
L_i = 1.0

# === FIXED VECTORS ===
NORMAL = np.array([0, 0, 1])
VIEW_DIR = np.array([0, 0, 1])

# === VECTOR HELPERS ===
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def reflect(v, n):
    return v - 2 * np.dot(v, n) * n

def spherical_to_cartesian(theta, phi):
    """Convert spherical coordinates to Cartesian"""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

# === MULTIPLE BRDF MODELS ===
def phong_brdf(wi, wo, n=NORMAL, power=DEFAULT_SPECULAR_POWER):
    """Phong BRDF model"""
    r = reflect(-wi, n)
    dot = max(np.dot(r, wo), 0.0)
    return dot ** power

def blinn_phong_brdf(wi, wo, n=NORMAL, power=DEFAULT_SPECULAR_POWER):
    """Blinn-Phong BRDF model"""
    h = normalize(wi + wo)
    dot = max(np.dot(n, h), 0.0)
    return dot ** power

def cook_torrance_brdf(wi, wo, n=NORMAL, roughness=0.1, fresnel_0=0.04):
    """Simplified Cook-Torrance BRDF"""
    h = normalize(wi + wo)
    nh = max(np.dot(n, h), 0.0)
    nv = max(np.dot(n, wo), 0.0)
    nl = max(np.dot(n, wi), 0.0)
    vh = max(np.dot(wo, h), 0.0)
    
    # Fresnel (Schlick approximation)
    F = fresnel_0 + (1 - fresnel_0) * (1 - vh) ** 5
    
    # Distribution (GGX/Trowbridge-Reitz)
    alpha = roughness ** 2
    denom = nh ** 2 * (alpha ** 2 - 1) + 1
    D = alpha ** 2 / (np.pi * denom ** 2)
    
    # Geometry (Smith)
    k = (roughness + 1) ** 2 / 8
    g1_v = nv / (nv * (1 - k) + k)
    g1_l = nl / (nl * (1 - k) + k)
    G = g1_v * g1_l
    
    return D * F * G / max(4 * nv * nl, 1e-6)

# === ENHANCED SAMPLING FUNCTIONS ===
def uniform_sample_hemisphere():
    u1, u2 = np.random.rand(2)
    z = u1
    r = np.sqrt(1 - z ** 2)
    phi = 2 * np.pi * u2
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.array([x, y, z])

def cosine_weighted_sample_hemisphere():
    """Sample hemisphere with cosine weighting"""
    u1, u2 = np.random.rand(2)
    theta = np.arccos(np.sqrt(u1))
    phi = 2 * np.pi * u2
    return spherical_to_cartesian(theta, phi)

def phong_pdf(wi, r, power=DEFAULT_SPECULAR_POWER):
    cos_alpha = max(np.dot(normalize(wi), normalize(r)), 0.0)
    return (power + 1) / (2 * np.pi) * (cos_alpha ** power)

def importance_sample_phong_lobe(r, power=DEFAULT_SPECULAR_POWER):
    u1, u2 = np.random.rand(2)
    theta = np.arccos(u1 ** (1 / (power + 1)))
    phi = 2 * np.pi * u2
    sin_theta = np.sin(theta)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = np.cos(theta)

    up = np.array([0, 1, 0]) if abs(r[2]) > 0.999 else np.array([0, 0, 1])
    tangent = normalize(np.cross(up, r))
    bitangent = np.cross(r, tangent)
    direction = x * tangent + y * bitangent + z * r
    return normalize(direction)

# === ANALYTICAL SOLUTIONS ===
def analytical_phong_integral(power=DEFAULT_SPECULAR_POWER):
    """Analytical solution for Phong BRDF integral"""
    return 2 * np.pi / (power + 2)

# === ENHANCED MONTE CARLO INTEGRATORS ===
class SamplingResult:
    def __init__(self, estimate, samples, variance, time_taken, method_name):
        self.estimate = estimate
        self.samples = samples
        self.variance = variance
        self.time_taken = time_taken
        self.method_name = method_name
        self.efficiency = 1.0 / (variance * time_taken) if variance > 0 else np.inf

def estimate_radiance_method(sampling_method, brdf_func, n_samples=N_SAMPLES):
    """Generic radiance estimation with timing and variance calculation"""
    start_time = time.time()
    contributions = []
    samples = []
    
    for _ in range(n_samples):
        if sampling_method == 'uniform':
            wi = uniform_sample_hemisphere()
            pdf = 1 / (2 * np.pi)
        elif sampling_method == 'cosine':
            wi = cosine_weighted_sample_hemisphere()
            pdf = max(np.dot(NORMAL, wi), 0.0) / np.pi
        elif sampling_method == 'importance_phong':
            r = reflect(-VIEW_DIR, NORMAL)
            wi = importance_sample_phong_lobe(r, DEFAULT_SPECULAR_POWER)
            pdf = phong_pdf(wi, r, DEFAULT_SPECULAR_POWER)
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")
            
        fr = brdf_func(wi, VIEW_DIR, NORMAL)
        cos_theta = max(np.dot(NORMAL, wi), 0.0)
        contribution = (fr * L_i * cos_theta) / max(pdf, 1e-8)
        
        contributions.append(contribution)
        samples.append((wi, contribution))
    
    time_taken = time.time() - start_time
    estimate = np.mean(contributions)
    variance = np.var(contributions)
    
    return SamplingResult(estimate, samples, variance, time_taken, sampling_method)

# === 3D BRDF LOBE VISUALIZATION ===
def plot_interactive_brdf_lobe(brdf_type='phong', power=32, roughness=0.1):
    """Plot the BRDF lobe in 3D with multiple views"""
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(f'3D BRDF Lobe Visualization: {brdf_type.replace("_", "-").title()}', 
                 fontsize=16, fontweight='bold')
    
    # Select BRDF function based on type
    if brdf_type == 'phong':
        brdf_func = lambda wi, wo, n: phong_brdf(wi, wo, n, power)
        title_suffix = f'(n={power})'
    elif brdf_type == 'blinn_phong':
        brdf_func = lambda wi, wo, n: blinn_phong_brdf(wi, wo, n, power)
        title_suffix = f'(n={power})'
    elif brdf_type == 'cook_torrance':
        brdf_func = lambda wi, wo, n: cook_torrance_brdf(wi, wo, n, roughness)
        title_suffix = f'(α={roughness:.2f})'
    else:
        brdf_func = lambda wi, wo, n: phong_brdf(wi, wo, n, power)
        title_suffix = f'(n={power})'
    
    # Generate points on hemisphere
    theta = np.linspace(0, np.pi/2, 40)  # Only upper hemisphere
    phi = np.linspace(0, 2*np.pi, 80)
    theta, phi = np.meshgrid(theta, phi)
    
    # Convert to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi) 
    z = np.cos(theta)
    
    # Calculate BRDF values
    brdf_values = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            wi = np.array([x[i,j], y[i,j], z[i,j]])
            brdf_values[i,j] = brdf_func(wi, VIEW_DIR, NORMAL)
    
    # Normalize BRDF values for better visualization
    brdf_max = brdf_values.max()
    if brdf_max > 0:
        brdf_normalized = brdf_values / brdf_max
    else:
        brdf_normalized = brdf_values
    
    # Plot 1: 3D BRDF lobe (Interactive)
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Create surface plot with BRDF values as colors
    surf = ax1.plot_surface(x, y, z, 
                           facecolors=plt.cm.plasma(brdf_normalized), 
                           alpha=0.8, linewidth=0, antialiased=True)
    
    # Add some styling
    ax1.set_title(f'{brdf_type.replace("_", "-").title()} BRDF Lobe {title_suffix}', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('X Direction', fontsize=10)
    ax1.set_ylabel('Y Direction', fontsize=10)
    ax1.set_zlabel('Z Direction', fontsize=10)
    
    # Set equal aspect ratio and clean appearance
    ax1.set_box_aspect([1,1,0.8])
    ax1.grid(True, alpha=0.3)
    
    # Add a color bar for BRDF values
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
    mappable.set_array(brdf_values)
    cbar1 = plt.colorbar(mappable, ax=ax1, shrink=0.6, aspect=20)
    cbar1.set_label('BRDF Value', fontsize=10)
    
    # Plot 2: 2D polar plot
    ax2 = fig.add_subplot(132, projection='polar')
    
    # Create polar contour plot
    contour = ax2.contourf(phi, theta, brdf_values, levels=20, cmap='plasma')
    ax2.set_title('BRDF Polar View', fontsize=12, fontweight='bold', pad=20)
    ax2.set_theta_zero_location('E')  # 0 degrees at right
    ax2.set_ylim(0, np.pi/2)
    
    # Add color bar
    cbar2 = plt.colorbar(contour, ax=ax2, shrink=0.6, aspect=20)
    cbar2.set_label('BRDF Value', fontsize=10)
    
    # Plot 3: 1D slice through BRDF
    ax3 = fig.add_subplot(133)
    
    # Calculate 1D slice at phi=0 (through the main lobe)
    angles = np.linspace(0, np.pi/2, 100)
    brdf_slice = []
    for angle in angles:
        wi = np.array([np.sin(angle), 0, np.cos(angle)])
        brdf_slice.append(brdf_func(wi, VIEW_DIR, NORMAL))
    
    ax3.plot(np.degrees(angles), brdf_slice, 'navy', linewidth=3, label=brdf_type.replace('_', '-').title())
    ax3.set_xlabel('Angle from Normal (degrees)', fontsize=10)
    ax3.set_ylabel('BRDF Value', fontsize=10)
    ax3.set_title(f'BRDF Cross-Section {title_suffix}', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Add some statistical info
    mean_brdf = np.mean(brdf_slice)
    max_brdf = np.max(brdf_slice)
    ax3.text(0.05, 0.85, f'Max: {max_brdf:.3f}\nMean: {mean_brdf:.3f}', 
             transform=ax3.transAxes, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    plt.show()

def create_brdf_model_comparison():
    """Compare all three BRDF models in 3D"""
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('BRDF Model Comparison: 3D Visualization', fontsize=18, fontweight='bold')
    
    models = [
        ('phong', 'Phong BRDF', 32),
        ('blinn_phong', 'Blinn-Phong BRDF', 32), 
        ('cook_torrance', 'Cook-Torrance BRDF', 0.1)
    ]
    
    # Generate hemisphere points
    theta = np.linspace(0, np.pi/2, 30)
    phi = np.linspace(0, 2*np.pi, 60)
    theta, phi = np.meshgrid(theta, phi)
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    for idx, (model_type, model_name, param) in enumerate(models):
        # Calculate BRDF values for this model
        brdf_values = np.zeros_like(x)
        
        if model_type == 'phong':
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    wi = np.array([x[i,j], y[i,j], z[i,j]])
                    brdf_values[i,j] = phong_brdf(wi, VIEW_DIR, NORMAL, param)
        elif model_type == 'blinn_phong':
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    wi = np.array([x[i,j], y[i,j], z[i,j]])
                    brdf_values[i,j] = blinn_phong_brdf(wi, VIEW_DIR, NORMAL, param)
        elif model_type == 'cook_torrance':
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    wi = np.array([x[i,j], y[i,j], z[i,j]])
                    brdf_values[i,j] = cook_torrance_brdf(wi, VIEW_DIR, NORMAL, param)
        
        # Normalize for visualization
        brdf_max = brdf_values.max()
        if brdf_max > 0:
            brdf_normalized = brdf_values / brdf_max
        else:
            brdf_normalized = brdf_values
        
        # 3D plot
        ax_3d = fig.add_subplot(2, 3, idx+1, projection='3d')
        surf = ax_3d.plot_surface(x, y, z, 
                                 facecolors=plt.cm.viridis(brdf_normalized),
                                 alpha=0.8, linewidth=0, antialiased=True)
        
        param_str = f'n={param}' if model_type in ['phong', 'blinn_phong'] else f'α={param}'
        ax_3d.set_title(f'{model_name}\n({param_str})', fontsize=12, fontweight='bold')
        ax_3d.set_box_aspect([1,1,0.8])
        ax_3d.grid(True, alpha=0.3)
        
        # 1D comparison plot
        ax_1d = fig.add_subplot(2, 3, idx+4)
        angles = np.linspace(0, np.pi/2, 100)
        brdf_slice = []
        
        for angle in angles:
            wi = np.array([np.sin(angle), 0, np.cos(angle)])
            if model_type == 'phong':
                brdf_slice.append(phong_brdf(wi, VIEW_DIR, NORMAL, param))
            elif model_type == 'blinn_phong':
                brdf_slice.append(blinn_phong_brdf(wi, VIEW_DIR, NORMAL, param))
            elif model_type == 'cook_torrance':
                brdf_slice.append(cook_torrance_brdf(wi, VIEW_DIR, NORMAL, param))
        
        colors = ['#e74c3c', '#f39c12', '#27ae60']
        ax_1d.plot(np.degrees(angles), brdf_slice, color=colors[idx], linewidth=3, 
                  label=model_name)
        ax_1d.set_xlabel('Angle from Normal (degrees)')
        ax_1d.set_ylabel('BRDF Value')
        ax_1d.set_title(f'{model_name} Profile')
        ax_1d.grid(True, alpha=0.3)
        ax_1d.legend()
    
    plt.tight_layout()
    
    # Add comparison note
    fig.text(0.02, 0.02, 
             'Note: Each model shows different specular characteristics.\n' +
             'Phong: Classic reflection model | Blinn-Phong: More physically plausible | Cook-Torrance: Physically-based', 
             fontsize=11, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.show()

def create_interactive_3d_explorer():
    """Create an interactive 3D BRDF explorer with real-time updates"""
    from matplotlib.widgets import Slider, RadioButtons
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Interactive 3D BRDF Explorer', fontsize=16, fontweight='bold')
    
    # Create subplots with space for controls
    ax_3d = fig.add_subplot(221, projection='3d')
    ax_polar = fig.add_subplot(222, projection='polar') 
    ax_1d = fig.add_subplot(223)
    ax_samples = fig.add_subplot(224)
    
    # Control area
    plt.subplots_adjust(bottom=0.25)
    
    # Initial parameters
    current_params = {'model': 'phong', 'power': 32, 'roughness': 0.1}
    
    def update_3d_plots():
        # Clear all axes
        ax_3d.clear()
        ax_polar.clear()
        ax_1d.clear()
        ax_samples.clear()
        
        model = current_params['model']
        power = current_params['power']
        roughness = current_params['roughness']
        
        # Select BRDF function
        if model == 'phong':
            brdf_func = lambda wi, wo, n: phong_brdf(wi, wo, n, power)
            title_suffix = f'Phong (n={power})'
        elif model == 'blinn_phong':
            brdf_func = lambda wi, wo, n: blinn_phong_brdf(wi, wo, n, power) 
            title_suffix = f'Blinn-Phong (n={power})'
        elif model == 'cook_torrance':
            brdf_func = lambda wi, wo, n: cook_torrance_brdf(wi, wo, n, roughness)
            title_suffix = f'Cook-Torrance (α={roughness:.2f})'
        
        # Generate hemisphere points
        theta = np.linspace(0, np.pi/2, 25)
        phi = np.linspace(0, 2*np.pi, 50)
        theta, phi = np.meshgrid(theta, phi)
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        # Calculate BRDF values
        brdf_values = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                wi = np.array([x[i,j], y[i,j], z[i,j]])
                brdf_values[i,j] = brdf_func(wi, VIEW_DIR, NORMAL)
        
        # Normalize
        brdf_max = brdf_values.max()
        if brdf_max > 0:
            brdf_normalized = brdf_values / brdf_max
        else:
            brdf_normalized = brdf_values
        
        # 3D plot
        surf = ax_3d.plot_surface(x, y, z, 
                                 facecolors=plt.cm.plasma(brdf_normalized),
                                 alpha=0.9, linewidth=0, antialiased=True)
        ax_3d.set_title(f'3D BRDF Lobe: {title_suffix}', fontweight='bold')
        ax_3d.set_box_aspect([1,1,0.8])
        
        # Polar plot
        contour = ax_polar.contourf(phi, theta, brdf_values, levels=15, cmap='plasma')
        ax_polar.set_title('Polar View', fontweight='bold')
        
        # 1D slice
        angles = np.linspace(0, np.pi/2, 100)
        brdf_slice = []
        for angle in angles:
            wi = np.array([np.sin(angle), 0, np.cos(angle)])
            brdf_slice.append(brdf_func(wi, VIEW_DIR, NORMAL))
        
        ax_1d.plot(np.degrees(angles), brdf_slice, 'navy', linewidth=3)
        ax_1d.set_xlabel('Angle (degrees)')
        ax_1d.set_ylabel('BRDF Value')
        ax_1d.set_title('Cross-Section Profile')
        ax_1d.grid(True, alpha=0.3)
        
        # Sample comparison
        uniform_result = estimate_radiance_method('uniform', brdf_func, 200)
        if model == 'phong':
            importance_result = estimate_radiance_method('importance_phong', brdf_func, 200)
        else:
            importance_result = estimate_radiance_method('cosine', brdf_func, 200)
        
        uniform_dirs = np.array([s[0][:2] for s in uniform_result.samples])
        importance_dirs = np.array([s[0][:2] for s in importance_result.samples])
        
        ax_samples.scatter(uniform_dirs[:100, 0], uniform_dirs[:100, 1], 
                          alpha=0.6, s=15, color='red', label='Uniform')
        ax_samples.scatter(importance_dirs[:100, 0], importance_dirs[:100, 1], 
                          alpha=0.6, s=15, color='blue', label='Importance/Cosine')
        
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
        ax_samples.add_patch(circle)
        ax_samples.set_xlim(-1.1, 1.1)
        ax_samples.set_ylim(-1.1, 1.1)
        ax_samples.set_aspect('equal')
        ax_samples.set_title('Sample Distribution')
        ax_samples.legend()
        ax_samples.grid(True, alpha=0.3)
        
        fig.canvas.draw()
    
    # Create interactive controls
    ax_power = plt.axes([0.15, 0.1, 0.3, 0.03])
    ax_roughness = plt.axes([0.15, 0.05, 0.3, 0.03])
    ax_radio = plt.axes([0.6, 0.05, 0.3, 0.15])
    
    slider_power = Slider(ax_power, 'Specular Power', 1, 128, valinit=32, valfmt='%d')
    slider_roughness = Slider(ax_roughness, 'Roughness', 0.01, 1.0, valinit=0.1, valfmt='%.2f')
    radio = RadioButtons(ax_radio, ('phong', 'blinn_phong', 'cook_torrance'))
    
    def update_power(val):
        current_params['power'] = int(slider_power.val)
        update_3d_plots()
    
    def update_roughness(val):
        current_params['roughness'] = slider_roughness.val
        update_3d_plots()
    
    def update_model(label):
        current_params['model'] = label
        update_3d_plots()
    
    slider_power.on_changed(update_power)
    slider_roughness.on_changed(update_roughness)
    radio.on_clicked(update_model)
    
    # Initial plot
    update_3d_plots()
    
    plt.show()

# === CLEAN VISUALIZATION FUNCTIONS ===
def create_professional_brdf_visualization():
    """Create a clean, professional BRDF visualization"""
    
    # Create figure with proper layout
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('BRDF Importance Sampling Analysis', fontsize=18, fontweight='bold', y=0.95)
    
    # Create custom grid layout with better spacing
    gs = gridspec.GridSpec(3, 4, figure=fig, 
                          left=0.06, right=0.96, top=0.90, bottom=0.18,
                          hspace=0.4, wspace=0.35)
    
    # Main plots
    ax1 = fig.add_subplot(gs[0, 0:2])  # BRDF profile
    ax2 = fig.add_subplot(gs[0, 2:4])  # Sample distribution
    ax3 = fig.add_subplot(gs[1, 0:2])  # Method comparison
    ax4 = fig.add_subplot(gs[1, 2:4])  # Efficiency comparison
    ax5 = fig.add_subplot(gs[2, 0:2])  # Convergence
    ax6 = fig.add_subplot(gs[2, 2:4])  # Variance reduction
    
    # Control panel area (bottom) - properly positioned
    control_area = fig.add_axes([0.05, 0.02, 0.9, 0.12])
    control_area.set_xlim(0, 1)
    control_area.set_ylim(0, 1)
    control_area.axis('off')
    control_area.text(0.02, 0.8, 'Interactive Controls:', fontsize=12, fontweight='bold')
    control_area.text(0.02, 0.4, 'Adjust parameters and observe real-time changes in sampling efficiency', 
                     fontsize=10, style='italic', color='gray')
    
    # Initial parameters
    current_params = {'power': 32, 'roughness': 0.1, 'brdf_type': 'phong'}
    
    def update_all_plots():
        # Clear all axes
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.clear()
        
        power = current_params['power']
        roughness = current_params['roughness']
        brdf_type = current_params['brdf_type']
        
        # Select BRDF function
        if brdf_type == 'phong':
            brdf_func = lambda wi, wo, n: phong_brdf(wi, wo, n, power)
            title_suffix = f'Phong (n={power})'
        elif brdf_type == 'blinn_phong':
            brdf_func = lambda wi, wo, n: blinn_phong_brdf(wi, wo, n, power)
            title_suffix = f'Blinn-Phong (n={power})'
        elif brdf_type == 'cook_torrance':
            brdf_func = lambda wi, wo, n: cook_torrance_brdf(wi, wo, n, roughness)
            title_suffix = f'Cook-Torrance (α={roughness:.2f})'
        
        # === Plot 1: BRDF Profile ===
        angles = np.linspace(0, np.pi/2, 100)
        brdf_values = []
        for angle in angles:
            wi = np.array([np.sin(angle), 0, np.cos(angle)])
            brdf_values.append(brdf_func(wi, VIEW_DIR, NORMAL))
        
        ax1.plot(np.degrees(angles), brdf_values, 'navy', linewidth=3, label=brdf_type.replace('_', '-').title())
        ax1.set_xlabel('Angle from Normal (degrees)')
        ax1.set_ylabel('BRDF Value')
        ax1.set_title(f'BRDF Profile: {title_suffix}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # === Generate sampling results ===
        uniform_result = estimate_radiance_method('uniform', brdf_func, 500)
        cosine_result = estimate_radiance_method('cosine', brdf_func, 500)
        if brdf_type == 'phong':
            importance_result = estimate_radiance_method('importance_phong', brdf_func, 500)
        else:
            importance_result = cosine_result  # Fallback
        
        # === Plot 2: Sample Distribution Comparison ===
        uniform_dirs = np.array([s[0] for s in uniform_result.samples])
        importance_dirs = np.array([s[0] for s in importance_result.samples])
        
        # Plot subset of samples for clarity
        n_show = min(200, len(uniform_dirs))
        
        # Create circular boundary
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2, linestyle='--', alpha=0.7)
        ax2.add_patch(circle)
        
        # Plot samples with different colors and shapes
        ax2.scatter(uniform_dirs[:n_show,0], uniform_dirs[:n_show,1], 
                   alpha=0.6, s=25, color='#e74c3c', label='Uniform', 
                   marker='o', edgecolors='white', linewidth=0.5)
        ax2.scatter(importance_dirs[:n_show,0], importance_dirs[:n_show,1], 
                   alpha=0.7, s=25, color='#27ae60', label='Importance', 
                   marker='^', edgecolors='white', linewidth=0.5)
        
        ax2.set_xlim(-1.1, 1.1)
        ax2.set_ylim(-1.1, 1.1)
        ax2.set_xlabel('X Direction')
        ax2.set_ylabel('Y Direction')
        ax2.set_title('Sample Distribution Comparison')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', framealpha=0.9)
        
        # === Plot 3: Method Comparison ===
        methods = ['Uniform', 'Cosine\nWeighted', 'Importance']
        estimates = [uniform_result.estimate, cosine_result.estimate, importance_result.estimate]
        errors = [np.sqrt(uniform_result.variance), np.sqrt(cosine_result.variance), np.sqrt(importance_result.variance)]
        colors = ['#e74c3c', '#f39c12', '#27ae60']
        
        bars = ax3.bar(methods, estimates, yerr=errors, color=colors, alpha=0.8, capsize=8, width=0.6)
        ax3.set_ylabel('Estimated Radiance')
        ax3.set_title('Sampling Method Comparison')
        
        # Add analytical solution if available
        if brdf_type == 'phong':
            analytical = analytical_phong_integral(power)
            ax3.axhline(y=analytical, color='black', linestyle='--', linewidth=2,
                       label=f'Analytical: {analytical:.4f}')
            ax3.legend()
        
        # Add value labels on bars
        for bar, est, err in zip(bars, estimates, errors):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + err + 0.01,
                    f'{est:.4f}', ha='center', va='bottom', fontweight='bold')
        
        ax3.grid(True, alpha=0.3, axis='y')
        
        # === Plot 4: Efficiency Comparison ===
        variances = [uniform_result.variance, cosine_result.variance, importance_result.variance]
        
        # Calculate relative efficiency (normalized to uniform sampling)
        uniform_var = uniform_result.variance
        relative_efficiencies = [1.0,  # Uniform baseline
                               uniform_var / cosine_result.variance if cosine_result.variance > 0 else 1.0,
                               uniform_var / importance_result.variance if importance_result.variance > 0 else 1.0]
        
        bars = ax4.bar(methods, relative_efficiencies, color=colors, alpha=0.8, width=0.6)
        ax4.set_ylabel('Relative Efficiency\n(vs Uniform Sampling)')
        ax4.set_title('Sampling Efficiency Comparison')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, eff in zip(bars, relative_efficiencies):
            height = bar.get_height()
            if eff > 1.0:
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{eff:.1f}×', ha='center', va='bottom', fontweight='bold', fontsize=11)
            else:
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{eff:.2f}×', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Add improvement callout for importance sampling
        if relative_efficiencies[2] > 1.0:
            ax4.annotate(f'{relative_efficiencies[2]:.1f}× more efficient', 
                        xy=(2, relative_efficiencies[2]), xytext=(1.5, relative_efficiencies[2] + 2),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2),
                        fontsize=11, fontweight='bold', color='green',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        # === Plot 5: Convergence Analysis ===
        sample_sizes = [25, 50, 100, 250, 500, 1000, 2000]
        uniform_estimates = []
        importance_estimates = []
        
        for n in sample_sizes:
            u_result = estimate_radiance_method('uniform', brdf_func, n)
            i_result = estimate_radiance_method('importance_phong', brdf_func, n) if brdf_type == 'phong' else u_result
            uniform_estimates.append(u_result.estimate)
            importance_estimates.append(i_result.estimate)
        
        ax5.semilogx(sample_sizes, uniform_estimates, 'o-', color='#e74c3c', 
                    linewidth=2, markersize=6, label='Uniform')
        ax5.semilogx(sample_sizes, importance_estimates, 's-', color='#27ae60', 
                    linewidth=2, markersize=6, label='Importance')
        
        if brdf_type == 'phong':
            analytical = analytical_phong_integral(power)
            ax5.axhline(y=analytical, color='black', linestyle='--', linewidth=2, label='Analytical')
        
        ax5.set_xlabel('Number of Samples')
        ax5.set_ylabel('Estimated Radiance')
        ax5.set_title('Convergence Analysis')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # === Plot 6: Variance Reduction ===
        uniform_vars = []
        importance_vars = []
        
        for n in sample_sizes:
            u_result = estimate_radiance_method('uniform', brdf_func, n)
            i_result = estimate_radiance_method('importance_phong', brdf_func, n) if brdf_type == 'phong' else u_result
            uniform_vars.append(u_result.variance)
            importance_vars.append(i_result.variance)
        
        # Calculate variance reduction ratios with safety checks
        variance_ratios = []
        for u_var, i_var in zip(uniform_vars, importance_vars):
            if i_var > 1e-10:  # Avoid division by very small numbers
                ratio = u_var / i_var
                variance_ratios.append(min(ratio, 100))  # Cap at 100x for readability
            else:
                variance_ratios.append(1.0)  # No improvement if importance variance is too small
        
        ax6.semilogx(sample_sizes, variance_ratios, 'o-', color='#9b59b6', 
                    linewidth=3, markersize=8, markerfacecolor='white', 
                    markeredgecolor='#9b59b6', markeredgewidth=2)
        ax6.set_xlabel('Number of Samples')
        ax6.set_ylabel('Variance Reduction Factor\n(Uniform/Importance)')
        ax6.set_title('Variance Reduction Analysis')
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=1, color='red', linestyle=':', alpha=0.7, linewidth=2, label='No improvement')
        ax6.set_ylim(0.5, max(max(variance_ratios), 10))
        ax6.legend(loc='upper right')
        
        # Add average improvement text
        avg_improvement = np.mean([r for r in variance_ratios if r > 1.0])
        if avg_improvement > 1.0:
            ax6.text(0.05, 0.85, f'Avg. Reduction:\n{avg_improvement:.1f}×', 
                    transform=ax6.transAxes, fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        fig.canvas.draw()
    
    # Add interactive controls with better spacing
    ax_power = plt.axes([0.1, 0.06, 0.2, 0.02])
    ax_roughness = plt.axes([0.1, 0.03, 0.2, 0.02])
    ax_radio = plt.axes([0.4, 0.02, 0.5, 0.06])
    
    slider_power = Slider(ax_power, 'Specular Power', 1, 128, valinit=32, valfmt='%d')
    slider_roughness = Slider(ax_roughness, 'Roughness', 0.01, 1.0, valinit=0.1, valfmt='%.2f')
    
    # Style the radio buttons
    radio = RadioButtons(ax_radio, ('Phong BRDF', 'Blinn-Phong BRDF', 'Cook-Torrance BRDF'), 
                        active=0, activecolor='#27ae60')
    radio.labels[0].set_fontsize(10)
    radio.labels[1].set_fontsize(10)
    radio.labels[2].set_fontsize(10)
    
    # Map radio button labels to internal names
    brdf_mapping = {
        'Phong BRDF': 'phong',
        'Blinn-Phong BRDF': 'blinn_phong',
        'Cook-Torrance BRDF': 'cook_torrance'
    }
    
    def update_power(val):
        current_params['power'] = int(slider_power.val)
        update_all_plots()
    
    def update_roughness(val):
        current_params['roughness'] = slider_roughness.val
        update_all_plots()
    
    def update_brdf(label):
        current_params['brdf_type'] = brdf_mapping[label]
        update_all_plots()
    
    slider_power.on_changed(update_power)
    slider_roughness.on_changed(update_roughness)
    radio.on_clicked(update_brdf)
    
    # Initial plot
    update_all_plots()
    
    plt.show()

# === SIMPLE STATIC COMPARISON ===
def create_clean_static_comparison():
    """Create a clean static comparison without interactive elements"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('BRDF Importance Sampling: Static Analysis', fontsize=16, fontweight='bold')
    
    # Generate results
    uniform_result = estimate_radiance_method('uniform', phong_brdf, 1000)
    importance_result = estimate_radiance_method('importance_phong', phong_brdf, 1000)
    analytical = analytical_phong_integral()
    
    # Plot 1: BRDF Profile
    angles = np.linspace(0, np.pi/2, 100)
    brdf_values = []
    for angle in angles:
        wi = np.array([np.sin(angle), 0, np.cos(angle)])
        brdf_values.append(phong_brdf(wi, VIEW_DIR, NORMAL))
    
    ax1.plot(np.degrees(angles), brdf_values, 'navy', linewidth=3)
    ax1.set_xlabel('Angle from Normal (degrees)')
    ax1.set_ylabel('BRDF Value')
    ax1.set_title(f'Phong BRDF Profile (n={DEFAULT_SPECULAR_POWER})')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sample Distributions Comparison
    uniform_dirs = np.array([s[0] for s in uniform_result.samples])
    importance_dirs = np.array([s[0] for s in importance_result.samples])
    
    ax2.scatter(uniform_dirs[:200,0], uniform_dirs[:200,1], 
               alpha=0.6, s=20, color='red', label='Uniform', edgecolors='none')
    ax2.scatter(importance_dirs[:200,0], importance_dirs[:200,1], 
               alpha=0.6, s=20, color='blue', label='Importance', edgecolors='none')
    
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax2.add_patch(circle)
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_xlabel('X Direction')
    ax2.set_ylabel('Y Direction')
    ax2.set_title('Sample Distribution Comparison')
    ax2.set_aspect('equal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Method Comparison
    methods = ['Uniform', 'Importance', 'Analytical']
    estimates = [uniform_result.estimate, importance_result.estimate, analytical]
    errors = [np.sqrt(uniform_result.variance), np.sqrt(importance_result.variance), 0]
    colors = ['#e74c3c', '#27ae60', '#34495e']
    
    bars = ax3.bar(methods, estimates, yerr=errors, color=colors, alpha=0.8, capsize=8)
    ax3.set_ylabel('Radiance Estimate')
    ax3.set_title('Method Comparison')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, est, err in zip(bars, estimates, errors):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + err + 0.005,
                f'{est:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Performance Metrics (simplified and cleaner)
    # Show key metrics in a cleaner format
    metrics_data = {
        'Uniform': {'Variance': uniform_result.variance, 'Std Error': np.sqrt(uniform_result.variance)},
        'Importance': {'Variance': importance_result.variance, 'Std Error': np.sqrt(importance_result.variance)}
    }
    
    methods_clean = ['Uniform', 'Importance']
    variances = [uniform_result.variance, importance_result.variance]
    std_errors = [np.sqrt(v) for v in variances]
    
    x = np.arange(len(methods_clean))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, variances, width, label='Variance', color='#e74c3c', alpha=0.7)
    bars2 = ax4.bar(x + width/2, std_errors, width, label='Std Error', color='#3498db', alpha=0.7)
    
    ax4.set_ylabel('Values')
    ax4.set_title('Variance and Standard Error Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods_clean)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_yscale('log')
    
    # Add improvement ratio with safety check
    if importance_result.variance > 1e-10:
        variance_improvement = min(uniform_result.variance / importance_result.variance, 100)  # Cap at 100x
    else:
        variance_improvement = 1.0  # No meaningful improvement
    ax4.text(0.5, 0.85, f'Variance Reduction: {variance_improvement:.1f}×', 
             transform=ax4.transAxes, ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    
    # Add summary text in a better location with capped variance improvement
    if importance_result.variance > 1e-10:
        variance_improvement = min(uniform_result.variance / importance_result.variance, 100)
    else:
        variance_improvement = 1.0
        
    fig.text(0.02, 0.95, f'Variance Reduction: {variance_improvement:.1f}× | ' +
             f'Analytical: {analytical:.4f} | ' + 
             f'Samples: {N_SAMPLES}', 
             fontsize=10, fontweight='bold', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.show()

# === CONVERGENCE ANALYSIS ===
def analyze_convergence_clean():
    """Clean convergence analysis with professional presentation"""
    
    sample_sizes = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
    n_runs = 5
    
    uniform_means = []
    uniform_stds = []
    importance_means = []
    importance_stds = []
    
    print("Running convergence analysis...")
    
    for i, n in enumerate(sample_sizes):
        print(f"Testing {n} samples ({i+1}/{len(sample_sizes)})")
        
        uniform_results = []
        importance_results = []
        
        for _ in range(n_runs):
            u_result = estimate_radiance_method('uniform', phong_brdf, n)
            i_result = estimate_radiance_method('importance_phong', phong_brdf, n)
            uniform_results.append(u_result.estimate)
            importance_results.append(i_result.estimate)
        
        uniform_means.append(np.mean(uniform_results))
        uniform_stds.append(np.std(uniform_results))
        importance_means.append(np.mean(importance_results))
        importance_stds.append(np.std(importance_results))
    
    # Create clean plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Convergence Analysis with Confidence Intervals', fontsize=16, fontweight='bold')
    
    analytical = analytical_phong_integral()
    
    # Plot 1: Convergence with error bars
    ax1.errorbar(sample_sizes, uniform_means, yerr=uniform_stds, 
                label='Uniform Sampling', marker='o', capsize=5, linewidth=2, 
                color='#e74c3c', markersize=6)
    ax1.errorbar(sample_sizes, importance_means, yerr=importance_stds, 
                label='Importance Sampling', marker='s', capsize=5, linewidth=2, 
                color='#27ae60', markersize=6)
    ax1.axhline(y=analytical, color='black', linestyle='--', linewidth=2,
               label=f'Analytical Solution: {analytical:.4f}')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of Samples')
    ax1.set_ylabel('Estimated Radiance')
    ax1.set_title('Convergence Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Variance reduction
    variance_reduction = np.array(uniform_stds) / np.array(importance_stds)
    ax2.semilogx(sample_sizes, variance_reduction, 'o-', linewidth=3, 
                markersize=8, color='#9b59b6')
    ax2.set_xlabel('Number of Samples')
    ax2.set_ylabel('Standard Deviation Ratio (Uniform/Importance)')
    ax2.set_title('Variance Reduction Factor')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='red', linestyle=':', alpha=0.7, label='No improvement')
    ax2.legend()
    
    # Add summary statistics with capped values
    avg_improvement = np.mean([r for r in variance_reduction if 1.0 < r < 50])
    if avg_improvement > 1.0:
        fig.text(0.02, 0.02, f'Average Standard Deviation Reduction: {avg_improvement:.1f}×', 
                 fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    else:
        fig.text(0.02, 0.02, 'Variance reduction varies by sample size', 
                 fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return sample_sizes, uniform_means, uniform_stds, importance_means, importance_stds

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("=== Enhanced BRDF Importance Sampling Visualizer ===")
    print(f"Default Specular Power: {DEFAULT_SPECULAR_POWER}")
    print(f"Number of Samples: {N_SAMPLES}")
    print()
    
    # Quick comparison
    print("Running numerical comparison...")
    uniform_result = estimate_radiance_method('uniform', phong_brdf)
    importance_result = estimate_radiance_method('importance_phong', phong_brdf)
    analytical = analytical_phong_integral()
    
    print(f"Analytical Solution      : {analytical:.6f}")
    print(f"Uniform Sampling        : {uniform_result.estimate:.6f} ± {np.sqrt(uniform_result.variance):.6f}")
    print(f"Importance Sampling     : {importance_result.estimate:.6f} ± {np.sqrt(importance_result.variance):.6f}")
    print(f"Variance Reduction      : {uniform_result.variance / importance_result.variance:.1f}×")
    print()
    
    # Clean menu
    while True:
        print("\n" + "="*60)
        print("VISUALIZATION OPTIONS")
        print("="*60)
        print("1. 🎯 Interactive BRDF Explorer (Recommended)")
        print("2. 📊 Clean Static Comparison")
        print("3. 📈 Convergence Analysis")
        print("4. 🔄 Interactive 3D Explorer")
        print("5. 📊 BRDF Model Comparison")
        print("6. 🚪 Exit")
        print("="*60)
        
        choice = input("Select option (1-7): ").strip()
        
        if choice == '1':
            print("\n🚀 Opening Interactive BRDF Explorer...")
            print("   • Use sliders to adjust parameters")
            print("   • Switch between BRDF models")
            print("   • Observe real-time changes")
            create_professional_brdf_visualization()
        elif choice == '2':
            print("\n📊 Creating Static Comparison...")
            create_clean_static_comparison()
        elif choice == '3':
            print("\n📈 Running Convergence Analysis...")
            analyze_convergence_clean()
        elif choice == '4':
            print("\n🔄 Opening Interactive 3D Explorer...")
            print("   • Real-time 3D BRDF exploration")
            print("   • Live parameter adjustments")
            print("   • Multiple visualization views")
            create_interactive_3d_explorer()
        elif choice == '5':
            print("\n📊 Creating BRDF Model Comparison...")
            print("   • Side-by-side model comparison")
            print("   • 3D and 1D visualizations")
            print("   • Phong, Blinn-Phong, Cook-Torrance")
            create_brdf_model_comparison()
        elif choice == '6':
            print("\n👋 Thanks for using the BRDF Visualizer!")
            break
        else:
            print("❌ Invalid choice. Please select 1-7.")