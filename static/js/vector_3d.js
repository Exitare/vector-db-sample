/**
 * 3D Vector Visualization
 * Creates an interactive 3D scatter plot of vector embeddings using Three.js
 * Color-coded by cancer type for pattern exploration
 */

class Vector3DVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.points = null;
        this.data = null;
        
        // Color palette for different cancer types
        this.colorPalette = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
            '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43',
            '#C44569', '#F8B500', '#6C5CE7', '#A29BFE', '#6C5B7B',
            '#F39C12', '#E74C3C', '#9B59B6', '#3498DB', '#1ABC9C'
        ];
        
        this.init();
    }
    
    init() {
        console.log('Initializing 3D visualizer...');
        
        if (!this.container) {
            console.error('3D container not found');
            this.showError('3D container element not found');
            return;
        }
        
        // Check if Three.js is loaded
        if (typeof THREE === 'undefined') {
            console.error('Three.js is not loaded');
            this.showError('Three.js library failed to load');
            return;
        }
        
        console.log('Three.js loaded successfully');
        
        // Test WebGL support
        if (!this.testWebGLSupport()) {
            console.error('WebGL not supported');
            this.showError('WebGL is not supported by your browser');
            return;
        }
        
        try {
            console.log('Setting up scene...');
            this.setupScene();
            console.log('Scene setup complete');
            
            console.log('Setting up camera...');
            this.setupCamera();
            console.log('Camera setup complete');
            
            console.log('Setting up renderer...');
            this.setupRenderer();
            console.log('Renderer setup complete');
            
            console.log('Setting up controls...');
            this.setupControls();  // This might fail if OrbitControls isn't loaded
            console.log('Controls setup complete');
            
            console.log('Setting up lights...');
            this.setupLights();
            console.log('Lights setup complete');
            
            console.log('Loading data...');
            this.loadData();
            console.log('Data loading initiated');
            
            console.log('Starting animation...');
            this.animate();
            
            console.log('3D visualization initialized successfully');
        } catch (error) {
            console.error('Error during 3D initialization:', error);
            this.showError('Failed to initialize 3D visualization: ' + error.message);
            return;
        }
        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
    }
    
    testWebGLSupport() {
        try {
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
            return !!gl;
        } catch (e) {
            return false;
        }
    }
    
    setupScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xf8f9fa);
        
        // Add grid helper for reference
        const gridHelper = new THREE.GridHelper(20, 20, 0xcccccc, 0xeeeeee);
        this.scene.add(gridHelper);
        
        // Add axes helper
        const axesHelper = new THREE.AxesHelper(10);
        this.scene.add(axesHelper);
    }
    
    setupCamera() {
        const width = this.container.clientWidth || 400;
        const height = this.container.clientHeight || 400;
        
        this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        this.camera.position.set(15, 15, 15);
        this.camera.lookAt(0, 0, 0);
    }
    
    setupRenderer() {
        const width = this.container.clientWidth || 400;
        const height = this.container.clientHeight || 400;
        
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true,
            alpha: true 
        });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        this.container.appendChild(this.renderer.domElement);
    }
    
    setupControls() {
        console.log('Setting up controls...');
        
        // For now, skip OrbitControls and just make it work without them
        console.log('Skipping OrbitControls - visualization will work with basic camera');
        this.controls = null;
        
        // Add basic mouse interaction without OrbitControls
        this.addBasicMouseControls();
    }
    
    addBasicMouseControls() {
        // Simple camera rotation without OrbitControls
        let mouseDown = false;
        let mouseX = 0;
        let mouseY = 0;
        
        this.renderer.domElement.addEventListener('mousedown', (event) => {
            mouseDown = true;
            mouseX = event.clientX;
            mouseY = event.clientY;
        });
        
        this.renderer.domElement.addEventListener('mouseup', () => {
            mouseDown = false;
        });
        
        this.renderer.domElement.addEventListener('mousemove', (event) => {
            if (!mouseDown) return;
            
            const deltaX = event.clientX - mouseX;
            const deltaY = event.clientY - mouseY;
            
            // Simple camera rotation
            this.camera.position.x = Math.cos(deltaX * 0.01) * 20;
            this.camera.position.z = Math.sin(deltaX * 0.01) * 20;
            this.camera.position.y += deltaY * 0.1;
            
            this.camera.lookAt(0, 0, 0);
            
            mouseX = event.clientX;
            mouseY = event.clientY;
        });
    }
    
    setupLights() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);
        
        // Directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(20, 20, 20);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);
    }
    
    async loadData() {
        try {
            console.log('Loading 3D visualization data...');
            const response = await fetch('/api/vectors-3d');
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log('3D API Response:', {
                success: data.success,
                vectorCount: data.vectors?.length,
                totalAvailable: data.total_available,
                returned: data.returned
            });
            
            if (data.success) {
                this.data = data.vectors;
                console.log('First data point:', this.data[0]);
                console.log('About to create visualization...');
                this.createVisualization();
                console.log('Visualization created, hiding loading state...');
                this.hideLoadingState();
                console.log('Loading state hidden');
            } else {
                console.error('API returned error:', data.error);
                this.showError('Failed to load 3D data: ' + data.error);
            }
        } catch (error) {
            console.error('Error loading 3D data:', error);
            this.showError('Network error loading 3D visualization: ' + error.message);
        }
    }
    
    createVisualization() {
        console.log('createVisualization() called');
        
        if (!this.data || this.data.length === 0) {
            console.error('No data available for 3D visualization');
            this.showError('No data available for 3D visualization');
            return;
        }
        
        console.log('Creating visualization with', this.data.length, 'data points');
        
        try {
            // Get unique cancer types for color mapping
            const cancerTypes = [...new Set(this.data.map(d => d.cancer_type))];
            console.log('Cancer types found:', cancerTypes);
            
            const colorMap = {};
            cancerTypes.forEach((type, index) => {
                colorMap[type] = this.colorPalette[index % this.colorPalette.length];
            });
            
            console.log('Color map created:', colorMap);
            
            // Create geometry and materials
            const geometry = new THREE.BufferGeometry();
            const positions = [];
            const colors = [];
            
            // Extract positions and colors from data
            this.data.forEach(point => {
                // Use first 3 dimensions of embedding for 3D positioning
                // Scale positions to fit nicely in the scene
                const scale = 10;
                positions.push(
                    (point.embedding[0] || 0) * scale,
                    (point.embedding[1] || 0) * scale, 
                    (point.embedding[2] || 0) * scale
                );
                
                // Convert hex color to RGB
                const color = new THREE.Color(colorMap[point.cancer_type] || '#999999');
                colors.push(color.r, color.g, color.b);
            });
            
            console.log('Positions and colors created, setting geometry attributes...');
            
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            
            console.log('Geometry attributes set, creating material...');
            
            // Create point material
            const material = new THREE.PointsMaterial({
                size: 0.5,
                vertexColors: true,
                transparent: true,
                opacity: 0.8,
                sizeAttenuation: true
            });
            
            console.log('Material created, creating points mesh...');
            
            // Create points mesh
            this.points = new THREE.Points(geometry, material);
            this.scene.add(this.points);
            
            console.log('Points added to scene, creating legend...');
            
            // Create legend
            this.createLegend(colorMap);
            
            console.log('Legend created, setting up interactivity...');
            
            // Add interactivity
            this.setupInteractivity();
            
            console.log('Visualization creation complete!');
        } catch (error) {
            console.error('Error in createVisualization:', error);
            this.showError('Error creating 3D visualization: ' + error.message);
        }
    }
    
    createLegend(colorMap) {
        const legendContainer = document.createElement('div');
        legendContainer.className = 'visualization-3d-legend';
        legendContainer.innerHTML = `
            <div class="legend-header">
                <h6 class="fw-bold mb-2">
                    <i class="fas fa-palette me-1"></i>Cancer Types
                </h6>
            </div>
            <div class="legend-items">
                ${Object.entries(colorMap).map(([type, color]) => `
                    <div class="legend-item" data-cancer-type="${type}">
                        <span class="legend-color" style="background-color: ${color}"></span>
                        <span class="legend-label">${type}</span>
                    </div>
                `).join('')}
            </div>
        `;
        
        this.container.appendChild(legendContainer);
    }
    
    setupInteractivity() {
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        
        this.renderer.domElement.addEventListener('click', (event) => {
            const rect = this.renderer.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
            
            raycaster.setFromCamera(mouse, this.camera);
            const intersects = raycaster.intersectObject(this.points);
            
            if (intersects.length > 0) {
                const index = intersects[0].index;
                const pointData = this.data[index];
                this.showPointInfo(pointData, event);
            }
        });
    }
    
    showPointInfo(pointData, event) {
        // Create or update tooltip
        let tooltip = document.getElementById('vector-3d-tooltip');
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.id = 'vector-3d-tooltip';
            tooltip.className = 'vector-3d-tooltip';
            document.body.appendChild(tooltip);
        }
        
        tooltip.innerHTML = `
            <div class="tooltip-header">
                <strong>Vector: ${pointData.id}</strong>
            </div>
            <div class="tooltip-content">
                <div><strong>Cancer Type:</strong> ${pointData.cancer_type}</div>
                <div><strong>Position:</strong> [${pointData.embedding.slice(0, 3).map(v => v.toFixed(3)).join(', ')}]</div>
                ${pointData.metadata ? `<div><strong>Metadata:</strong> ${JSON.stringify(pointData.metadata, null, 2)}</div>` : ''}
            </div>
        `;
        
        tooltip.style.display = 'block';
        tooltip.style.left = event.pageX + 10 + 'px';
        tooltip.style.top = event.pageY + 10 + 'px';
        
        // Hide tooltip after 3 seconds
        setTimeout(() => {
            tooltip.style.display = 'none';
        }, 3000);
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        if (this.controls) {
            this.controls.update();
        }
        
        // Optional: rotate points slowly for better visibility
        if (this.points) {
            this.points.rotation.y += 0.001;
        }
        
        this.renderer.render(this.scene, this.camera);
    }
    
    onWindowResize() {
        if (!this.container) return;
        
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
    
    hideLoadingState() {
        console.log('hideLoadingState() called');
        const placeholder = this.container.querySelector('#vector-3d-placeholder');
        if (placeholder) {
            console.log('Found placeholder, hiding it');
            console.log('Placeholder current display:', placeholder.style.display);
            console.log('Placeholder computed style:', window.getComputedStyle(placeholder).display);
            
            // Try multiple methods to hide it
            placeholder.style.display = 'none';
            placeholder.style.visibility = 'hidden';
            placeholder.style.opacity = '0';
            placeholder.classList.add('d-none');
            
            // Also try removing it completely
            setTimeout(() => {
                if (placeholder.parentNode) {
                    console.log('Removing placeholder from DOM');
                    placeholder.parentNode.removeChild(placeholder);
                }
            }, 100);
            
            console.log('Placeholder hidden with multiple methods');
        } else {
            console.log('No placeholder found to hide');
        }
    }
    
    showError(message) {
        const placeholder = this.container.querySelector('#vector-3d-placeholder');
        if (placeholder) {
            placeholder.innerHTML = `
                <div class="text-center text-danger">
                    <i class="fas fa-exclamation-triangle fa-2x mb-2"></i>
                    <div>${message}</div>
                </div>
            `;
        }
    }
    
    destroy() {
        if (this.renderer) {
            this.container.removeChild(this.renderer.domElement);
        }
        if (this.controls) {
            this.controls.dispose();
        }
        window.removeEventListener('resize', () => this.onWindowResize());
    }
}

// Initialize 3D visualization when DOM is loaded
// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('=== 3D VISUALIZATION INIT START ===');
    console.log('DOM loaded, checking for 3D container...');
    
    const container = document.getElementById('vector-3d-container');
    if (container) {
        console.log('✓ 3D container found, initializing visualization...');
        
        // Wait a bit for all scripts to load
        setTimeout(() => {
            console.log('Starting 3D visualization initialization...');
            console.log('THREE available:', typeof THREE !== 'undefined');
            
            try {
                console.log('Creating Vector3DVisualizer instance...');
                window.vector3DVisualizer = new Vector3DVisualizer('vector-3d-container');
                console.log('✓ Vector3DVisualizer created successfully');
            } catch (error) {
                console.error('✗ CRITICAL ERROR - Failed to create Vector3DVisualizer:', error);
                console.error('Error stack:', error.stack);
                
                // Show detailed error in the container
                const placeholder = container.querySelector('#vector-3d-placeholder');
                if (placeholder) {
                    placeholder.innerHTML = `
                        <div class="text-center text-danger">
                            <i class="fas fa-exclamation-triangle fa-2x mb-2"></i>
                            <div class="fw-bold">3D Visualization Failed</div>
                            <small class="text-muted d-block mt-2">${error.message}</small>
                            <div class="mt-2">
                                <button class="btn btn-sm btn-outline-danger" onclick="console.log('Detailed error:', ${JSON.stringify(error.stack)})">
                                    Show Console Details
                                </button>
                            </div>
                        </div>
                    `;
                }
            }
        }, 1000); // Increased timeout to 1 second
    } else {
        console.error('✗ 3D container not found in DOM');
    }
});
