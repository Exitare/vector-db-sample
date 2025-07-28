/**
 * 3D Vector Visualization
 * Creates an interactive 3D scatter plot of vector embeddings using Three.js
 * Color-coded by cancer type for pattern exploration
 */

class Vector3DVisualizer {
    constructor(containerId, data) {
        console.log('🚀 Initializing Vector3DVisualizer with data:', data);
        this.container = document.getElementById(containerId);
        this.data = data;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.raycaster = null;
        this.mouse = null;
        this.clickablePoints = []; // Array to store individual mesh objects
        this.pointData = []; // Array to store corresponding data for each point
        
        // Define a color palette for different cancer types
        this.colorPalette = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#F4D03F'
        ];
        
        // Make this instance globally available for event delegation
        window.vector3DVisualizer = this;
        console.log('✅ Vector3DVisualizer instance assigned to window.vector3DVisualizer');
        
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
        // Force the container to have the correct dimensions
        const containerStyles = window.getComputedStyle(this.container);
        let width = this.container.clientWidth || parseInt(containerStyles.width) || 800;
        let height = this.container.clientHeight || parseInt(containerStyles.height) || 600;
        
        // For fullscreen 3D page, ensure we use viewport dimensions
        const isFullscreen = this.container.classList.contains('vector-3d-container') && 
                            this.container.parentElement.classList.contains('vector-3d-fullscreen');
        
        if (isFullscreen) {
            // Calculate 95vh for fullscreen mode
            height = Math.floor(window.innerHeight * 0.95);
            width = Math.max(width, window.innerWidth - 40); // Account for margins
        }
        
        this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        this.camera.position.set(15, 15, 15);
        this.camera.lookAt(0, 0, 0);
    }
    
    setupRenderer() {
        // Force the container to have the correct dimensions
        const containerStyles = window.getComputedStyle(this.container);
        let width = this.container.clientWidth || parseInt(containerStyles.width) || 800;
        let height = this.container.clientHeight || parseInt(containerStyles.height) || 600;
        
        // For fullscreen 3D page, ensure we use viewport dimensions
        const isFullscreen = this.container.classList.contains('vector-3d-container') && 
                            this.container.parentElement.classList.contains('vector-3d-fullscreen');
        
        if (isFullscreen) {
            // Calculate 95vh for fullscreen mode
            height = Math.floor(window.innerHeight * 0.95);
            width = Math.max(width, window.innerWidth - 40); // Account for margins
        }
        
        console.log(`Setting up renderer with dimensions: ${width}x${height}`);
        console.log(`Container clientWidth: ${this.container.clientWidth}, clientHeight: ${this.container.clientHeight}`);
        console.log(`Is fullscreen: ${isFullscreen}`);
        
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
        console.log('THREE.OrbitControls available:', typeof THREE.OrbitControls);
        console.log('THREE.OrbitControls:', THREE.OrbitControls);
        
        // Check if OrbitControls is available
        if (typeof THREE.OrbitControls !== 'undefined') {
            console.log('✅ OrbitControls found, setting up advanced camera controls');
            
            try {
                this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
                console.log('OrbitControls instance created successfully');
                
                // Configure OrbitControls settings for full rotation freedom
                this.controls.enableDamping = true;
                this.controls.dampingFactor = 0.05;
                this.controls.screenSpacePanning = false;
                this.controls.minDistance = 3;
                this.controls.maxDistance = 100;
                
                // Allow full rotation in all directions
                this.controls.maxPolarAngle = Math.PI; // Allow full vertical rotation (0 to 180 degrees)
                this.controls.minPolarAngle = 0; // Allow looking straight down
                this.controls.maxAzimuthAngle = Infinity; // Allow unlimited horizontal rotation
                this.controls.minAzimuthAngle = -Infinity; // Allow unlimited horizontal rotation
                
                // Enable all types of controls
                this.controls.enableRotate = true;
                this.controls.enableZoom = true;
                this.controls.enablePan = true;
                
                // Set rotation and interaction speeds
                this.controls.rotateSpeed = 1.0;
                this.controls.zoomSpeed = 1.2;
                this.controls.panSpeed = 0.8;
                
                // Ensure mouse buttons are properly configured
                this.controls.mouseButtons = {
                    LEFT: THREE.MOUSE.ROTATE,    // Left mouse button for rotation
                    MIDDLE: THREE.MOUSE.DOLLY,   // Middle mouse button for zoom
                    RIGHT: THREE.MOUSE.PAN       // Right mouse button for panning
                };
                
                // Enable touch controls for mobile
                this.controls.touches = {
                    ONE: THREE.TOUCH.ROTATE,     // One finger for rotation
                    TWO: THREE.TOUCH.DOLLY_PAN   // Two fingers for zoom and pan
                };
                
                console.log('OrbitControls configured successfully with full rotation freedom');
                console.log('Controls settings:', {
                    enableRotate: this.controls.enableRotate,
                    enableZoom: this.controls.enableZoom,
                    enablePan: this.controls.enablePan,
                    maxAzimuthAngle: this.controls.maxAzimuthAngle,
                    minAzimuthAngle: this.controls.minAzimuthAngle,
                    rotateSpeed: this.controls.rotateSpeed
                });
                
            } catch (error) {
                console.error('Failed to create OrbitControls:', error);
                console.warn('⚠️ Falling back to basic mouse controls due to OrbitControls error');
                this.controls = null;
                this.addBasicMouseControls();
            }
            
        } else {
            console.warn('⚠️ OrbitControls not found, falling back to basic mouse controls');
            this.controls = null;
            this.addBasicMouseControls();
        }
    }
    
    addBasicMouseControls() {
        // Enhanced camera rotation without OrbitControls
        let mouseDown = false;
        let mouseX = 0;
        let mouseY = 0;
        let phi = 0; // Horizontal rotation
        let theta = Math.PI / 4; // Vertical rotation
        let radius = 20; // Distance from center - make it mutable
        
        this.renderer.domElement.addEventListener('mousedown', (event) => {
            if (event.button === 0) { // Left mouse button only
                mouseDown = true;
                mouseX = event.clientX;
                mouseY = event.clientY;
                event.preventDefault();
            }
        });
        
        document.addEventListener('mouseup', () => {
            mouseDown = false;
        });
        
        this.renderer.domElement.addEventListener('mousemove', (event) => {
            if (!mouseDown) return;
            
            const deltaX = event.clientX - mouseX;
            const deltaY = event.clientY - mouseY;
            
            // Update rotation angles
            phi -= deltaX * 0.01; // Horizontal rotation (left/right)
            theta = Math.max(0.1, Math.min(Math.PI - 0.1, theta + deltaY * 0.01)); // Vertical rotation (up/down)
            
            // Calculate new camera position using spherical coordinates
            this.camera.position.x = radius * Math.sin(theta) * Math.cos(phi);
            this.camera.position.y = radius * Math.cos(theta);
            this.camera.position.z = radius * Math.sin(theta) * Math.sin(phi);
            
            this.camera.lookAt(0, 0, 0);
            
            mouseX = event.clientX;
            mouseY = event.clientY;
            event.preventDefault();
        });
        
        // Add zoom with mouse wheel
        this.renderer.domElement.addEventListener('wheel', (event) => {
            const zoomSpeed = 0.1;
            const direction = event.deltaY > 0 ? 1 : -1;
            radius = Math.max(5, Math.min(50, radius + direction * zoomSpeed * radius));
            
            // Update camera position with new radius
            this.camera.position.x = radius * Math.sin(theta) * Math.cos(phi);
            this.camera.position.y = radius * Math.cos(theta);
            this.camera.position.z = radius * Math.sin(theta) * Math.sin(phi);
            
            this.camera.lookAt(0, 0, 0);
            
            event.preventDefault();
        });
        
        console.log('Basic mouse controls with full rotation enabled');
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
        
        // Debug: Log first few data points to understand structure
        console.log('First data point:', this.data[0]);
        console.log('Sample embedding:', this.data[0]?.embedding);
        console.log('Embedding type:', typeof this.data[0]?.embedding);
        console.log('Embedding is array:', Array.isArray(this.data[0]?.embedding));
        
        try {
            // Get unique cancer types for color mapping
            const cancerTypes = [...new Set(this.data.map(d => d?.cancer_type || 'Unknown').filter(Boolean))];
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
            this.data.forEach((point, index) => {
                // Skip invalid points
                if (!point || typeof point !== 'object') {
                    console.warn(`Skipping invalid point at index ${index}:`, point);
                    return;
                }
                
                console.log(`Processing point ${index}:`, point);
                
                // Check if embedding exists and has required dimensions
                if (!point.embedding || !Array.isArray(point.embedding)) {
                    console.warn(`Point ${index} has invalid embedding:`, point.embedding);
                    return; // Skip this point
                }
                
                if (point.embedding.length < 3) {
                    console.warn(`Point ${index} embedding has less than 3 dimensions:`, point.embedding.length);
                    // Pad with zeros if needed
                    while (point.embedding.length < 3) {
                        point.embedding.push(0);
                    }
                }
                
                // Use first 3 dimensions of embedding for 3D positioning
                // Scale positions to fit nicely in the scene
                const scale = 10;
                positions.push(
                    (point.embedding[0] || 0) * scale,
                    (point.embedding[1] || 0) * scale, 
                    (point.embedding[2] || 0) * scale
                );
                
                // Convert hex color to RGB
                const cancerType = point.cancer_type || 'Unknown';
                const color = new THREE.Color(colorMap[cancerType] || '#999999');
                colors.push(color.r, color.g, color.b);
            });
            
            console.log('Positions and colors created, creating individual meshes for clickability...');
            
            // Create individual meshes for each point (better for clicking than THREE.Points)
            this.clickablePoints = []; // Store individual point meshes
            this.pointsGroup = new THREE.Group(); // Group to hold all point meshes
            
            const sphereGeometry = new THREE.SphereGeometry(0.08, 8, 8); // Smaller spheres for points (reduced from 0.15)
            
            this.data.forEach((point, index) => {
                // Skip invalid points
                if (!point || typeof point !== 'object') {
                    console.warn(`Skipping invalid point at index ${index} for mesh creation:`, point);
                    return;
                }
                
                // Safety check for embedding
                if (!point.embedding || !Array.isArray(point.embedding)) {
                    console.warn(`Skipping point ${index} - invalid embedding:`, point.embedding);
                    return; // Skip this point
                }
                
                // Ensure we have at least 3 dimensions
                if (point.embedding.length < 3) {
                    console.warn(`Point ${index} embedding has less than 3 dimensions, padding with zeros`);
                    while (point.embedding.length < 3) {
                        point.embedding.push(0);
                    }
                }
                
                // Get color for this point
                const cancerType = point.cancer_type || 'Unknown';
                const color = new THREE.Color(colorMap[cancerType] || '#999999');
                
                // Create material for this point
                const material = new THREE.MeshBasicMaterial({ 
                    color: color,
                    transparent: true,
                    opacity: 0.8
                });
                
                // Create mesh
                const pointMesh = new THREE.Mesh(sphereGeometry, material);
                
                // Set position
                const scale = 10;
                pointMesh.position.set(
                    (point.embedding[0] || 0) * scale,
                    (point.embedding[1] || 0) * scale,
                    (point.embedding[2] || 0) * scale
                );
                
                // Store reference to data for clicking
                pointMesh.userData = {
                    index: index,
                    pointData: point
                };
                
                // Add to group and clickable array
                this.pointsGroup.add(pointMesh);
                this.clickablePoints.push(pointMesh);
            });
            
            // Add the group to the scene
            this.scene.add(this.pointsGroup);
            
            console.log(`Created ${this.clickablePoints.length} clickable point meshes`);
            
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
        console.log('Creating legend with color map:', colorMap);
        
        // Find the legend content div in the template
        const legendContent = document.getElementById('legend-content');
        if (!legendContent) {
            console.error('Legend content div not found');
            return;
        }
        
        // Clear the loading message and populate with legend items
        legendContent.innerHTML = Object.entries(colorMap).map(([type, color]) => `
            <div class="legend-item" data-cancer-type="${type}">
                <span class="legend-color" style="background-color: ${color}"></span>
                <span>${type}</span>
            </div>
        `).join('');
        
        console.log('Legend populated successfully');
    }
    
    setupInteractivity() {
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        let hoveredPoint = null;
        
        // Add click event listener for closing tooltip when clicking outside
        document.addEventListener('click', (event) => {
            const tooltip = document.getElementById('vector-3d-tooltip');
            if (tooltip && tooltip.style.display !== 'none') {
                // Don't hide if clicking on the tooltip itself, any of its children, or the canvas
                if (!tooltip.contains(event.target) && 
                    !this.renderer.domElement.contains(event.target) &&
                    !event.target.classList.contains('find-similar-btn')) {
                    this.hideTooltip();
                }
            }
        });
        
        // Add mouse move for hover effects
        this.renderer.domElement.addEventListener('mousemove', (event) => {
            const rect = this.renderer.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
            
            raycaster.setFromCamera(mouse, this.camera);
            const intersects = raycaster.intersectObjects(this.clickablePoints);
            
            // Reset previous hovered point
            if (hoveredPoint) {
                hoveredPoint.material.opacity = 0.8;
                hoveredPoint = null;
                this.renderer.domElement.style.cursor = 'default';
            }
            
            // Set new hovered point
            if (intersects.length > 0) {
                hoveredPoint = intersects[0].object;
                hoveredPoint.material.opacity = 1.0;
                this.renderer.domElement.style.cursor = 'pointer';
            }
        });
        
        this.renderer.domElement.addEventListener('click', (event) => {
            const rect = this.renderer.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
            
            raycaster.setFromCamera(mouse, this.camera);
            
            // Raycast against individual point meshes
            const intersects = raycaster.intersectObjects(this.clickablePoints);
            
            if (intersects.length > 0) {
                const intersectedMesh = intersects[0].object;
                const pointData = intersectedMesh.userData.pointData;
                
                console.log('🎯 Point clicked:', pointData.id);
                
                // Hide any existing tooltip before showing a new one
                this.hideTooltip();
                
                // Show the new tooltip
                this.showPointInfo(pointData, event);
            } else {
                // Hide tooltip if clicking on empty space
                this.hideTooltip();
            }
        });
    }
    
    showPointInfo(pointData, event) {
        console.log('📋 Showing tooltip for point:', pointData.id);
        
        // Create or update tooltip
        let tooltip = document.getElementById('vector-3d-tooltip');
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.id = 'vector-3d-tooltip';
            tooltip.className = 'vector-3d-tooltip';
            document.body.appendChild(tooltip);
        }
        
        // Create unique button ID to avoid conflicts
        const buttonId = `find-similar-btn-${pointData.id.replace(/[^a-zA-Z0-9]/g, '-')}`;
        
        tooltip.innerHTML = `
            <div class="tooltip-header">
                <strong>Vector: ${pointData.id}</strong>
            </div>
            <div class="tooltip-content">
                <div><strong>Cancer Type:</strong> ${pointData.cancer_type}</div>
                <div><strong>Position:</strong> [${pointData.embedding.slice(0, 3).map(v => v.toFixed(3)).join(', ')}]</div>
                ${pointData.metadata ? `<div><strong>Metadata:</strong> ${JSON.stringify(pointData.metadata, null, 2)}</div>` : ''}
                <div class="tooltip-actions mt-2">
                    <button id="${buttonId}" class="btn btn-sm btn-primary find-similar-btn" data-vector-id="${pointData.id}">
                        Find Similar Vectors
                    </button>
                </div>
            </div>
        `;
        
        // Position tooltip more carefully
        const tooltipWidth = 300; // Approximate tooltip width
        const tooltipHeight = 150; // Approximate tooltip height
        const padding = 10;
        
        // Get viewport dimensions
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;
        
        console.log('🔍 Viewport info:', {
            viewportWidth, 
            viewportHeight, 
            scrollTop, 
            scrollLeft,
            eventPageX: event.pageX,
            eventPageY: event.pageY,
            eventClientX: event.clientX,
            eventClientY: event.clientY
        });
        
        // Use clientX/Y instead of pageX/Y to get viewport-relative coordinates
        let left = event.clientX + padding;
        let top = event.clientY + padding;
        
        // Adjust position if tooltip would go off-screen (viewport-relative)
        if (left + tooltipWidth > viewportWidth) {
            left = event.clientX - tooltipWidth - padding;
        }
        if (top + tooltipHeight > viewportHeight) {
            top = event.clientY - tooltipHeight - padding;
        }
        
        // Ensure tooltip stays within viewport bounds
        left = Math.max(padding, Math.min(left, viewportWidth - tooltipWidth - padding));
        top = Math.max(padding, Math.min(top, viewportHeight - tooltipHeight - padding));
        
        // Convert to page coordinates for positioning
        const finalLeft = left + scrollLeft;
        const finalTop = top + scrollTop;
        
        tooltip.style.display = 'block';
        tooltip.style.left = finalLeft + 'px';
        tooltip.style.top = finalTop + 'px';
        tooltip.style.position = 'absolute';
        tooltip.style.zIndex = '99999';
        
        // Force visibility check - if tooltip is still not visible, position it at a safe location
        setTimeout(() => {
            const tooltipRect = tooltip.getBoundingClientRect();
            const isVisible = tooltipRect.top >= 0 && 
                             tooltipRect.left >= 0 && 
                             tooltipRect.bottom <= viewportHeight && 
                             tooltipRect.right <= viewportWidth;
            
            console.log('🔍 Tooltip visibility check:', {
                isVisible,
                tooltipRect: {
                    top: tooltipRect.top,
                    left: tooltipRect.left,
                    bottom: tooltipRect.bottom,
                    right: tooltipRect.right
                },
                viewport: {viewportWidth, viewportHeight}
            });
            
            if (!isVisible) {
                console.log('⚠️ Tooltip not visible, repositioning to safe location');
                // Position at center-right of viewport as fallback
                const safeLeft = Math.max(50, viewportWidth - 350);  // 50px from right edge
                const safeTop = Math.max(50, viewportHeight / 2 - 75); // Center vertically
                
                tooltip.style.left = (safeLeft + scrollLeft) + 'px';
                tooltip.style.top = (safeTop + scrollTop) + 'px';
                
                console.log('✅ Tooltip repositioned to safe location:', {
                    safeLeft: safeLeft + scrollLeft,
                    safeTop: safeTop + scrollTop
                });
            }
        }, 10);
        
        // Store tooltip position for debugging
        tooltip.dataset.left = finalLeft;
        tooltip.dataset.top = finalTop;
        tooltip.dataset.vectorId = pointData.id;
        
        // Debug: Check if button was created properly
        const button = tooltip.querySelector('.find-similar-btn');
        console.log('🔘 Button created:', button);
        console.log('🔘 Button classes:', button ? button.className : 'no button found');
        console.log('🔘 Button data-vector-id:', button ? button.getAttribute('data-vector-id') : 'no button found');
        console.log('🔘 Tooltip positioned at:', {
            finalLeft, 
            finalTop, 
            width: tooltipWidth, 
            height: tooltipHeight,
            viewportRelative: {left, top},
            willBeVisible: finalTop >= scrollTop && finalTop <= scrollTop + viewportHeight
        });
        
        // Add direct event listener to the button to ensure it works
        if (button) {
            // Remove any existing listeners to avoid duplicates
            button.removeEventListener('click', this.handleButtonClick);
            
            // Add direct click handler
            this.handleButtonClick = (e) => {
                console.log('🎯 Direct button click handler triggered!');
                e.preventDefault();
                e.stopPropagation();
                e.stopImmediatePropagation();
                
                const vectorId = button.getAttribute('data-vector-id');
                console.log('Direct handler - Vector ID:', vectorId);
                
                if (vectorId && this.findSimilarVectors) {
                    console.log('Calling findSimilarVectors directly...');
                    this.findSimilarVectors(vectorId);
                    this.hideTooltip();
                } else {
                    console.error('Missing vectorId or findSimilarVectors method');
                }
                
                return false;
            };
            
            button.addEventListener('click', this.handleButtonClick, true);
            console.log('✅ Direct button event listener added');
        }
        
        // Re-enable canvas pointer events immediately after tooltip is positioned
        // This allows users to click on other points without first hiding the tooltip
        setTimeout(() => {
            if (this.renderer && this.renderer.domElement) {
                this.renderer.domElement.style.pointerEvents = 'auto';
                console.log('🎯 Re-enabled canvas pointer events for continued interaction');
            }
        }, 100);
        
        // Clear any existing timeout
        if (this.tooltipTimeout) {
            clearTimeout(this.tooltipTimeout);
        }
        
        // Auto-hide tooltip after 15 seconds
        this.tooltipTimeout = setTimeout(() => {
            this.hideTooltip();
        }, 15000);
    }
    
    hideTooltip() {
        const tooltip = document.getElementById('vector-3d-tooltip');
        if (tooltip) {
            tooltip.style.display = 'none';
        }
        if (this.tooltipTimeout) {
            clearTimeout(this.tooltipTimeout);
            this.tooltipTimeout = null;
        }
        
        // Ensure canvas pointer events are always enabled when tooltip is hidden
        if (this.renderer && this.renderer.domElement) {
            this.renderer.domElement.style.pointerEvents = 'auto';
            console.log('🎯 Re-enabled canvas pointer events after hiding tooltip');
        }
    }
    
    async findSimilarVectors(vectorId) {
        console.log('Finding similar vectors for:', vectorId);
        
        try {
            // Show loading state in search results
            this.updateSearchResults({
                success: false,
                loading: true,
                message: `Finding vectors similar to ${vectorId}...`
            });
            
            // Call the nearest-neighbors API
            const response = await fetch(`/api/nearest-neighbors/${encodeURIComponent(vectorId)}`);
            const data = await response.json();
            
            if (data.success) {
                console.log('Found similar vectors:', data.results);
                this.displaySimilarityResults(data);
            } else {
                console.error('Error finding similar vectors:', data.error);
                this.updateSearchResults({
                    success: false,
                    error: data.error
                });
            }
        } catch (error) {
            console.error('Network error finding similar vectors:', error);
            this.updateSearchResults({
                success: false,
                error: 'Network error: ' + error.message
            });
        }
    }
    
    displaySimilarityResults(data) {
        // Filter out results with distance 0 (exact matches) to find actual similar vectors
        const similarVectors = data.results.filter(result => result.distance > 0);
        
        if (similarVectors.length === 0) {
            // No similar vectors found (only exact matches)
            this.updateSearchResults({
                success: false,
                error: `No similar vectors found for "${data.query_vector_id}". Only exact matches were found.`
            });
        } else {
            // Update the search results section with similar vectors
            this.updateSearchResults({
                success: true,
                query_vector_id: data.query_vector_id,
                results: similarVectors
            });
        }
    }
    
    updateSearchResults(data) {
        // Find the new similar vectors section elements
        const section = document.getElementById('similar-vectors-section');
        const loading = document.getElementById('similar-vectors-loading');
        const results = document.getElementById('similar-vectors-results');  
        const error = document.getElementById('similar-vectors-error');
        const queryInfo = document.getElementById('similar-vectors-query-info');
        
        if (!section || !loading || !results || !error || !queryInfo) {
            console.warn('Similar vectors section elements not found');
            return;
        }
        
        // Show the section
        section.style.display = 'block';
        
        // Hide all states initially using CSS classes
        loading.style.display = 'none';
        // Don't override Bootstrap's .row display - let CSS handle it
        results.style.display = '';
        // Ensure error div is hidden using CSS class
        error.classList.remove('show-error');
        error.classList.add('d-none');
        
        if (data.loading) {
            // Show loading state
            loading.style.display = 'block';
            results.style.display = 'none';
            queryInfo.textContent = data.message || 'Finding similar vectors...';
            results.innerHTML = '';
            return;
        }
        
        if (!data.success) {
            // Show error state using CSS class to override Bootstrap
            error.classList.add('show-error');
            error.classList.remove('d-none');
            error.querySelector('.error-message').textContent = data.error || 'Unknown error occurred';
            results.innerHTML = '';
            queryInfo.textContent = 'Error occurred during search';
            return;
        }
        
        // Show success state with results
        loading.style.display = 'none';
        // Hide error using CSS class removal
        error.classList.remove('show-error');
        error.classList.add('d-none');
        
        queryInfo.textContent = `Found ${data.results.length} vectors similar to "${data.query_vector_id}"`;
        
        // Clear previous results
        results.innerHTML = '';
        
        // Display each similar vector as a card
        data.results.forEach((result, index) => {
            const colDiv = document.createElement('div');
            // Bootstrap classes for 4 columns per row (col-lg-3 col-md-4 col-sm-6)
            // lg: 4 cards (12/4 = 3), md: 3 cards, sm: 2 cards
            colDiv.className = 'col-lg-3 col-md-4 col-sm-6 mb-3';
            colDiv.classList.add('animate-fade-in');
            colDiv.style.animationDelay = `${index * 0.1}s`;
            
            // Format metadata display
            let metadataDisplay = 'No metadata';
            if (result.metadata) {
                const metadataEntries = Object.entries(result.metadata);
                if (metadataEntries.length > 0) {
                    metadataDisplay = metadataEntries.map(([key, value]) => 
                        `<strong>${key}:</strong> ${value}`
                    ).join('<br>');
                }
            }
            
            colDiv.innerHTML = `
                <div class="card h-100 border-0 shadow-sm">
                    <div class="card-header bg-transparent border-0 pb-2">
                        <div class="d-flex justify-content-between align-items-start">
                            <h6 class="mb-0 text-primary fw-semibold">
                                <i class="fas fa-vector-square me-1"></i>
                                ${result.id}
                            </h6>
                            <small class="badge bg-secondary">
                                Rank #${index + 1}
                            </small>
                        </div>
                    </div>
                    <div class="card-body pt-0">
                        <div class="mb-2">
                            <strong class="text-muted small">Distance:</strong>
                            <span class="badge bg-info ms-1">${result.distance.toFixed(4)}</span>
                        </div>
                        
                        <div class="mb-2">
                            <strong class="text-muted small">Metadata:</strong>
                            <div class="small mt-1">${metadataDisplay}</div>
                        </div>
                        
                        ${result.document ? `
                            <div class="mb-0">
                                <strong class="text-muted small">Document:</strong>
                                <div class="small mt-1 text-truncate" title="${result.document}">
                                    ${result.document.length > 100 ? result.document.substring(0, 100) + '...' : result.document}
                                </div>
                            </div>
                        ` : ''}
                    </div>
                </div>
            `;
            
            results.appendChild(colDiv);
        });
        
        // Scroll to results section smoothly
        setTimeout(() => {
            section.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
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
        
        // Force the container to have the correct dimensions
        const containerStyles = window.getComputedStyle(this.container);
        let width = this.container.clientWidth || parseInt(containerStyles.width) || 800;
        let height = this.container.clientHeight || parseInt(containerStyles.height) || 600;
        
        // For fullscreen 3D page, ensure we use viewport dimensions
        const isFullscreen = this.container.classList.contains('vector-3d-container') && 
                            this.container.parentElement.classList.contains('vector-3d-fullscreen');
        
        if (isFullscreen) {
            // Calculate 95vh for fullscreen mode
            height = Math.floor(window.innerHeight * 0.95);
            width = Math.max(width, window.innerWidth - 40); // Account for margins
        }
        
        console.log(`Window resize - updating to dimensions: ${width}x${height}`);
        
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

// Initialize the 3D visualization when DOM is loaded

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

// Global event delegation for find-similar-btn clicks
// This ensures the button is always clickable regardless of when/how the tooltip is created
document.body.addEventListener('click', function(e) {
    console.log('🔍 Document body click detected:', e.target.tagName, e.target.className);
    console.log('🔍 Click coordinates:', {clientX: e.clientX, clientY: e.clientY, pageX: e.pageX, pageY: e.pageY});
    console.log('🔍 Event target details:', {
        tagName: e.target.tagName,
        className: e.target.className,
        id: e.target.id,
        classList: Array.from(e.target.classList || []),
        hasClass: e.target.classList?.contains('find-similar-btn')
    });
    
    // Check current tooltip position for debugging
    const tooltip = document.getElementById('vector-3d-tooltip');
    if (tooltip && tooltip.style.display !== 'none') {
        const tooltipRect = tooltip.getBoundingClientRect();
        console.log('🔍 Tooltip bounds:', {
            left: tooltipRect.left,
            top: tooltipRect.top,
            right: tooltipRect.right,
            bottom: tooltipRect.bottom,
            width: tooltipRect.width,
            height: tooltipRect.height
        });
        console.log('🔍 Click within tooltip bounds:', 
            e.clientX >= tooltipRect.left && e.clientX <= tooltipRect.right &&
            e.clientY >= tooltipRect.top && e.clientY <= tooltipRect.bottom
        );
    }
    
    // Check if the clicked element is the button or if we need to look for it in the event path
    let targetButton = null;
    
    // First check if we clicked directly on the button
    if (e.target.classList && e.target.classList.contains('find-similar-btn')) {
        targetButton = e.target;
        console.log('🎯 Direct button click detected');
    } else {
        // Check if the click was on a child element of the button
        targetButton = e.target.closest('.find-similar-btn');
        if (targetButton) {
            console.log('🎯 Button click detected via closest()');
        } else {
            // Check all elements under the cursor position to find the button
            const elementsUnderCursor = document.elementsFromPoint(e.clientX, e.clientY);
            console.log('🔍 Elements under cursor:', elementsUnderCursor.map(el => el.tagName + '.' + (el.className || '')));
            
            for (let element of elementsUnderCursor) {
                if (element.classList && element.classList.contains('find-similar-btn')) {
                    targetButton = element;
                    console.log('🎯 Button found under cursor position');
                    break;
                }
            }
            
            // If still not found, check if we're clicking within the tooltip and look for the button there
            if (!targetButton && tooltip && tooltip.style.display !== 'none') {
                const tooltipRect = tooltip.getBoundingClientRect();
                const isWithinTooltip = e.clientX >= tooltipRect.left && e.clientX <= tooltipRect.right &&
                                      e.clientY >= tooltipRect.top && e.clientY <= tooltipRect.bottom;
                
                if (isWithinTooltip) {
                    const buttonInTooltip = tooltip.querySelector('.find-similar-btn');
                    if (buttonInTooltip) {
                        targetButton = buttonInTooltip;
                        console.log('🎯 Button found within tooltip bounds');
                    }
                }
            }
        }
    }
    
    if (targetButton) {
        console.log('🎯 Find similar button clicked via global delegation');
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        
        const vectorId = targetButton.getAttribute('data-vector-id');
        console.log('Vector ID:', vectorId);
        
        // Get the visualizer instance and call findSimilarVectors
        if (window.vector3DVisualizer && typeof window.vector3DVisualizer.findSimilarVectors === 'function') {
            console.log('Calling findSimilarVectors...');
            window.vector3DVisualizer.findSimilarVectors(vectorId);
            
            // Hide the tooltip
            const tooltip = document.getElementById('vector-3d-tooltip');
            if (tooltip) {
                tooltip.style.display = 'none';
            }
        } else {
            console.error('Vector3DVisualizer instance not found or findSimilarVectors method not available');
            console.log('window.vector3DVisualizer:', window.vector3DVisualizer);
        }
        
        return false;
    }
}, true); // Use capture phase to ensure it fires before other handlers

console.log('✅ Global event delegation for find-similar-btn has been set up');
