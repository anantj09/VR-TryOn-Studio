import * as THREE from 'three';
import { modelContainer, onWindowResize } from './scene-setup.js';
import { loadModel } from './model-loader.js';
import { broadcastState } from './ws-client.js';

// DOM Elements
export const uploadContainer = document.getElementById('upload-container');
export const uploadBox = document.getElementById('upload-box');
export const fileInput = document.getElementById('file-input');
export const instructionsPanel = document.getElementById('instructions-panel');
export const measurementsPanel = document.getElementById('measurements-panel');
export const statusDot = document.getElementById('status-dot');
export const statusText = document.getElementById('status-text');
export const modelInfoText = document.getElementById('model-info-text');

const valHeight = document.getElementById('val-height');
const valChest = document.getElementById('val-chest');
const valWaist = document.getElementById('val-waist');
const valHip = document.getElementById('val-hip');

export let currentModel = null;
export let currentModelUrl = null;
export let currentModelMeasurements = null;

// Helper to update status indicators
export function updateStatus(text, stateClass) {
    if (statusText) statusText.textContent = text;
    if (statusDot) statusDot.className = "status-dot " + stateClass;
}

// Clean up existing model
export function removeCurrentModel() {
    broadcastState({ type: 'unload' });
    currentModelUrl = null;
    currentModelMeasurements = null;
    if (currentModel) {
        modelContainer.remove(currentModel);
        currentModel.traverse((child) => {
            if (child.isMesh) {
                child.geometry.dispose();
                if (Array.isArray(child.material)) {
                    child.material.forEach(mat => mat.dispose());
                } else if (child.material) {
                    child.material.dispose();
                }
            }
        });
        currentModel = null;
        modelInfoText.textContent = "No model active";
        
        // Reset status indicator to standard ready state
        if ('xr' in navigator) {
            navigator.xr.isSessionSupported('immersive-vr').then((supported) => {
                if (supported) {
                    updateStatus("Disconnected (Ready for VR)", "active");
                } else {
                    updateStatus("System Ready (Flat Mode)", "active");
                }
            }).catch(() => {
                updateStatus("System Ready (Flat Mode)", "active");
            });
        } else {
            updateStatus("System Ready (Flat Mode)", "active");
        }
    }
}

// Measurements UI panel handlers
export function showMeasurementsPanel(measurements) {
    if (!measurementsPanel) return;
    if (valHeight) valHeight.textContent = `${Math.round(measurements.heightCm)}cm`;
    if (valChest) valChest.textContent = `${Math.round(measurements.chestCm)}cm`;
    if (valWaist) valWaist.textContent = `${Math.round(measurements.waistCm)}cm`;
    if (valHip) valHip.textContent = `${Math.round(measurements.hipCm)}cm`;
    measurementsPanel.style.display = 'block';
}

export function hideMeasurementsPanel() {
    if (measurementsPanel) {
        measurementsPanel.style.display = 'none';
    }
}

// Model File Handler (now uploads photo to backend)
export function handleModelFile(file) {
    if (!file) return;

    // Validate that the file is an image (allow fallback to extension check if mime type is missing/generic on mobile)
    const isImage = file.type.startsWith('image/') || /\.(jpe?g|png|gif|webp|heic|heif)$/i.test(file.name);
    if (!isImage) {
        alert("Please upload a valid image file (PNG or JPEG).");
        return;
    }

    updateStatus("Uploading photo... 0%", "loading");
    removeCurrentModel();
    hideMeasurementsPanel();

    // Hide upload box while showing the scene
    uploadContainer.style.display = 'none';

    // Prepare Multipart form upload
    const formData = new FormData();
    formData.append('photo', file);

    const backendHost = window.location.hostname;
    const backendUrl = `http://${backendHost}:8000/api/v1/generate-mesh`;

    const xhr = new XMLHttpRequest();
    xhr.open('POST', backendUrl);

    // Send the Colab tunnel URL if configured
    const colabUrl = localStorage.getItem('colab_tunnel_url') || '';
    if (colabUrl) {
        xhr.setRequestHeader('X-Colab-Tunnel-URL', colabUrl);
        console.log(`Routing visual 3D generation to Colab via tunnel: ${colabUrl}`);
    }

    // Track upload progress
    xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
            const percent = Math.round((e.loaded / e.total) * 100);
            updateStatus(`Uploading photo... ${percent}%`, "loading");
        }
    });

    // Handle completed request
    xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
            try {
                const data = JSON.parse(xhr.responseText);
                const meshUrl = data.meshUrl;
                const measurements = data.measurements;
                
                // Resolve absolute URL dynamically based on host
                const absoluteMeshUrl = meshUrl.startsWith('http') ? meshUrl : `http://${backendHost}:8000${meshUrl}`;
                
                updateStatus("Downloading generated 3D twin...", "loading");

                loadModel(
                    absoluteMeshUrl,
                    modelContainer,
                    (model) => {
                        currentModel = model;
                        currentModelUrl = absoluteMeshUrl;
                        currentModelMeasurements = measurements;
                        updateStatus("Model Rendered (1.8m Aligned)", "active");
                        
                        broadcastState({
                            type: 'load',
                            url: absoluteMeshUrl,
                            measurements: measurements
                        });
                        
                        // Show measurements HUD
                        showMeasurementsPanel(measurements);

                        // Update model info text
                        modelInfoText.innerHTML = `
                            Model: Generated Digital Twin<br>
                            Source Photo: ${file.name}<br>
                            Polygons: ${countPolygons(model).toLocaleString()} faces
                        `;
                        
                        showLoadAnotherButton();
                    },
                    (progress) => {
                        if (progress.total > 0) {
                            const percent = Math.round((progress.loaded / progress.total) * 100);
                            updateStatus(`Downloading model... ${percent}%`, "loading");
                        }
                    },
                    (error) => {
                        console.error("Error loading generated model:", error);
                        updateStatus("Load Failed", "loading");
                        alert("Failed to render the generated 3D model. Verify the mesh file was compiled correctly.");
                        uploadContainer.style.display = 'flex';
                    }
                );
            } catch (err) {
                console.error("Failed to parse backend response:", err);
                triggerFallback("Failed to compile server response.");
            }
        } else {
            console.warn(`Server error (${xhr.status}):`, xhr.responseText);
            triggerFallback(`Server error (${xhr.status})`);
        }
    });

    // Handle connection failures
    xhr.addEventListener('error', () => {
        console.warn("XHR Connection failed, falling back.");
        triggerFallback("Network connection error");
    });

    xhr.addEventListener('timeout', () => {
        console.warn("XHR Upload timed out.");
        triggerFallback("Connection timed out");
    });

    // Helper to run fallback
    function triggerFallback(reason) {
        console.warn(`Reconstruction failed (${reason}), loading local demo model...`);
        updateStatus("Connection failed. Loading demo mesh...", "loading");
        
        const sampleUrl = './data/demo/9dc6216c-3ba3-403b-9fc8-1783c828b8c4_mesh.gltf';
        loadModel(
            sampleUrl,
            modelContainer,
            (model) => {
                currentModel = model;
                updateStatus("Model Loaded (Backend Offline)", "active");
                
                const mockMeasurements = {
                    heightCm: 172.5,
                    chestCm: 92.0,
                    waistCm: 80.0,
                    hipCm: 95.0
                };

                const absoluteDemoUrl = new URL(sampleUrl, window.location.href).href;
                currentModelUrl = absoluteDemoUrl;
                currentModelMeasurements = mockMeasurements;
                
                broadcastState({
                    type: 'load',
                    url: absoluteDemoUrl,
                    measurements: mockMeasurements
                });
                
                showMeasurementsPanel(mockMeasurements);

                modelInfoText.innerHTML = `
                    File: 9dc6216c_mesh.gltf (Demo)<br>
                    Note: Backend offline (${reason}).<br>
                    Polygons: ${countPolygons(model).toLocaleString()} faces
                `;
                
                showLoadAnotherButton();
            },
            null,
            (err) => {
                console.error("Demo fallback failed:", err);
                updateStatus("Load Failed", "loading");
                alert("Connection to backend failed and demo fallback could not be loaded.");
                uploadContainer.style.display = 'flex';
            }
        );
    }

    // Set status to processing once upload completes
    xhr.upload.addEventListener('load', () => {
        updateStatus("Running 3D AI Reconstruction...", "loading");
    });

    xhr.send(formData);
}

// Sample Model Loader
export function handleSampleModel() {
    updateStatus("Loading Demo Model...", "loading");
    removeCurrentModel();
    hideMeasurementsPanel();
    uploadContainer.style.display = 'none';

    const sampleUrl = './data/demo/9dc6216c-3ba3-403b-9fc8-1783c828b8c4_mesh.gltf';

    loadModel(
        sampleUrl,
        modelContainer,
        (model) => {
            currentModel = model;
            updateStatus("Model Rendered (1.8m Aligned)", "active");
            
            // Show measurements HUD with sample mock measurements
            const mockMeasurements = {
                heightCm: 172.5,
                chestCm: 92.0,
                waistCm: 80.0,
                hipCm: 95.0
            };

            const absoluteDemoUrl = new URL(sampleUrl, window.location.href).href;
            currentModelUrl = absoluteDemoUrl;
            currentModelMeasurements = mockMeasurements;
            
            broadcastState({
                type: 'load',
                url: absoluteDemoUrl,
                measurements: mockMeasurements
            });
            
            showMeasurementsPanel(mockMeasurements);

            modelInfoText.innerHTML = `
                File: 9dc6216c_mesh.gltf (Demo)<br>
                Render Scale: Y=1.8m (Scaled)<br>
                Polygons: ${countPolygons(model).toLocaleString()} faces
            `;
            
            showLoadAnotherButton();
        },
        (progress) => {
            if (progress.total > 0) {
                const percent = Math.round((progress.loaded / progress.total) * 100);
                updateStatus(`Loading... ${percent}%`, "loading");
            }
        },
        (error) => {
            console.error("Error loading sample model:", error);
            updateStatus("Load Failed", "loading");
            alert("Failed to load demo mesh. Check console logs.");
            uploadContainer.style.display = 'flex';
        }
    );
}

// Helper to count faces/polygons in model
function countPolygons(object) {
    let count = 0;
    object.traverse((child) => {
        if (child.isMesh && child.geometry.index) {
            count += child.geometry.index.count / 3;
        } else if (child.isMesh && child.geometry.attributes.position) {
            count += child.geometry.attributes.position.count / 3;
        }
    });
    return count;
}

// Floating HUD "Load Another" button controller
export function showLoadAnotherButton() {
    let btn = document.getElementById('load-another-btn');
    if (!btn) {
        btn = document.createElement('button');
        btn.id = 'load-another-btn';
        btn.className = 'btn btn-secondary interactive';
        btn.textContent = 'Upload New Twin';
        btn.style.marginTop = '12px';
        btn.addEventListener('click', () => {
            removeCurrentModel();
            hideMeasurementsPanel();
            uploadContainer.style.display = 'flex';
            btn.style.display = 'none';
            // Reset container scale/rotation
            modelContainer.rotation.set(0, 0, 0);
            modelContainer.scale.set(1, 1, 1);

            // Force layout and scale correction on new upload trigger
            window.scrollTo(0, 0);
            document.body.scrollTop = 0;
            document.body.scrollLeft = 0;
            document.documentElement.scrollTop = 0;
            document.documentElement.scrollLeft = 0;
            onWindowResize();
        });
        document.querySelector('.status-panel').appendChild(btn);
    }
    btn.style.display = 'inline-flex';
}

// Event Listeners for UI interaction
export function setupUI() {
    // Load sample model
    const loadSampleBtn = document.getElementById('load-sample-btn');
    if (loadSampleBtn) {
        loadSampleBtn.addEventListener('click', (e) => {
            e.preventDefault();  // prevent label activation on mobile
            e.stopPropagation(); // prevent triggering fileInput click
            handleSampleModel();
        });
    }
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleModelFile(e.target.files[0]);
            // Clear input value so that the user can select the same file again if needed
            fileInput.value = '';
        }
    });

    // Drag and drop support
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.classList.add('drag-over');
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.classList.remove('drag-over');
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleModelFile(e.dataTransfer.files[0]);
        }
    });

    // Collapsible Navigation Controls Panel
    if (instructionsPanel) {
        const instructionsList = instructionsPanel.querySelector('ul');
        const toggleIcon = instructionsPanel.querySelector('.toggle-icon');

        instructionsPanel.addEventListener('click', () => {
            const isCollapsed = instructionsPanel.classList.contains('collapsed');
            if (isCollapsed) {
                instructionsPanel.classList.remove('collapsed');
                instructionsList.style.display = 'block';
                toggleIcon.textContent = '▲';
            } else {
                instructionsPanel.classList.add('collapsed');
                instructionsList.style.display = 'none';
                toggleIcon.textContent = '▼';
            }
        });
    }

    // Settings Panel Controls
    const settingsBtn = document.getElementById('settings-btn');
    const settingsPanel = document.getElementById('settings-panel');
    const settingsCloseBtn = document.getElementById('settings-close-btn');
    const colabUrlInput = document.getElementById('colab-url-input');

    if (settingsBtn && settingsPanel) {
        settingsBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const isHidden = settingsPanel.style.display === 'none';
            settingsPanel.style.display = isHidden ? 'block' : 'none';
        });
    }

    if (settingsCloseBtn && settingsPanel) {
        settingsCloseBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            settingsPanel.style.display = 'none';
        });
    }

    if (colabUrlInput) {
        // Load cached URL on startup
        const cachedUrl = localStorage.getItem('colab_tunnel_url');
        if (cachedUrl) {
            colabUrlInput.value = cachedUrl;
        }

        colabUrlInput.addEventListener('input', () => {
            localStorage.setItem('colab_tunnel_url', colabUrlInput.value.trim());
        });
    }

    // Close settings panel when clicking outside of it
    document.addEventListener('click', (e) => {
        if (settingsPanel && settingsPanel.style.display === 'block') {
            if (!settingsPanel.contains(e.target) && e.target !== settingsBtn && !settingsBtn.contains(e.target)) {
                settingsPanel.style.display = 'none';
            }
        }
    });

    // Portal Overlay Card Triggers
    const portalOverlay = document.getElementById('portal-overlay');
    const portalCardTwin = document.getElementById('portal-card-twin');
    const portalCardPremade = document.getElementById('portal-card-premade');
    const backPortalBtn = document.getElementById('back-portal-btn');
    const premadePanel = document.getElementById('premade-assets-panel');
    const logoTitle = document.getElementById('logo-title');
    const logoSubtitle = document.getElementById('logo-subtitle');

    if (portalCardTwin && portalOverlay) {
        portalCardTwin.addEventListener('click', () => {
            portalOverlay.style.opacity = '0';
            setTimeout(() => {
                portalOverlay.style.visibility = 'hidden';
            }, 400);
            uploadContainer.style.display = 'flex';
            if (premadePanel) premadePanel.style.display = 'none';
            if (backPortalBtn) backPortalBtn.style.display = 'block';
            if (logoTitle) logoTitle.textContent = "4D-HUMANS VR";
            if (logoSubtitle) logoSubtitle.textContent = "Immersive Mesh Inspector";
            if (settingsBtn) settingsBtn.style.display = 'inline-flex';
            updateStatus("System Ready (Twin Mode)", "active");
        });
    }

    if (portalCardPremade && portalOverlay) {
        portalCardPremade.addEventListener('click', () => {
            portalOverlay.style.opacity = '0';
            setTimeout(() => {
                portalOverlay.style.visibility = 'hidden';
            }, 400);
            uploadContainer.style.display = 'none';
            if (premadePanel) {
                premadePanel.style.display = 'flex';
                // Reset search box before loading
                const searchInput = document.getElementById('premade-search');
                if (searchInput) searchInput.value = '';
                loadPremadeAssetsList();
            }
            if (backPortalBtn) backPortalBtn.style.display = 'block';
            if (logoTitle) logoTitle.textContent = "PREMADE ASSETS";
            if (logoSubtitle) logoSubtitle.textContent = "Object Showcase Catalog";
            if (settingsBtn) settingsBtn.style.display = 'none';
            if (settingsPanel) settingsPanel.style.display = 'none';
            updateStatus("System Ready (Showcase Mode)", "active");
        });
    }

    if (backPortalBtn && portalOverlay) {
        backPortalBtn.addEventListener('click', () => {
            removeCurrentModel();
            hideMeasurementsPanel();
            
            // Hide the load another button if it exists
            const loadAnotherBtn = document.getElementById('load-another-btn');
            if (loadAnotherBtn) loadAnotherBtn.style.display = 'none';
            
            portalOverlay.style.visibility = 'visible';
            portalOverlay.style.opacity = '1';
            
            uploadContainer.style.display = 'none';
            if (premadePanel) premadePanel.style.display = 'none';
            backPortalBtn.style.display = 'none';
            if (logoTitle) logoTitle.textContent = "4D-HUMANS VR";
            if (logoSubtitle) logoSubtitle.textContent = "Immersive Mesh Inspector";
            if (settingsBtn) settingsBtn.style.display = 'none';
            if (settingsPanel) settingsPanel.style.display = 'none';
            updateStatus("Selecting Mode...", "active");
        });
    }

    // Active Search Filtering
    const searchInput = document.getElementById('premade-search');
    if (searchInput) {
        searchInput.addEventListener('input', () => {
            const query = searchInput.value.toLowerCase().trim();
            const queryTerms = query.split(/\s+/).filter(t => t !== '');
            
            if (queryTerms.length === 0) {
                renderAssets(allPremadeAssets);
                return;
            }
            
            const filtered = allPremadeAssets.filter(asset => {
                const parsedName = formatAssetName(asset.filename).toLowerCase();
                const displayName = (asset.customName || parsedName.split(' > ').pop() || asset.filename).toLowerCase();
                const category = (asset.customCategory || parsedName.substring(0, parsedName.lastIndexOf(' > ')) || "General").toLowerCase();
                
                // Card matches query if all typed search terms are present in name or category
                return queryTerms.every(term => 
                    displayName.includes(term) || 
                    category.includes(term) || 
                    asset.filename.toLowerCase().includes(term)
                );
            });
            renderAssets(filtered);
        });
    }
}

// Variable to store loaded premade catalog list for filtering
let allPremadeAssets = [];

export function formatAssetName(filename) {
    let name = filename.replace(/\.(glb|gltf)$/i, '');
    // Strip standard metadata and random number suffixes
    name = name.replace(/_(sc|fi|sa|ai|ta|ut|am|go|ro|di|co|pl|de|la|se|ho|pr|ba|cl|ae|cr|te|to|ag|fu|ac|of|so|pa|or|pu|su|ma)\d+/gi, '');
    name = name.replace(/_\d+/g, '');
    if (name.startsWith('home_')) name = name.substring(5);
    
    // Capitalize and format path segments
    return name.split('_').map(word => {
        if (!word) return '';
        if (word.length <= 4) return word.toUpperCase();
        return word.charAt(0).toUpperCase() + word.slice(1);
    }).filter(word => word !== '').join(' > ');
}

export function loadPremadeAssetsList() {
    const listContainer = document.getElementById('premade-list');
    const badge = document.getElementById('asset-count-badge');
    if (!listContainer) return;
    
    listContainer.innerHTML = '<div style="color: var(--text-muted); font-size: 12px; padding: 12px;">Loading catalog...</div>';
    
    const backendHost = window.location.hostname;
    const backendUrl = `http://${backendHost}:8000/api/v1/premade/list`;
    
    fetch(backendUrl)
        .then(response => response.json())
        .then(data => {
            allPremadeAssets = data;
            if (badge) badge.textContent = `${data.length} Assets`;
            renderAssets(data);
        })
        .catch(err => {
            console.error("Failed to load premade assets list:", err);
            listContainer.innerHTML = '<div style="color: #ff0055; font-size: 12px; padding: 12px;">Failed to load catalog. Verify backend is running.</div>';
        });
}

function renderAssets(assets) {
    const listContainer = document.getElementById('premade-list');
    if (!listContainer) return;
    
    listContainer.innerHTML = '';
    
    if (assets.length === 0) {
        listContainer.innerHTML = '<div style="color: var(--text-muted); font-size: 12px; padding: 12px;">No assets found.</div>';
        return;
    }
    
    const backendHost = window.location.hostname;
    
    assets.forEach(asset => {
        const card = document.createElement('div');
        card.className = 'premade-card interactive';
        card.setAttribute('role', 'button');
        card.setAttribute('tabindex', '0');
        
        // Retrieve display elements (fall back to dynamic parser if custom fields are missing)
        const parsedName = formatAssetName(asset.filename);
        const displayName = asset.customName || parsedName.split(' > ').pop() || asset.filename;
        const category = asset.customCategory || parsedName.substring(0, parsedName.lastIndexOf(' > ')) || "General";
        
        const sizeMb = (asset.sizeBytes / (1024 * 1024)).toFixed(2);
        
        card.innerHTML = `
            <div class="premade-card-category">${category}</div>
            <div class="premade-card-title">${displayName}</div>
            <div class="premade-card-meta">
                <span>${sizeMb} MB</span>
                <span class="badge-ready" style="color: var(--success-color); font-size: 8px;">● READY</span>
            </div>
        `;
        
        let absoluteUrl = `http://${backendHost}:8000/static/premade/${asset.filename}`;
        if (asset.customScale !== null && asset.customScale !== undefined) {
            absoluteUrl += `?scale=${asset.customScale}`;
        }
        card.addEventListener('click', () => {
            const activeCards = listContainer.querySelectorAll('.premade-card');
            activeCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            loadPremadeAsset(absoluteUrl, asset.filename, displayName, category);
        });
        
        listContainer.appendChild(card);
    });
}

export function loadPremadeAsset(url, filename, displayName, category) {
    updateStatus("Downloading asset...", "loading");
    removeCurrentModel();
    hideMeasurementsPanel();
    
    loadModel(
        url,
        modelContainer,
        (model) => {
            currentModel = model;
            currentModelUrl = url;
            currentModelMeasurements = null; // No physical sizing measurements for premade models
            
            // Broadcast state to spectator view
            broadcastState({
                type: 'load',
                url: url,
                measurements: null
            });
            
            updateStatus("Model Rendered (Aligned)", "active");
            
            // Compute dimensions of the loaded/scaled model
            const box = new THREE.Box3().setFromObject(model);
            const size = new THREE.Vector3();
            box.getSize(size);
            
            // Update HUD info bar with model details and dimensions
            modelInfoText.innerHTML = `
                Asset: ${displayName}<br>
                Size: ${size.x.toFixed(2)}m x ${size.y.toFixed(2)}m x ${size.z.toFixed(2)}m<br>
                Polygons: ${countPolygons(model).toLocaleString()} faces
            `;
        },
        (progress) => {
            if (progress.total > 0) {
                const percent = Math.round((progress.loaded / progress.total) * 100);
                updateStatus(`Downloading... ${percent}%`, "loading");
            }
        },
        (error) => {
            console.error("Error loading premade asset:", error);
            updateStatus("Load Failed", "loading");
            alert("Failed to render the 3D model. Verify the file path and format.");
        }
    );
}

