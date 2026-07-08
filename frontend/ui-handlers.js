import * as THREE from 'three';
import { modelContainer, onWindowResize, updateSceneTheme } from './scene-setup.js';
import { loadModel } from './model-loader.js';
import { broadcastState } from './ws-client.js';

// DOM Elements
export const uploadContainer = document.getElementById('upload-container');
export const portraitSelectorWrapper = document.getElementById('portrait-selector-wrapper');
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

// Virtual try-on selection state variables
export let selectedGarmentId = null;
export let selectedGarmentData = null;
let catalogItems = [];
let currentCatalogPage = 1;
const itemsPerPage = 20;
let tempSelectedGarmentId = null;
let tempSelectedGarmentData = null;

let currentUploadedFile = null;
let lastTryonId = null;
let lastTryonImageUrl = null;
export let isTryonRunning = false;

export function checkGenerateButtonsState() {
    const tryonGenerate4dBtn = document.getElementById('tryon-generate-4d-btn');
    const tryonGenerateLhmBtn = document.getElementById('tryon-generate-lhm-btn');
    const hasSelected = !!lastTryonId;
    
    [tryonGenerate4dBtn, tryonGenerateLhmBtn].forEach(btn => {
        if (btn) {
            btn.disabled = !hasSelected;
            if (!hasSelected) {
                btn.style.opacity = '0.5';
                btn.style.cursor = 'not-allowed';
                btn.style.pointerEvents = 'none';
            } else {
                btn.style.opacity = '1';
                btn.style.cursor = 'pointer';
                btn.style.pointerEvents = 'auto';
            }
        }
    });
}

function updateInteractiveElementsState(running) {
    isTryonRunning = running;
    const changeGarmentBtn = document.getElementById('change-garment-btn');
    const removeGarmentBtn = document.getElementById('remove-garment-btn');
    const removePortraitBtn = document.getElementById('remove-portrait-btn');
    const loadSampleBtnNew = document.getElementById('load-sample-btn-new');
    const changePortraitLabel = document.getElementById('change-portrait-label');
    
    const elements = [changeGarmentBtn, removeGarmentBtn, removePortraitBtn, loadSampleBtnNew, changePortraitLabel];
    elements.forEach(el => {
        if (el) {
            el.disabled = running;
            if (running) {
                el.style.opacity = '0.5';
                el.style.pointerEvents = 'none';
                el.style.cursor = 'not-allowed';
            } else {
                el.style.opacity = '1';
                el.style.pointerEvents = 'auto';
                el.style.cursor = 'pointer';
            }
        }
    });
}

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
    const tryonBox = document.getElementById('tryon-output-preview-box');
    if (tryonBox) tryonBox.style.display = 'none';
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
        if (modelInfoText) modelInfoText.textContent = "No model active";

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

// Measurements UI panel handlers (Disabled per user request)
export function showMeasurementsPanel(measurements) {}

export function hideMeasurementsPanel() {}

export function checkTryonButtonState() {
    const runTryonBtn = document.getElementById('run-tryon-btn');
    if (!runTryonBtn) return;

    if (currentUploadedFile) {
        runTryonBtn.disabled = false;
        runTryonBtn.style.opacity = '1';
        runTryonBtn.style.cursor = 'pointer';

        if (selectedGarmentId) {
            runTryonBtn.textContent = "Run Virtual Try-On";
        } else {
            runTryonBtn.textContent = "Generate 3D Digital Twin";
        }
    } else {
        runTryonBtn.disabled = true;
        runTryonBtn.style.opacity = '0.5';
        runTryonBtn.style.cursor = 'not-allowed';
        runTryonBtn.textContent = "Run Virtual Try-On";
    }

    // Re-enable load demo button
    const loadSampleBtnNew = document.getElementById('load-sample-btn-new');
    if (loadSampleBtnNew) {
        loadSampleBtnNew.disabled = false;
        loadSampleBtnNew.style.opacity = '1';
        loadSampleBtnNew.style.cursor = 'pointer';
    }
}

export function removePortraitSelection() {
    if (isTryonRunning) return;
    currentUploadedFile = null;

    const portraitEmpty = document.getElementById('portrait-preview-empty');
    const portraitSelected = document.getElementById('portrait-preview-selected');
    const selectedPortraitImg = document.getElementById('selected-portrait-img');

    if (portraitEmpty && portraitSelected && selectedPortraitImg) {
        selectedPortraitImg.src = '';
        portraitEmpty.style.display = 'flex';
        portraitSelected.style.display = 'none';

        const wrapper = document.getElementById('portrait-selector-wrapper');
        if (wrapper) {
            wrapper.style.borderColor = 'var(--border-color)';
            wrapper.style.background = 'rgba(8, 8, 16, 0.6)';
        }
    }

    const fileInputEl = document.getElementById('file-input');
    if (fileInputEl) fileInputEl.value = '';
    const cameraInputEl = document.getElementById('camera-input');
    if (cameraInputEl) cameraInputEl.value = '';

    checkTryonButtonState();
}

// Model File Handler (now uploads photo to backend)
export function handleModelFile(file) {
    if (isTryonRunning) return;
    if (!file) return;

    // Validate that the file is an image
    const isImage = file.type.startsWith('image/') || /\.(jpe?g|png|gif|webp|heic|heif)$/i.test(file.name);
    if (!isImage) {
        alert("Please upload a valid image file (PNG or JPEG).");
        return;
    }

    currentUploadedFile = file;

    const portraitEmpty = document.getElementById('portrait-preview-empty');
    const portraitSelected = document.getElementById('portrait-preview-selected');
    const selectedPortraitImg = document.getElementById('selected-portrait-img');
    const selectedPortraitName = document.getElementById('selected-portrait-name');

    if (portraitEmpty && portraitSelected && selectedPortraitImg && selectedPortraitName) {
        selectedPortraitName.textContent = file.name;

        const reader = new FileReader();
        reader.onload = (e) => {
            selectedPortraitImg.src = e.target.result;
            portraitEmpty.style.display = 'none';
            portraitSelected.style.display = 'flex';

            const wrapper = document.getElementById('portrait-selector-wrapper');
            if (wrapper) {
                wrapper.style.borderColor = 'var(--accent-color)';
                wrapper.style.background = 'var(--card-hover-bg)';
            }
        };
        reader.readAsDataURL(file);
    }

    checkTryonButtonState();
}

function runTryonOnly(file, clothingId, useSuperResolution = true) {
    updateInteractiveElementsState(true);
    updateStatus("Running virtual try-on...", "loading");
    if (modelInfoText) modelInfoText.textContent = "CatVTON";

    const formData = new FormData();
    formData.append('photo', file);
    formData.append('clothing_id', clothingId);
    formData.append('use_super_resolution', useSuperResolution);

    const backendHost = window.location.hostname;
    const backendUrl = `http://${backendHost}:8000/api/v1/tryon-only`;

    const xhr = new XMLHttpRequest();
    xhr.open('POST', backendUrl);

    const gpuUrl = localStorage.getItem('gpu_server_url') || '';
    if (gpuUrl) {
        xhr.setRequestHeader('X-GPU-Server-URL', gpuUrl);
    }

    xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
            updateStatus("Running virtual try-on...", "loading");
        }
    });

    xhr.addEventListener('load', () => {
        updateInteractiveElementsState(false);
        checkTryonButtonState();
        if (xhr.status >= 200 && xhr.status < 300) {
            try {
                const data = JSON.parse(xhr.responseText);
                lastTryonId = data.tryonId;
                lastTryonImageUrl = data.tryonImageUrl;
                checkGenerateButtonsState();

                // Show fullscreen preview overlay
                const fullscreenOverlay = document.getElementById('tryon-fullscreen-overlay');
                const fullscreenImg = document.getElementById('tryon-fullscreen-img');

                if (fullscreenOverlay && fullscreenImg) {
                    const absoluteTryonUrl = lastTryonImageUrl.startsWith('http') ? lastTryonImageUrl : `http://${backendHost}:8000${lastTryonImageUrl}`;
                    fullscreenImg.src = absoluteTryonUrl;
                    fullscreenImg.style.display = 'block';
                    const placeholder = document.getElementById('tryon-fullscreen-placeholder');
                    if (placeholder) placeholder.style.display = 'none';

                    // Hide the main upload container and show the fullscreen overlay
                    uploadContainer.style.display = 'none';
                    fullscreenOverlay.style.display = 'flex';
                    fullscreenOverlay.style.opacity = '1';
                    fullscreenOverlay.style.visibility = 'visible';

                    // Load previous try-ons list dynamically
                    loadTryonHistory();

                    // Update success label dynamically
                    const successLabel = document.getElementById('tryon-success-label');
                    if (successLabel) {
                        successLabel.textContent = useSuperResolution 
                            ? "Try-On Completed (With Image Enhancement)" 
                            : "Try-On Completed";
                    }
                }

                updateStatus("Virtual try-on ready!", "active");
            } catch (err) {
                console.error("Failed to parse try-on response:", err);
                updateStatus("Try-on failed", "loading");
                alert("Failed to parse try-on response from server.");
            }
        } else {
            console.warn(`Server error (${xhr.status}):`, xhr.responseText);
            updateStatus("Try-on failed", "loading");
            alert(`Virtual try-on failed: Server error (${xhr.status})`);
        }
    });

    xhr.addEventListener('error', () => {
        updateInteractiveElementsState(false);
        checkTryonButtonState();
        updateStatus("Connection failed", "loading");
        alert("Virtual try-on connection failed.");
    });

    xhr.send(formData);
}

function reconstructTwinFromImage(file) {
    updateStatus("Uploading photo...", "loading");
    removeCurrentModel();
    hideMeasurementsPanel();

    // Hide upload box while showing the scene
    uploadContainer.style.display = 'none';

    const formData = new FormData();
    formData.append('photo', file);

    const backendHost = window.location.hostname;
    const backendUrl = `http://${backendHost}:8000/api/v1/generate-mesh`;

    const xhr = new XMLHttpRequest();
    xhr.open('POST', backendUrl);

    const colabUrl = localStorage.getItem('colab_tunnel_url') || '';
    if (colabUrl) {
        xhr.setRequestHeader('X-Colab-Tunnel-URL', colabUrl);
    }

    xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
            updateStatus("Uploading photo...", "loading");
        }
    });

    xhr.upload.addEventListener('load', () => {
        updateStatus("Running 3D AI Reconstruction...", "loading");
    });

    xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
            try {
                const data = JSON.parse(xhr.responseText);
                const meshUrl = data.meshUrl;
                const measurements = data.measurements;

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

                        showMeasurementsPanel(measurements);

                        if (modelInfoText) modelInfoText.textContent = "4D-Humans Mannequin";

                        showLoadAnotherButton();
                    },
                    (progress) => {
                        updateStatus("Downloading generated 3D twin...", "loading");
                    },
                    (error) => {
                        console.error("Error loading generated model:", error);
                        updateStatus("Load Failed", "loading");
                        alert("Failed to render the 3D model.");
                        uploadContainer.style.display = 'flex';
                        checkTryonButtonState();
                    }
                );
            } catch (err) {
                console.error("Failed to parse backend response:", err);
                triggerFallback("Failed to compile server response.");
                checkTryonButtonState();
            }
        } else {
            console.warn(`Server error (${xhr.status}):`, xhr.responseText);
            triggerFallback(`Server error (${xhr.status})`);
            checkTryonButtonState();
        }
    });

    xhr.addEventListener('error', () => {
        triggerFallback("Network connection error");
        checkTryonButtonState();
    });
    xhr.send(formData);
}

function reconstructTwinFromTryon(tryonId, method = "4dhumans") {
    updateStatus("Reconstructing 3D twin...", "loading");
    removeCurrentModel();
    hideMeasurementsPanel();

    // Hide fullscreen overlay and upload box
    const fullscreenOverlay = document.getElementById('tryon-fullscreen-overlay');
    if (fullscreenOverlay) {
        fullscreenOverlay.style.opacity = '0';
        fullscreenOverlay.style.visibility = 'hidden';
        setTimeout(() => {
            fullscreenOverlay.style.display = 'none';
        }, 400);
    }
    uploadContainer.style.display = 'none';

    const formData = new FormData();
    formData.append('tryon_id', tryonId);
    formData.append('method', method);

    const backendHost = window.location.hostname;
    const backendUrl = `http://${backendHost}:8000/api/v1/generate-mesh`;

    const xhr = new XMLHttpRequest();
    xhr.open('POST', backendUrl);

    const colabUrl = localStorage.getItem('colab_tunnel_url') || '';
    if (colabUrl) {
        xhr.setRequestHeader('X-Colab-Tunnel-URL', colabUrl);
    }

    const gpuUrl = localStorage.getItem('gpu_server_url') || '';
    if (gpuUrl) {
        xhr.setRequestHeader('X-GPU-Server-URL', gpuUrl);
    }

    xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
            updateStatus("Processing body sizing...", "loading");
        }
    });

    xhr.upload.addEventListener('load', () => {
        updateStatus("Running 3D AI Reconstruction...", "loading");
    });

    xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
            try {
                const data = JSON.parse(xhr.responseText);
                const meshUrl = data.meshUrl;
                const measurements = data.measurements;
                const tryonImageUrl = data.tryonImageUrl;

                const absoluteMeshUrl = meshUrl.startsWith('http') ? meshUrl : `http://${backendHost}:8000${meshUrl}`;

                // Show 2D try-on output image in the measurements HUD
                const tryonBox = document.getElementById('tryon-output-preview-box');
                const tryonImg = document.getElementById('tryon-output-preview-img');
                if (tryonImageUrl && tryonBox && tryonImg) {
                    const absoluteTryonUrl = tryonImageUrl.startsWith('http') ? tryonImageUrl : `http://${backendHost}:8000${tryonImageUrl}`;
                    tryonImg.src = absoluteTryonUrl;
                    tryonBox.style.display = 'block';
                }

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

                        showMeasurementsPanel(measurements);

                        if (modelInfoText) {
                            modelInfoText.textContent = method === "lhmpp" ? "LHM++ Splat" : "4D-Humans Mannequin";
                        }

                        showLoadAnotherButton();

                        // Reset step 2 UI state back to default upload box
                        resetTryonStep2UI();
                    },
                    (progress) => {
                        updateStatus("Downloading generated 3D twin...", "loading");
                    },
                    (error) => {
                        console.error("Error loading generated model:", error);
                        updateStatus("Load Failed", "loading");
                        alert("Failed to render the 3D model.");
                        uploadContainer.style.display = 'flex';
                        resetTryonStep2UI();
                    }
                );
            } catch (err) {
                console.error("Failed to parse backend response:", err);
                triggerFallback("Failed to compile server response.");
                resetTryonStep2UI();
            }
        } else {
            console.warn(`Server error (${xhr.status}):`, xhr.responseText);
            triggerFallback(`Server error (${xhr.status})`);
            resetTryonStep2UI();
        }
    });

    xhr.addEventListener('error', () => {
        triggerFallback("Network connection error");
        resetTryonStep2UI();
    });
    xhr.send(formData);
}

function resetTryonStep2UI() {
    const fullscreenOverlay = document.getElementById('tryon-fullscreen-overlay');
    if (fullscreenOverlay) {
        fullscreenOverlay.style.display = 'none';
    }
    lastTryonId = null;
    lastTryonImageUrl = null;
}

export function triggerFallback(reason) {
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

            if (modelInfoText) modelInfoText.textContent = "4D-Humans Mannequin (Demo)";

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

            if (modelInfoText) modelInfoText.textContent = "4D-Humans Mannequin (Demo)";

            showLoadAnotherButton();
        },
        (progress) => {
            updateStatus("Loading model...", "loading");
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
        btn.textContent = 'Create New Twin';
        btn.style.marginTop = '12px';
        btn.addEventListener('click', () => {
            removeCurrentModel();
            hideMeasurementsPanel();
            
            // Instantly hide tryon fullscreen overlay to prevent double click requirement
            const fullscreenOverlay = document.getElementById('tryon-fullscreen-overlay');
            if (fullscreenOverlay) {
                fullscreenOverlay.style.display = 'none';
                fullscreenOverlay.style.opacity = '0';
                fullscreenOverlay.style.visibility = 'hidden';
            }
            
            uploadContainer.style.display = 'flex';
            btn.style.display = 'none';
            
            // Hide mesh history panel
            const historyPanel = document.getElementById('mesh-history-panel');
            if (historyPanel) {
                historyPanel.style.display = 'none';
                historyPanel.classList.remove('active-modal');
            }
            // Hide floating buttons
            const floatAvatars = document.getElementById('floating-avatars-btn');
            const floatNav = document.getElementById('floating-navigation-btn');
            if (floatAvatars) floatAvatars.style.display = 'none';
            if (floatNav) floatNav.style.display = 'none';
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
    
    // Show floating buttons
    const floatAvatars = document.getElementById('floating-avatars-btn');
    const floatNav = document.getElementById('floating-navigation-btn');
    if (floatAvatars) floatAvatars.style.display = 'flex';
    if (floatNav) floatNav.style.display = 'flex';

    // Show mesh history panel and fetch list
    const historyPanel = document.getElementById('mesh-history-panel');
    if (historyPanel) {
        historyPanel.style.display = 'flex';
        loadMeshHistory();
    }
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
            fileInput.value = '';
        }
    });

    // Camera input change
    const cameraInput = document.getElementById('camera-input');
    if (cameraInput) {
        cameraInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleModelFile(e.target.files[0]);
                cameraInput.value = '';
            }
        });
    }

    // Drag and drop support
    if (portraitSelectorWrapper) {
        portraitSelectorWrapper.addEventListener('dragover', (e) => {
            e.preventDefault();
            portraitSelectorWrapper.classList.add('drag-over');
        });

        portraitSelectorWrapper.addEventListener('dragleave', () => {
            portraitSelectorWrapper.classList.remove('drag-over');
        });

        portraitSelectorWrapper.addEventListener('drop', (e) => {
            e.preventDefault();
            portraitSelectorWrapper.classList.remove('drag-over');
            if (e.dataTransfer.files.length > 0) {
                handleModelFile(e.dataTransfer.files[0]);
            }
        });
    }

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
    const gpuUrlInput = document.getElementById('gpu-url-input');

    if (settingsBtn && settingsPanel) {
        settingsBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const isOpen = (window.innerWidth <= 768) ? 
                settingsPanel.classList.contains('active-modal') : 
                (settingsPanel.style.display !== 'none' && settingsPanel.style.display !== '');
                
            if (isOpen) {
                settingsPanel.style.display = 'none';
                settingsPanel.classList.remove('active-modal');
            } else {
                settingsPanel.style.display = 'block';
                settingsPanel.classList.add('active-modal');
            }
        });
    }

    if (settingsCloseBtn && settingsPanel) {
        settingsCloseBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            settingsPanel.style.display = 'none';
            settingsPanel.classList.remove('active-modal');
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

    if (gpuUrlInput) {
        // Load cached URL on startup
        const cachedUrl = localStorage.getItem('gpu_server_url');
        if (cachedUrl) {
            gpuUrlInput.value = cachedUrl;
        }

        gpuUrlInput.addEventListener('input', () => {
            localStorage.setItem('gpu_server_url', gpuUrlInput.value.trim());
        });
    }

    // Set up try-on selection buttons
    const openCatalogBtn = document.getElementById('open-catalog-btn');
    const changeGarmentBtn = document.getElementById('change-garment-btn');
    const removeGarmentBtn = document.getElementById('remove-garment-btn');

    if (openCatalogBtn) openCatalogBtn.addEventListener('click', openCatalog);
    if (changeGarmentBtn) changeGarmentBtn.addEventListener('click', openCatalog);
    if (removeGarmentBtn) removeGarmentBtn.addEventListener('click', removeGarmentSelection);

    // Set up portrait remover
    const removePortraitBtn = document.getElementById('remove-portrait-btn');
    if (removePortraitBtn) removePortraitBtn.addEventListener('click', removePortraitSelection);

    // Modal buttons setup
    const srModal = document.getElementById('sr-modal');
    const srYesBtn = document.getElementById('sr-yes-btn');
    const srNoBtn = document.getElementById('sr-no-btn');
    const srCancelBtn = document.getElementById('sr-cancel-btn');

    // Main run button
    const runTryonBtn = document.getElementById('run-tryon-btn');

    const triggerRunTryonWithOption = (useSR) => {
        if (srModal) srModal.style.display = 'none';
        const loadSampleBtnNew = document.getElementById('load-sample-btn-new');
        
        if (runTryonBtn) {
            runTryonBtn.disabled = true;
            runTryonBtn.style.opacity = '0.5';
            runTryonBtn.style.cursor = 'not-allowed';
            runTryonBtn.textContent = "Running Try-On...";
        }
        if (loadSampleBtnNew) {
            loadSampleBtnNew.disabled = true;
            loadSampleBtnNew.style.opacity = '0.5';
            loadSampleBtnNew.style.cursor = 'not-allowed';
        }
        runTryonOnly(currentUploadedFile, selectedGarmentId, useSR);
    };

    if (srYesBtn) srYesBtn.addEventListener('click', () => triggerRunTryonWithOption(true));
    if (srNoBtn) srNoBtn.addEventListener('click', () => triggerRunTryonWithOption(false));
    if (srCancelBtn) srCancelBtn.addEventListener('click', () => {
        if (srModal) srModal.style.display = 'none';
    });

    if (runTryonBtn) {
        runTryonBtn.addEventListener('click', () => {
            if (!currentUploadedFile) return;

            const loadSampleBtnNew = document.getElementById('load-sample-btn-new');
            if (selectedGarmentId) {
                if (srModal) {
                    srModal.style.display = 'flex';
                } else {
                    // Fallback if modal is missing from DOM
                    triggerRunTryonWithOption(true);
                }
            } else {
                runTryonBtn.disabled = true;
                runTryonBtn.style.opacity = '0.5';
                runTryonBtn.style.cursor = 'not-allowed';
                runTryonBtn.textContent = "Reconstructing...";

                if (loadSampleBtnNew) {
                    loadSampleBtnNew.disabled = true;
                    loadSampleBtnNew.style.opacity = '0.5';
                    loadSampleBtnNew.style.cursor = 'not-allowed';
                }
                reconstructTwinFromImage(currentUploadedFile);
            }
        });
    }

    // Load fallback new button
    const loadSampleBtnNew = document.getElementById('load-sample-btn-new');
    if (loadSampleBtnNew) {
        loadSampleBtnNew.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            handleSampleModel();
        });
    }

    // Set up step 2 fullscreen preview buttons
    const tryonGenerate4dBtn = document.getElementById('tryon-generate-4d-btn');
    const tryonGenerateLhmBtn = document.getElementById('tryon-generate-lhm-btn');
    const tryonBackBtn = document.getElementById('tryon-back-btn');
    if (tryonGenerate4dBtn) {
        tryonGenerate4dBtn.addEventListener('click', () => {
            if (lastTryonId) {
                reconstructTwinFromTryon(lastTryonId, "4dhumans");
            }
        });
    }
    if (tryonGenerateLhmBtn) {
        tryonGenerateLhmBtn.addEventListener('click', () => {
            if (lastTryonId) {
                reconstructTwinFromTryon(lastTryonId, "lhmpp");
            }
        });
    }
    if (tryonBackBtn) {
        tryonBackBtn.addEventListener('click', () => {
            const fullscreenOverlay = document.getElementById('tryon-fullscreen-overlay');
            if (fullscreenOverlay) {
                fullscreenOverlay.style.display = 'none';
            }
            uploadContainer.style.display = 'flex';
        });
    }

    // Catalog modal overlay close triggers
    const catalogModal = document.getElementById('catalog-modal');
    const catalogCloseBtn = document.getElementById('catalog-close-btn');
    if (catalogCloseBtn && catalogModal) {
        catalogCloseBtn.addEventListener('click', closeCatalog);
        catalogModal.addEventListener('click', (e) => {
            if (e.target === catalogModal) {
                closeCatalog();
            }
        });
    }

    // Catalog search and filter event listeners
    const catalogSearch = document.getElementById('catalog-search');
    const catalogCategoryFilter = document.getElementById('catalog-category-filter');
    const catalogGenderFilters = document.getElementById('catalog-gender-filters');

    if (catalogSearch) catalogSearch.addEventListener('input', () => { currentCatalogPage = 1; renderCatalogGrid(); });
    if (catalogCategoryFilter) catalogCategoryFilter.addEventListener('change', () => { currentCatalogPage = 1; renderCatalogGrid(); });

    if (catalogGenderFilters) {
        const genderButtons = catalogGenderFilters.querySelectorAll('.tab-btn');
        genderButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                genderButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentCatalogPage = 1;
                renderCatalogGrid();
            });
        });
    }

    // Catalog pagination and confirm buttons
    const catalogPrevBtn = document.getElementById('catalog-prev-page');
    const catalogNextBtn = document.getElementById('catalog-next-page');
    const catalogConfirmBtn = document.getElementById('catalog-confirm-btn');

    if (catalogPrevBtn) {
        catalogPrevBtn.addEventListener('click', () => {
            if (currentCatalogPage > 1) {
                currentCatalogPage--;
                renderCatalogGrid();
            }
        });
    }

    if (catalogNextBtn) {
        catalogNextBtn.addEventListener('click', () => {
            const filtered = getFilteredCatalog();
            const totalPages = Math.ceil(filtered.length / itemsPerPage) || 1;
            if (currentCatalogPage < totalPages) {
                currentCatalogPage++;
                renderCatalogGrid();
            }
        });
    }

    if (catalogConfirmBtn) {
        catalogConfirmBtn.addEventListener('click', confirmGarmentSelection);
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
            if (backPortalBtn) backPortalBtn.style.display = 'flex';
            if (logoTitle) logoTitle.textContent = "3D Try-On Studio";
            if (logoSubtitle) logoSubtitle.textContent = "Immersive Mesh Inspector";
            if (settingsBtn) settingsBtn.style.display = 'inline-flex';
            const statusPanel = document.querySelector('.status-panel');
            if (statusPanel) statusPanel.style.display = 'flex';
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
            if (backPortalBtn) backPortalBtn.style.display = 'flex';
            if (logoTitle) logoTitle.textContent = "PREMADE ASSETS";
            if (logoSubtitle) logoSubtitle.textContent = "Object Showcase Catalog";
            if (settingsBtn) settingsBtn.style.display = 'none';
            if (settingsPanel) settingsPanel.style.display = 'none';
            const statusPanel = document.querySelector('.status-panel');
            if (statusPanel) statusPanel.style.display = 'none';
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

            const historyPanel = document.getElementById('mesh-history-panel');
            if (historyPanel) {
                historyPanel.style.display = 'none';
                historyPanel.classList.remove('active-modal');
            }
            // Hide floating buttons
            const floatAvatars = document.getElementById('floating-avatars-btn');
            const floatNav = document.getElementById('floating-navigation-btn');
            if (floatAvatars) floatAvatars.style.display = 'none';
            if (floatNav) floatNav.style.display = 'none';

            portalOverlay.style.visibility = 'visible';
            portalOverlay.style.opacity = '1';

            uploadContainer.style.display = 'none';
            if (premadePanel) premadePanel.style.display = 'none';
            backPortalBtn.style.display = 'none';
            if (logoTitle) logoTitle.textContent = "3D Try-On Studio";
            if (logoSubtitle) logoSubtitle.textContent = "Immersive Mesh Inspector";
            if (settingsBtn) settingsBtn.style.display = 'none';
            if (settingsPanel) settingsPanel.style.display = 'none';
            const statusPanel = document.querySelector('.status-panel');
            if (statusPanel) {
                statusPanel.style.display = 'none';
                statusPanel.classList.remove('active-modal');
            }
            updateStatus("Selecting Mode...", "active");
        });
    }

    const meshHistoryCloseBtn = document.getElementById('mesh-history-close-btn');
    const meshHistoryPanel = document.getElementById('mesh-history-panel');
    if (meshHistoryCloseBtn && meshHistoryPanel) {
        meshHistoryCloseBtn.addEventListener('click', () => {
            meshHistoryPanel.style.display = 'none';
            meshHistoryPanel.classList.remove('active-modal');
        });
    }

    // Mobile Close Triggers
    const statusCloseBtn = document.getElementById('status-close-btn');
    const statusPanelEl = document.querySelector('.status-panel');
    if (statusCloseBtn && statusPanelEl) {
        statusCloseBtn.addEventListener('click', () => {
            statusPanelEl.classList.remove('active-modal');
        });
    }

    const instructionsCloseBtn = document.getElementById('instructions-close-btn');
    const instructionsPanelEl = document.getElementById('instructions-panel');
    if (instructionsCloseBtn && instructionsPanelEl) {
        instructionsCloseBtn.addEventListener('click', (e) => {
            e.stopPropagation(); // prevent toggling collapsibility
            instructionsPanelEl.classList.remove('active-modal');
        });
    }

    // Floating Corner Buttons Event Listeners
    const floatingAvatarsBtn = document.getElementById('floating-avatars-btn');
    const floatingNavigationBtn = document.getElementById('floating-navigation-btn');

    if (floatingAvatarsBtn && meshHistoryPanel) {
        floatingAvatarsBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const isOpen = (window.innerWidth <= 768) ? 
                meshHistoryPanel.classList.contains('active-modal') : 
                (meshHistoryPanel.style.display !== 'none' && meshHistoryPanel.style.display !== '');
                
            if (isOpen) {
                meshHistoryPanel.style.display = 'none';
                meshHistoryPanel.classList.remove('active-modal');
            } else {
                meshHistoryPanel.style.display = 'flex';
                meshHistoryPanel.classList.add('active-modal');
                loadMeshHistory();
            }
        });
    }

    if (floatingNavigationBtn && instructionsPanelEl) {
        floatingNavigationBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const isOpen = (window.innerWidth <= 768) ? 
                instructionsPanelEl.classList.contains('active-modal') : 
                (instructionsPanelEl.style.display !== 'none' && instructionsPanelEl.style.display !== '');
                
            if (isOpen) {
                instructionsPanelEl.style.display = 'none';
                instructionsPanelEl.classList.remove('active-modal');
            } else {
                instructionsPanelEl.style.display = 'flex';
                instructionsPanelEl.classList.add('active-modal');
                instructionsPanelEl.classList.remove('collapsed');
                const ul = instructionsPanelEl.querySelector('ul');
                if (ul) ul.style.display = 'block';
            }
        });
    }

    // Shortcut History Access buttons inside 3D Studio main page
    const shortcutTryonBtn = document.getElementById('shortcut-tryon-history-btn');
    const shortcutMeshBtn = document.getElementById('shortcut-mesh-history-btn');

    if (shortcutTryonBtn) {
        shortcutTryonBtn.addEventListener('click', () => {
            const overlay = document.getElementById('tryon-fullscreen-overlay');
            const mainImg = document.getElementById('tryon-fullscreen-img');
            const successLabel = document.getElementById('tryon-success-label');
            const tryonHistoryModal = document.getElementById('tryon-history-modal');
            if (overlay) {
                overlay.style.display = 'flex';
                overlay.style.opacity = '1';
                overlay.style.visibility = 'visible';
                if (mainImg) {
                    mainImg.src = ''; // Clear image until selected
                    mainImg.style.display = 'none';
                }
                const placeholder = document.getElementById('tryon-fullscreen-placeholder');
                if (placeholder) placeholder.style.display = 'block';
                if (successLabel) successLabel.textContent = "Saved Try-On Images";
                
                lastTryonId = null;
                checkGenerateButtonsState();
                
                if (tryonHistoryModal) tryonHistoryModal.style.display = 'flex';
                loadTryonHistory();
            }
        });
    }

    if (shortcutMeshBtn) {
        shortcutMeshBtn.addEventListener('click', () => {
            uploadContainer.style.display = 'none';
            if (meshHistoryPanel) {
                meshHistoryPanel.style.display = 'flex';
                meshHistoryPanel.classList.add('active-modal');
                loadMeshHistory();
            }
            showLoadAnotherButton();
        });
    }

    // Try-On History Modal event listeners
    const openTryonHistoryBtn = document.getElementById('open-tryon-history-btn');
    const tryonHistoryCloseBtn = document.getElementById('tryon-history-close-btn');
    const tryonHistoryModal = document.getElementById('tryon-history-modal');

    if (openTryonHistoryBtn && tryonHistoryModal) {
        openTryonHistoryBtn.addEventListener('click', () => {
            tryonHistoryModal.style.display = 'flex';
            loadTryonHistory();
        });
    }

    if (tryonHistoryCloseBtn && tryonHistoryModal) {
        tryonHistoryCloseBtn.addEventListener('click', () => {
            tryonHistoryModal.style.display = 'none';
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

    // ── Theme Toggle Logic ──
    initThemeSystem();

    // Hide status panel on initial portal landing load
    const statusPanel = document.querySelector('.status-panel');
    if (statusPanel) statusPanel.style.display = 'none';
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

            if (modelInfoText) modelInfoText.textContent = displayName;
        },
        (progress) => {
            updateStatus("Downloading asset...", "loading");
        },
        (error) => {
            console.error("Error loading premade asset:", error);
            updateStatus("Load Failed", "loading");
            alert("Failed to render the 3D model. Verify the file path and format.");
        }
    );
}

// Catalog helpers
export function openCatalog() {
    if (isTryonRunning) return;
    const modal = document.getElementById('catalog-modal');
    if (!modal) return;

    modal.style.display = 'flex';
    setTimeout(() => modal.classList.add('active'), 10);

    // Copy currently selected garment to temporary modal state
    tempSelectedGarmentId = selectedGarmentId;
    tempSelectedGarmentData = selectedGarmentData;

    const confirmBtn = document.getElementById('catalog-confirm-btn');
    if (confirmBtn) {
        confirmBtn.disabled = !tempSelectedGarmentId;
    }
    updateSelectionInfo();

    // Fetch catalog if empty
    if (catalogItems.length === 0) {
        fetchCatalog();
    } else {
        currentCatalogPage = 1;
        renderCatalogGrid();
    }
}

export function closeCatalog() {
    const modal = document.getElementById('catalog-modal');
    if (!modal) return;

    modal.classList.remove('active');
    setTimeout(() => modal.style.display = 'none', 300);
}

function fetchCatalog() {
    const grid = document.getElementById('catalog-grid');
    if (grid) {
        grid.innerHTML = '<div style="grid-column: 1/-1; color: var(--text-muted); font-size: 13px; text-align: center; padding: 40px 0;">Loading catalog items...</div>';
    }

    const backendHost = window.location.hostname;
    const backendUrl = `http://${backendHost}:8000/api/v1/clothing-catalog`;

    fetch(backendUrl)
        .then(response => response.json())
        .then(data => {
            catalogItems = data;
            currentCatalogPage = 1;
            renderCatalogGrid();
        })
        .catch(err => {
            console.error("Failed to load clothing catalog:", err);
            if (grid) {
                grid.innerHTML = '<div style="grid-column: 1/-1; color: #ff0055; font-size: 13px; text-align: center; padding: 40px 0;">Failed to load catalog. Verify backend is running.</div>';
            }
        });
}

function getFilteredCatalog() {
    const searchVal = (document.getElementById('catalog-search')?.value || '').toLowerCase().trim();

    // Gender filter
    let genderVal = 'all';
    const activeGenderBtn = document.querySelector('#catalog-gender-filters .tab-btn.active');
    if (activeGenderBtn) {
        genderVal = activeGenderBtn.getAttribute('data-gender');
    }

    // Category filter
    const categoryVal = document.getElementById('catalog-category-filter')?.value || 'all';

    return catalogItems.filter(item => {
        // Gender filter
        if (genderVal !== 'all' && item.gender !== genderVal) return false;

        // Category filter
        if (categoryVal !== 'all' && item.category !== categoryVal) return false;

        // Search filter
        if (searchVal) {
            const name = (item.name || '').toLowerCase();
            const brand = (item.brand || '').toLowerCase();
            const cat = (item.category || '').toLowerCase();
            return name.includes(searchVal) || brand.includes(searchVal) || cat.includes(searchVal);
        }

        return true;
    });
}

function renderCatalogGrid() {
    const grid = document.getElementById('catalog-grid');
    if (!grid) return;

    const filtered = getFilteredCatalog();
    const totalPages = Math.ceil(filtered.length / itemsPerPage) || 1;

    // Safety check for current page
    if (currentCatalogPage > totalPages) currentCatalogPage = totalPages;
    if (currentCatalogPage < 1) currentCatalogPage = 1;

    // Update pagination labels
    const pageInfo = document.getElementById('catalog-page-info');
    if (pageInfo) pageInfo.textContent = `Page ${currentCatalogPage} of ${totalPages}`;

    const prevBtn = document.getElementById('catalog-prev-page');
    const nextBtn = document.getElementById('catalog-next-page');
    if (prevBtn) prevBtn.disabled = currentCatalogPage === 1;
    if (nextBtn) nextBtn.disabled = currentCatalogPage === totalPages;

    grid.innerHTML = '';

    if (filtered.length === 0) {
        grid.innerHTML = '<div style="grid-column: 1/-1; color: var(--text-muted); font-size: 13px; text-align: center; padding: 40px 0;">No matching garments found.</div>';
        return;
    }

    const startIndex = (currentCatalogPage - 1) * itemsPerPage;
    const paginatedItems = filtered.slice(startIndex, startIndex + itemsPerPage);

    const backendHost = window.location.hostname;

    paginatedItems.forEach(item => {
        const card = document.createElement('div');
        const isSelected = item.id === tempSelectedGarmentId;
        card.className = `catalog-card interactive ${isSelected ? 'selected' : ''}`;
        card.setAttribute('data-id', item.id);

        // Resolve image URL
        const imgUrl = item.imageUrl.startsWith('http') ? item.imageUrl : `http://${backendHost}:8000${item.imageUrl}`;

        card.innerHTML = `
            <div class="catalog-card-image-box">
                <img class="catalog-card-image" src="${imgUrl}" alt="${item.name}">
            </div>
            <div class="catalog-card-info">
                <div class="catalog-card-category">${item.category} • ${item.gender.toUpperCase()}</div>
                <div class="catalog-card-title" title="${item.name}">${item.name}</div>
                <div class="catalog-card-brand">${item.brand}</div>
            </div>
        `;

        card.addEventListener('click', () => {
            tempSelectedGarmentId = item.id;
            tempSelectedGarmentData = item;

            // Highlight selected card and remove highlight from others
            grid.querySelectorAll('.catalog-card').forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');

            // Enable confirm button
            const confirmBtn = document.getElementById('catalog-confirm-btn');
            if (confirmBtn) confirmBtn.disabled = false;

            updateSelectionInfo();
        });

        grid.appendChild(card);
    });
}

function updateSelectionInfo() {
    const selectionInfo = document.getElementById('catalog-selection-info');
    if (!selectionInfo) return;

    if (tempSelectedGarmentData) {
        selectionInfo.textContent = `Selected: ${tempSelectedGarmentData.name} (${tempSelectedGarmentData.brand})`;
        selectionInfo.style.color = 'var(--success-color)';
    } else {
        selectionInfo.textContent = 'No garment selected';
        selectionInfo.style.color = 'var(--text-muted)';
    }
}

export function confirmGarmentSelection() {
    selectedGarmentId = tempSelectedGarmentId;
    selectedGarmentData = tempSelectedGarmentData;

    closeCatalog();
    updateMainGarmentPreview();
}

export function removeGarmentSelection() {
    if (isTryonRunning) return;
    selectedGarmentId = null;
    selectedGarmentData = null;
    tempSelectedGarmentId = null;
    tempSelectedGarmentData = null;

    updateMainGarmentPreview();
}

function updateMainGarmentPreview() {
    const emptyPreview = document.getElementById('garment-preview-empty');
    const selectedPreview = document.getElementById('garment-preview-selected');
    const imgEl = document.getElementById('selected-garment-img');
    const nameEl = document.getElementById('selected-garment-name');
    const metaEl = document.getElementById('selected-garment-meta');

    if (!emptyPreview || !selectedPreview) return;

    if (selectedGarmentId && selectedGarmentData) {
        const backendHost = window.location.hostname;
        const imgUrl = selectedGarmentData.imageUrl.startsWith('http') ? selectedGarmentData.imageUrl : `http://${backendHost}:8000${selectedGarmentData.imageUrl}`;

        if (imgEl) imgEl.src = imgUrl;
        if (nameEl) nameEl.textContent = selectedGarmentData.name;
        if (metaEl) metaEl.textContent = `Category: ${selectedGarmentData.category} • Brand: ${selectedGarmentData.brand}`;

        emptyPreview.style.display = 'none';
        selectedPreview.style.display = 'flex';

        // Style wrapper slightly differently when active
        const wrapper = document.getElementById('selected-garment-wrapper');
        if (wrapper) {
            wrapper.style.borderColor = 'var(--accent-color)';
            wrapper.style.background = 'var(--card-hover-bg)';
        }
    } else {
        emptyPreview.style.display = 'flex';
        selectedPreview.style.display = 'none';

        const wrapper = document.getElementById('selected-garment-wrapper');
        if (wrapper) {
            wrapper.style.borderColor = 'var(--border-color)';
            wrapper.style.background = 'var(--panel-bg)';
        }
    }

    // Update main trigger button state (Run Virtual Try-On vs Generate 3D Twin)
    checkTryonButtonState();
}

// ── Theme System ──
function initThemeSystem() {
    const savedTheme = localStorage.getItem('app-theme') || 'dark';

    // Apply saved theme immediately
    applyTheme(savedTheme);

    // Bind theme toggle buttons (portal overlay + HUD header + try-on overlay)
    const portalToggle = document.getElementById('portal-theme-toggle-btn');
    const hudToggle = document.getElementById('theme-toggle-btn');
    const tryonToggle = document.getElementById('tryon-theme-toggle-btn');

    const toggleHandler = (e) => {
        e.stopPropagation();
        toggleTheme();
    };

    if (portalToggle) portalToggle.addEventListener('click', toggleHandler);
    if (hudToggle) hudToggle.addEventListener('click', toggleHandler);
    if (tryonToggle) tryonToggle.addEventListener('click', toggleHandler);
}

function toggleTheme() {
    const currentTheme = localStorage.getItem('app-theme') || 'dark';
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    applyTheme(newTheme);
    localStorage.setItem('app-theme', newTheme);
}

function applyTheme(themeName) {
    const body = document.body;
    const isDark = themeName === 'dark';

    // Toggle the CSS class
    if (isDark) {
        body.classList.add('theme-dark');
    } else {
        body.classList.remove('theme-dark');
    }

    // SVG markup matching user design
    const sunSvg = `
      <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" fill="none">
        <circle cx="12" cy="12" r="4"></circle>
        <line x1="12" y1="2" x2="12" y2="4"></line>
        <line x1="12" y1="20" x2="12" y2="22"></line>
        <line x1="4.93" y1="4.93" x2="6.34" y2="6.34"></line>
        <line x1="17.66" y1="17.66" x2="19.07" y2="19.07"></line>
        <line x1="2" y1="12" x2="4" y2="12"></line>
        <line x1="20" y1="12" x2="22" y2="12"></line>
        <line x1="6.34" y1="17.66" x2="4.93" y2="19.07"></line>
        <line x1="19.07" y1="4.93" x2="17.66" y2="6.34"></line>
      </svg>
    `;
    const moonSvg = `
      <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" fill="none">
        <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"></path>
      </svg>
    `;

    const iconHtml = isDark ? moonSvg : sunSvg;
    const portalIcon = document.getElementById('portal-theme-icon');
    const hudIcon = document.getElementById('theme-icon');
    const tryonIcon = document.getElementById('tryon-theme-icon');
    if (portalIcon) portalIcon.innerHTML = iconHtml;
    if (hudIcon) hudIcon.innerHTML = iconHtml;
    if (tryonIcon) tryonIcon.innerHTML = iconHtml;

    // Update Three.js scene
    updateSceneTheme(themeName);
}

export function loadTryonHistory() {
    const grid = document.getElementById('tryon-history-grid');
    if (!grid) return;
    
    grid.innerHTML = '<span style="font-size: 11px; color: var(--text-muted);">Loading history...</span>';
    
    const backendHost = window.location.hostname;
    const url = `http://${backendHost}:8000/api/v1/tryon/list`;
    
    fetch(url)
        .then(res => res.json())
        .then(data => {
            if (!data || data.length === 0) {
                grid.innerHTML = '<span style="font-size: 11px; color: var(--text-muted); grid-column: span 4;">No previous try-ons found.</span>';
                return;
            }
            grid.innerHTML = '';
            data.forEach(item => {
                const imgUrl = item.imageUrl.startsWith('http') ? item.imageUrl : `http://${backendHost}:8000${item.imageUrl}`;
                const card = document.createElement('div');
                card.style.cssText = `
                    cursor: pointer;
                    border: 1px solid var(--border-color);
                    border-radius: var(--border-radius-sm);
                    overflow: hidden;
                    aspect-ratio: 3/4;
                    background: var(--input-bg);
                    transition: border-color 0.2s;
                `;
                card.className = 'interactive';
                card.innerHTML = `<img src="${imgUrl}" style="width: 100%; height: 100%; object-fit: cover;">`;
                
                card.addEventListener('click', () => {
                    // Reset borders of all thumbnails in this history grid
                    const allCards = grid.querySelectorAll('div');
                    allCards.forEach(c => {
                        c.style.borderColor = 'var(--border-color)';
                        c.style.boxShadow = 'none';
                    });
                    
                    // Highlight the clicked thumbnail card
                    card.style.borderColor = 'var(--accent-color)';
                    card.style.boxShadow = '0 0 8px var(--accent-glow)';
                    
                    lastTryonId = item.tryonId;
                    lastTryonImageUrl = item.imageUrl;
                    
                    checkGenerateButtonsState();
                    const tryonHistoryModal = document.getElementById('tryon-history-modal');
                    if (tryonHistoryModal) tryonHistoryModal.style.display = 'none';
                    
                    const mainImg = document.getElementById('tryon-fullscreen-img');
                    if (mainImg) {
                        mainImg.src = imgUrl;
                        mainImg.style.display = 'block';
                        const placeholder = document.getElementById('tryon-fullscreen-placeholder');
                        if (placeholder) placeholder.style.display = 'none';
                    }
                });
                grid.appendChild(card);
            });
        })
        .catch(err => {
            console.error("Failed to load try-on history:", err);
            grid.innerHTML = '<span style="font-size: 11px; color: var(--text-muted); grid-column: span 4;">Failed to load history.</span>';
        });
}

export function loadMeshHistory() {
    const listContainer = document.getElementById('mesh-history-list');
    if (!listContainer) return;
    
    listContainer.innerHTML = '<span style="font-size: 11px; color: var(--text-muted);">Loading history...</span>';
    
    const backendHost = window.location.hostname;
    const url = `http://${backendHost}:8000/api/v1/meshes/list`;
    
    fetch(url)
        .then(res => res.json())
        .then(data => {
            if (!data || data.length === 0) {
                listContainer.innerHTML = '<span style="font-size: 11px; color: var(--text-muted);">No previous avatars found.</span>';
                return;
            }
            listContainer.innerHTML = '';
            data.forEach(item => {
                const btn = document.createElement('div');
                btn.style.cssText = `
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                    cursor: pointer;
                    padding: 12px 16px;
                    border: 1px solid var(--border-color);
                    border-radius: var(--border-radius-sm);
                    background: var(--portal-card-bg);
                    font-size: 12px;
                    line-height: 1.4;
                    color: var(--text-color);
                    text-align: left;
                    overflow: hidden;
                    white-space: normal;
                    word-break: break-word;
                    overflow-wrap: break-word;
                    flex-shrink: 0;
                    transition: border-color 0.2s, background-color 0.2s;
                `;
                btn.className = 'interactive';
                btn.textContent = item.filename;
                
                btn.addEventListener('click', () => {
                    const fullMeshUrl = item.meshUrl.startsWith('http') ? item.meshUrl : `http://${backendHost}:8000${item.meshUrl}`;
                    loadAvatarModelDirectly(fullMeshUrl, item.filename);
                    
                    // Close the mesh history panel automatically!
                    const meshHistoryPanel = document.getElementById('mesh-history-panel');
                    if (meshHistoryPanel) {
                        meshHistoryPanel.style.display = 'none';
                        meshHistoryPanel.classList.remove('active-modal');
                    }
                });
                listContainer.appendChild(btn);
            });
        })
        .catch(err => {
            console.error("Failed to load mesh history:", err);
            listContainer.innerHTML = '<span style="font-size: 11px; color: var(--text-muted);">Failed to load history.</span>';
        });
}

export function loadAvatarModelDirectly(meshUrl, filename) {
    updateStatus("Loading selected avatar...", "loading");
    removeCurrentModel();
    
    uploadContainer.style.display = 'none';
    
    loadModel(
        meshUrl,
        modelContainer,
        (model) => {
            currentModel = model;
            currentModelUrl = meshUrl;
            updateStatus(`Rendered: ${filename}`, "active");
            
            broadcastState({
                type: 'model_loaded',
                meshUrl: meshUrl,
                measurements: {
                    heightCm: 180,
                    chestCm: 96,
                    waistCm: 82,
                    hipCm: 98
                }
            });
            showLoadAnotherButton();
        },
        null,
        (err) => {
            console.error("Failed to load historical model:", err);
            updateStatus("Failed to load model", "loading");
            alert("Failed to load historical model from server.");
        },
        1.8
    );
}
