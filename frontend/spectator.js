import { initScene, renderer, camera, scene, controls, modelContainer } from './scene-setup.js';
import { loadModel } from './model-loader.js';
import { initWebSocket, addMessageListener, broadcastState } from './ws-client.js';

const canvasContainer = document.getElementById('canvas-container');
const streamStatus = document.getElementById('stream-status');
const statusText = document.getElementById('status-text');
const statusDot = document.getElementById('status-dot');
const modelInfoText = document.getElementById('model-info-text');

const infoStreamer = document.getElementById('info-streamer');
const infoModel = document.getElementById('info-model');

const measurementsPanel = document.getElementById('measurements-panel');
const valHeight = document.getElementById('val-height');
const valChest = document.getElementById('val-chest');
const valWaist = document.getElementById('val-waist');
const valHip = document.getElementById('val-hip');

let currentModel = null;
let streamActive = false;
let lastCameraUpdateTime = 0;

function updateStatus(text, stateClass) {
    if (statusText) statusText.textContent = text;
    if (statusDot) {
        statusDot.className = "status-dot " + stateClass;
    }
}

function updateStreamBadge(active) {
    if (active) {
        streamStatus.textContent = "LIVE STREAMING";
        streamStatus.className = "stream-badge active";
    } else {
        streamStatus.textContent = "WAITING FOR CLIENT";
        streamStatus.className = "stream-badge";
    }
}

function showMeasurements(measurements) {
    if (!measurementsPanel) return;
    if (valHeight) valHeight.textContent = `${Math.round(measurements.heightCm)}cm`;
    if (valChest) valChest.textContent = `${Math.round(measurements.chestCm)}cm`;
    if (valWaist) valWaist.textContent = `${Math.round(measurements.waistCm)}cm`;
    if (valHip) valHip.textContent = `${Math.round(measurements.hipCm)}cm`;
    measurementsPanel.style.display = 'block';
}

function hideMeasurements() {
    if (measurementsPanel) {
        measurementsPanel.style.display = 'none';
    }
}

function removeCurrentModel() {
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
    }
    modelInfoText.textContent = "No active stream";
    infoModel.textContent = "None";
}

// Count polygons utility
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

// WebSocket state listener
function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'ws_status':
            if (data.status === 'connected') {
                updateStatus("Connected to relay", "active");
                updateStreamBadge(false);
                // Request current state from any active phone clients
                broadcastState({ type: 'sync_request' });
            } else {
                updateStatus("Disconnected", "loading");
                updateStreamBadge(false);
                streamActive = false;
                controls.enabled = true;
            }
            break;

        case 'load':
            let modelUrl = data.url;
            console.log("Original model URL received:", modelUrl);
            
            // Normalize URLs to prevent CORS errors on the spectator page
            if (modelUrl.includes(':8080')) {
                const urlObj = new URL(modelUrl);
                modelUrl = urlObj.pathname + urlObj.search;
            } else if (modelUrl.includes(':8000')) {
                const urlObj = new URL(modelUrl);
                urlObj.hostname = window.location.hostname;
                modelUrl = urlObj.href;
            }
            console.log("Normalized model URL for loading:", modelUrl);
            
            removeCurrentModel();
            hideMeasurements();
            
            infoStreamer.textContent = "Mobile Phone";
            infoModel.textContent = modelUrl.split('/').pop();
            updateStatus("Loading remote mesh...", "loading");

            loadModel(
                modelUrl,
                modelContainer,
                (model) => {
                    currentModel = model;
                    updateStatus("Mesh Synced", "active");
                    
                    if (data.measurements) {
                        showMeasurements(data.measurements);
                    }
                    
                    modelInfoText.innerHTML = `
                        Model: Synced Digital Twin<br>
                        Polygons: ${countPolygons(model).toLocaleString()} faces
                    `;
                },
                null,
                (err) => {
                    console.error("Failed to load synced model:", err);
                    updateStatus("Sync Failed", "loading");
                }
            );
            break;

        case 'unload':
            console.log("Unloading model from mobile");
            removeCurrentModel();
            hideMeasurements();
            infoStreamer.textContent = "None";
            updateStatus("Ready (Waiting for Load)", "active");
            updateStreamBadge(false);
            streamActive = false;
            controls.enabled = true;
            break;

        case 'transform':
            if (currentModel) {
                modelContainer.rotation.y = data.rotation;
                modelContainer.scale.set(data.scale, data.scale, data.scale);
            }
            break;

        case 'camera':
            // Sync camera coordinates from VR/Mobile device
            controls.enabled = false; // Disable OrbitControls to allow WS overrides
            camera.position.set(data.position.x, data.position.y, data.position.z);
            camera.quaternion.set(data.quaternion.x, data.quaternion.y, data.quaternion.z, data.quaternion.w);
            
            lastCameraUpdateTime = performance.now();
            if (!streamActive) {
                streamActive = true;
                updateStreamBadge(true);
                infoStreamer.textContent = "Immersive VR Session";
            }
            break;
    }
}

// Main spectator loop
function renderLoop() {
    if (controls.enabled) {
        controls.update();
    }

    // Check if camera streaming stopped (e.g. idle or VR ended)
    if (streamActive && performance.now() - lastCameraUpdateTime > 3000) {
        console.log("Camera updates stopped. Re-enabling OrbitControls.");
        streamActive = false;
        controls.enabled = true;
        updateStreamBadge(false);
        infoStreamer.textContent = "Mobile Phone";
        
        // Smoothly transition camera back to standard view
        camera.position.set(0, 1.6, 1.0);
        controls.target.set(0, 0.9, -1.5);
        controls.update();
    }

    renderer.render(scene, camera);
}

function startSpectator() {
    initScene(canvasContainer);
    
    // Setup desktop camera controls starting point
    camera.position.set(0, 1.6, 1.0);
    controls.target.set(0, 0.9, -1.5);
    controls.update();

    // Connect websocket
    addMessageListener(handleWebSocketMessage);
    initWebSocket();

    // Start loop
    renderer.setAnimationLoop(renderLoop);
}

startSpectator();
