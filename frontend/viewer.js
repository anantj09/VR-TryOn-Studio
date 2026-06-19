import * as THREE from 'three';
import { initScene, renderer, camera, scene, controls, modelContainer } from './scene-setup.js';
import { setupUI, instructionsPanel, currentModel, currentModelUrl, currentModelMeasurements } from './ui-handlers.js';
import { setupWebXR } from './xr-manager.js';
import { pollGamepad } from './gamepad.js';
import { initWebSocket, broadcastState, addMessageListener } from './ws-client.js';

// Respond to sync requests from newly connected spectators
addMessageListener((data) => {
    if (data.type === 'sync_request') {
        if (currentModel && currentModelUrl) {
            console.log("Broadcasting current state to new spectator...");
            broadcastState({
                type: 'load',
                url: currentModelUrl,
                measurements: currentModelMeasurements
            });
            broadcastState({
                type: 'transform',
                rotation: modelContainer.rotation.y,
                scale: modelContainer.scale.x
            });
        }
    }
});

// Get viewport canvas container
const canvasContainer = document.getElementById('canvas-container');

// State tracking for transformation changes
let lastRotationY = 0;
let lastScaleX = 1;

// Camera throttle variables
const tempPosition = new THREE.Vector3();
const tempQuaternion = new THREE.Quaternion();
let lastCameraUpdateTime = 0;
const CAMERA_UPDATE_INTERVAL = 33; // ~30 FPS updates

// Main Render Loop (used by Three.js WebXR)
function renderLoop() {
    // 1. Update controls in desktop mode
    if (controls.enabled) {
        controls.update();
    }

    // 2. Poll Bluetooth Gamepad controller inputs
    pollGamepad(modelContainer, renderer, instructionsPanel);

    // 3. Sync State if model is active
    if (currentModel) {
        // Track and broadcast model container transformations (rotations/scale)
        const rotationDiff = Math.abs(modelContainer.rotation.y - lastRotationY);
        const scaleDiff = Math.abs(modelContainer.scale.x - lastScaleX);
        
        if (rotationDiff > 0.002 || scaleDiff > 0.002) {
            broadcastState({
                type: 'transform',
                rotation: modelContainer.rotation.y,
                scale: modelContainer.scale.x
            });
            lastRotationY = modelContainer.rotation.y;
            lastScaleX = modelContainer.scale.x;
        }

        // Throttle and broadcast the camera's world position/orientation
        const now = performance.now();
        if (now - lastCameraUpdateTime >= CAMERA_UPDATE_INTERVAL) {
            camera.getWorldPosition(tempPosition);
            camera.getWorldQuaternion(tempQuaternion);
            
            broadcastState({
                type: 'camera',
                position: { x: tempPosition.x, y: tempPosition.y, z: tempPosition.z },
                quaternion: { x: tempQuaternion.x, y: tempQuaternion.y, z: tempQuaternion.z, w: tempQuaternion.w }
            });
            
            lastCameraUpdateTime = now;
        }
    }

    // 4. Render frame
    renderer.render(scene, camera);
}

// Initialize application components
function startApp() {
    // 1. Initialize scene/camera/renderer
    initScene(canvasContainer);
    
    // 2. Set up WebXR Button state
    setupWebXR();
    
    // 3. Set up UI event listeners
    setupUI();
    
    // 4. Initialize real-time WebSocket connection
    initWebSocket();
    
    // 5. Start Three.js Animation Loop
    renderer.setAnimationLoop(renderLoop);
}

// Launch application
startApp();
