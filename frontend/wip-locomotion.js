import * as THREE from 'three';
import { renderer } from './scene-setup.js';

let isListening = false;
let stepThreshold = 2.4; // Acceleration m/s^2 threshold for registering a step (lowered from 3.0 to be responsive)
let lastStepTime = 0;
const stepCooldown = 350; // Cooldown to avoid double-stepping in a single bounce

// High-pass filter variables for removing gravity manually if needed
let xPrev = 0, yPrev = 0, zPrev = 0;
const filterFactor = 0.85;
let activeMotionListener = null;

// Head rotation tracking variables to suppress false steps when turning head
const lastCameraQuaternion = new THREE.Quaternion();
let rotationLockTime = 0;
const ROTATION_THRESHOLD = 0.04; // radians per event (approx 2.3 degrees; relaxed to ignore normal head bobbing)
const ROTATION_COOLDOWN = 300; // ms to lock step detection after rotation

// Track active screen touch or cardboard button press
let isSelectPressed = false;
let xrController = null;

function onSelectStart() {
    isSelectPressed = true;
    console.log("WIP Locomotion: Trigger/Touch pressed (BACKWARD step mode active)");
}

function onSelectEnd() {
    isSelectPressed = false;
    console.log("WIP Locomotion: Trigger/Touch released (FORWARD step mode active)");
}

/**
 * Handles device accelerometer motion events to detect steps.
 */
function handleMotion(event, cameraRig, camera) {
    if (!isListening) return;

    const now = performance.now();

    // 1. Head Rotation Suppression Check via Gyroscope (rotationRate)
    let isRotating = false;
    const rot = event.rotationRate;
    if (rot) {
        const x = rot.alpha || 0; // rotation around Z (in deg/s)
        const y = rot.beta || 0;  // rotation around X (in deg/s)
        const z = rot.gamma || 0; // rotation around Y (in deg/s)
        const rotationSpeed = Math.sqrt(x*x + y*y + z*z);
        
        // If rotation speed exceeds 60 degrees/second, lock step detection
        // (Relaxed from 15 deg/s to prevent normal stepping head bobbing from locking it)
        if (rotationSpeed > 60) {
            isRotating = true;
            rotationLockTime = now;
        }
    }

    // Secondary Head Rotation Check: Camera quaternion delta (fallback)
    const currentQuaternion = camera.quaternion;
    const angleChange = lastCameraQuaternion.angleTo(currentQuaternion);
    lastCameraQuaternion.copy(currentQuaternion);

    if (angleChange > ROTATION_THRESHOLD) {
        isRotating = true;
        rotationLockTime = now; // Lock step detection during rotation
    }

    // Suppress any step detection during active head rotation and for cooldown window
    if (isRotating || (now - rotationLockTime < ROTATION_COOLDOWN)) {
        return;
    }

    // 2. Retrieve linear acceleration (gravity removed by browser if supported)
    let acc = event.acceleration;
    const rawAcc = event.accelerationIncludingGravity;
    
    // Fallback to accelerationIncludingGravity and filter it if linear acceleration is null
    if (!acc || (acc.x === null && acc.y === null && acc.z === null)) {
        if (!rawAcc) return;
        
        // High-pass filter to subtract gravity vector
        xPrev = filterFactor * xPrev + (1 - filterFactor) * rawAcc.x;
        yPrev = filterFactor * yPrev + (1 - filterFactor) * rawAcc.y;
        zPrev = filterFactor * zPrev + (1 - filterFactor) * rawAcc.z;
        
        acc = {
            x: rawAcc.x - xPrev,
            y: rawAcc.y - yPrev,
            z: rawAcc.z - zPrev
        };
    }

    // 3. Project linear acceleration onto the gravity vector to isolate vertical steps
    let magnitude = 0;
    if (rawAcc && rawAcc.x !== null && rawAcc.y !== null && rawAcc.z !== null) {
        const gMag = Math.sqrt(rawAcc.x * rawAcc.x + rawAcc.y * rawAcc.y + rawAcc.z * rawAcc.z);
        if (gMag > 0.1) {
            // Unit vector representing the vertical axis (direction of gravity reaction)
            const ux = rawAcc.x / gMag;
            const uy = rawAcc.y / gMag;
            const uz = rawAcc.z / gMag;

            // Project linear acceleration onto the vertical axis
            const verticalAcc = acc.x * ux + acc.y * uy + acc.z * uz;
            magnitude = Math.abs(verticalAcc);
        } else {
            // Fallback to standard 3D magnitude if gravity magnitude is invalid
            magnitude = Math.sqrt(acc.x * acc.x + acc.y * acc.y + acc.z * acc.z);
        }
    } else {
        // Fallback to standard 3D magnitude
        magnitude = Math.sqrt(acc.x * acc.x + acc.y * acc.y + acc.z * acc.z);
    }

    // 4. Register step on acceleration impact and step cooldown
    if (magnitude > stepThreshold && (now - lastStepTime) > stepCooldown) {
        lastStepTime = now;
        
        // Get absolute gaze direction of the VR camera
        const horizontalDir = new THREE.Vector3();
        camera.getWorldDirection(horizontalDir);
        
        // Keep movement strictly horizontal (project onto XZ plane)
        horizontalDir.y = 0;
        horizontalDir.normalize();
        
        // Check pitch (Y component of gaze direction) to determine forward vs backward
        const gazeDirection = new THREE.Vector3();
        camera.getWorldDirection(gazeDirection);
        
        // Walk backward if either:
        // A) User is holding the Cardboard trigger / screen (isSelectPressed)
        // B) User is looking straight down at their feet (pitch Y < -0.75, approx. -48 degrees)
        const isLookingDown = gazeDirection.y < -0.75;
        const isBackward = isSelectPressed || isLookingDown;
        const stepSize = isBackward ? -0.4 : 0.4;
        
        console.log(`WIP Step! Magnitude: ${magnitude.toFixed(2)}, Dir: ${isBackward ? 'BACKWARD' : 'FORWARD'} (Select: ${isSelectPressed}, Pitch: ${gazeDirection.y.toFixed(2)})`);
        
        // Translate the cameraRig
        cameraRig.position.addScaledVector(horizontalDir, stepSize);
    }
}

/**
 * Starts listening to accelerometer events for Walking-in-Place tracking.
 */
export function startWIPLocomotion(cameraRig, camera) {
    if (isListening) return;
    
    xPrev = 0; yPrev = 0; zPrev = 0;
    lastCameraQuaternion.copy(camera.quaternion);
    rotationLockTime = 0;
    isSelectPressed = false;
    
    activeMotionListener = (e) => handleMotion(e, cameraRig, camera);

    // Bind to WebXR select triggers (screen touch on mobile cardboard viewer)
    if (renderer && renderer.xr) {
        xrController = renderer.xr.getController(0);
        if (xrController) {
            xrController.addEventListener('selectstart', onSelectStart);
            xrController.addEventListener('selectend', onSelectEnd);
        }
    }

    // Touch events on the window to support manual tap-and-hold in Cardboard
    window.addEventListener('touchstart', onSelectStart, { passive: true });
    window.addEventListener('touchend', onSelectEnd, { passive: true });

    if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
        DeviceMotionEvent.requestPermission()
            .then(response => {
                if (response === 'granted') {
                    window.addEventListener('devicemotion', activeMotionListener);
                    isListening = true;
                    console.log("Walking-in-Place locomotion initialized with permissions.");
                } else {
                    console.warn("Walking-in-Place locomotion permission denied.");
                }
            })
            .catch(console.error);
    } else {
        window.addEventListener('devicemotion', activeMotionListener);
        isListening = true;
        console.log("Walking-in-Place locomotion initialized.");
    }
}

/**
 * Stops listening to accelerometer events.
 */
export function stopWIPLocomotion() {
    if (!isListening) return;
    
    if (activeMotionListener) {
        window.removeEventListener('devicemotion', activeMotionListener);
        activeMotionListener = null;
    }

    // Clean up WebXR select listeners
    if (xrController) {
        xrController.removeEventListener('selectstart', onSelectStart);
        xrController.removeEventListener('selectend', onSelectEnd);
        xrController = null;
    }

    // Clean up window touch listeners
    window.removeEventListener('touchstart', onSelectStart);
    window.removeEventListener('touchend', onSelectEnd);
    
    isSelectPressed = false;
    isListening = false;
    console.log("Walking-in-Place locomotion terminated.");
}
