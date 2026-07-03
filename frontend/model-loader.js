import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

/**
 * Loads a GLB/GLTF model, normalizes it to a viewable size,
 * and adds it to the target parent group.
 * 
 * Premade assets: max dimension is always normalized to 1.5m.
 * Human twins: height is scaled to exactly 1.8m.
 * 
 * @param {string} url - Object URL or network URL of the model.
 * @param {THREE.Group} parentGroup - The parent group (e.g. modelContainer) to attach the model.
 * @param {Function} onLoad - Callback triggered when loading completes, receiving the loaded model.
 * @param {Function} onProgress - Progress callback.
 * @param {Function} onError - Error callback.
 */
export function loadModel(url, parentGroup, onLoad, onProgress, onError) {
    const loader = new GLTFLoader();
    
    loader.load(
        url,
        (gltf) => {
            const model = gltf.scene;

            // Force world matrix update immediately to get accurate child coordinates
            model.updateMatrixWorld(true);

            // Compute bounding box and center of the unscaled raw model
            const box = new THREE.Box3().setFromObject(model);
            const center = new THREE.Vector3();
            box.getCenter(center);
            const size = new THREE.Vector3();
            box.getSize(size);

            // Parse custom scale override from query parameters if present
            let customScale = 1.0;
            try {
                const urlObj = new URL(url, window.location.href);
                const scaleParam = urlObj.searchParams.get('scale');
                if (scaleParam) {
                    customScale = parseFloat(scaleParam);
                }
            } catch (e) {
                // Ignore query parameter errors on local relative file paths
            }

            // Determine scale factor
            let scaleFactor = 1.0;
            if (url.includes('premade')) {
                // Normalize every premade asset so its largest dimension = 1.5m
                const maxDim = Math.max(size.x, size.y, size.z);
                scaleFactor = 1.5 / (maxDim || 1.5);
                // Apply custom scale multiplier from premade_mapping.json overrides
                scaleFactor *= customScale;
            } else {
                // Scale human twin height to exactly 1.8m
                const height = size.y;
                scaleFactor = 1.8 / (height || 1.8);
            }

            // Apply scale
            model.scale.set(scaleFactor, scaleFactor, scaleFactor);

            // Apply exact scaled local position offsets:
            // 1. Center horizontally (X and Z)
            // 2. Align base of bounding box exactly with floor (Y = 0)
            model.position.x = -center.x * scaleFactor;
            model.position.z = -center.z * scaleFactor;
            model.position.y = -box.min.y * scaleFactor;

            // Commit matrix updates for rendering
            model.updateMatrixWorld(true);

            // Shadows & Material Settings
            model.traverse((child) => {
                if (child.isMesh) {
                    child.castShadow = true;
                    child.receiveShadow = true;
                    if (child.material) {
                        child.material.shadowSide = THREE.DoubleSide;
                        
                        // Optimize roughness and metalness to make materials look natural under lights
                        if (child.material.isMeshStandardMaterial) {
                            child.material.roughness = 0.55;
                            child.material.metalness = 0.05;
                        }
                    }
                }
            });

            // Add model to parentGroup
            parentGroup.add(model);

            if (onLoad) onLoad(model);
        },
        onProgress,
        onError
    );
}
