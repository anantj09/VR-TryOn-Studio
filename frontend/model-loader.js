import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

/**
 * Loads a GLB/GLTF or PLY model, normalizes it to a viewable size,
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
export function createCircleTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 16;
    canvas.height = 16;
    const ctx = canvas.getContext('2d');
    
    // Create radial gradient for smooth round edges
    const grad = ctx.createRadialGradient(8, 8, 0, 8, 8, 8);
    grad.addColorStop(0, 'rgba(255, 255, 255, 1)');
    grad.addColorStop(0.5, 'rgba(255, 255, 255, 0.7)');
    grad.addColorStop(1, 'rgba(255, 255, 255, 0)');
    
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 16, 16);
    
    return new THREE.CanvasTexture(canvas);
}

export function loadModel(url, parentGroup, onLoad, onProgress, onError) {
    const isPly = url.toLowerCase().includes('.ply');
    
    if (isPly) {
        const loader = new PLYLoader();
        loader.load(
            url,
            (geometry) => {
                let model;
                
                // Map Spherical Harmonics f_dc_0, f_dc_1, f_dc_2 to colors if color attribute is missing
                if (!geometry.attributes.color && geometry.attributes.f_dc_0) {
                    const count = geometry.attributes.position.count;
                    const colors = new Float32Array(count * 3);
                    const f_dc_0 = geometry.attributes.f_dc_0.array;
                    const f_dc_1 = geometry.attributes.f_dc_1.array;
                    const f_dc_2 = geometry.attributes.f_dc_2.array;
                    
                    const SH_C0 = 0.28209479177387814;
                    for (let i = 0; i < count; i++) {
                        let r = 0.5 + SH_C0 * f_dc_0[i];
                        let g = 0.5 + SH_C0 * f_dc_1[i];
                        let b = 0.5 + SH_C0 * f_dc_2[i];
                        
                        colors[i * 3] = Math.max(0, Math.min(1, r));
                        colors[i * 3 + 1] = Math.max(0, Math.min(1, g));
                        colors[i * 3 + 2] = Math.max(0, Math.min(1, b));
                    }
                    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                    console.log(">>> Converted Gaussian Splat SH f_dc to RGB colors.");
                } else if (!geometry.attributes.color && geometry.attributes.features_0) {
                    const count = geometry.attributes.position.count;
                    const colors = new Float32Array(count * 3);
                    const f_0 = geometry.attributes.features_0.array;
                    const f_1 = geometry.attributes.features_1.array;
                    const f_2 = geometry.attributes.features_2.array;
                    
                    const SH_C0 = 0.28209479177387814;
                    for (let i = 0; i < count; i++) {
                        let r = 0.5 + SH_C0 * f_0[i];
                        let g = 0.5 + SH_C0 * f_1[i];
                        let b = 0.5 + SH_C0 * f_2[i];
                        
                        colors[i * 3] = Math.max(0, Math.min(1, r));
                        colors[i * 3 + 1] = Math.max(0, Math.min(1, g));
                        colors[i * 3 + 2] = Math.max(0, Math.min(1, b));
                    }
                    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                    console.log(">>> Converted Gaussian Splat SH features to RGB colors.");
                }
                
                // If the PLY has no faces/indices, it is a Point Cloud (Gaussian Splat)
                if (!geometry.index) {
                    const pointCount = geometry.attributes.position ? geometry.attributes.position.count : 0;
                    
                    // Larger point size to make the point cloud look solid and gap-free
                    let pointSize = 0.045;
                    if (pointCount > 300000) {
                        pointSize = 0.025;
                    } else if (pointCount > 150000) {
                        pointSize = 0.04;
                    } else if (pointCount > 80000) {
                        pointSize = 0.05;
                    } else {
                        pointSize = 0.06;
                    }
                    
                    const dotTexture = createCircleTexture();
                    
                    let material;
                    if (geometry.attributes.color) {
                        material = new THREE.PointsMaterial({
                            size: pointSize,
                            vertexColors: true,
                            sizeAttenuation: true,
                            transparent: true,
                            opacity: 1.0,
                            map: dotTexture,
                            alphaTest: 0.4,
                            depthWrite: true, // Prevents background points from bleeding through
                            depthTest: true
                        });
                    } else {
                        material = new THREE.PointsMaterial({
                            color: 0x00f0ff, // Cyberpunk cyan fallback
                            size: pointSize,
                            sizeAttenuation: true,
                            transparent: true,
                            opacity: 1.0,
                            map: dotTexture,
                            alphaTest: 0.4,
                            depthWrite: true,
                            depthTest: true
                        });
                    }
                    
                    model = new THREE.Points(geometry, material);
                    console.log(`>>> Rendered PLY as Points (Point Cloud). Count: ${pointCount}`);
                } else {
                    // Standard mesh with faces
                    if (!geometry.attributes.normal) {
                        geometry.computeVertexNormals();
                    }
                    
                    let material;
                    if (geometry.attributes.color) {
                        material = new THREE.MeshStandardMaterial({
                            vertexColors: true,
                            roughness: 0.6,
                            metalness: 0.1,
                            side: THREE.DoubleSide
                        });
                    } else {
                        material = new THREE.MeshStandardMaterial({
                            color: 0xcccccc,
                            roughness: 0.6,
                            metalness: 0.1,
                            side: THREE.DoubleSide
                        });
                    }
                    
                    model = new THREE.Mesh(geometry, material);
                    console.log(">>> Rendered PLY as Mesh (Polygon).");
                }
                
                // Force world matrix update immediately
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
                } catch (e) {}

                // Determine scale factor (scale human height to exactly 1.8m)
                const height = size.y;
                let scaleFactor = 1.8 / (height || 1.8);
                
                // Apply scale
                model.scale.set(scaleFactor, scaleFactor, scaleFactor);

                // Align center and base floor
                model.position.x = -center.x * scaleFactor;
                model.position.z = -center.z * scaleFactor;
                model.position.y = -box.min.y * scaleFactor;

                // Commit matrix updates for rendering
                model.updateMatrixWorld(true);

                // Shadows Settings
                model.castShadow = true;
                model.receiveShadow = true;

                // Add model to parentGroup
                parentGroup.add(model);

                if (onLoad) onLoad(model);
            },
            onProgress,
            onError
        );
    } else {
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
}
