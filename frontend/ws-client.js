let socket = null;
let socketReady = false;
const messageQueue = [];
const listeners = new Set();

export function initWebSocket() {
    const wsHost = window.location.hostname || 'localhost';
    const wsUrl = `ws://${wsHost}:8000/ws/stream`;
    
    console.log(`Connecting to WebSocket: ${wsUrl}`);
    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        console.log("WebSocket connected!");
        socketReady = true;
        
        // Notify listeners of connection status
        listeners.forEach(callback => callback({ type: 'ws_status', status: 'connected' }));
        
        while (messageQueue.length > 0 && socketReady) {
            const msg = messageQueue.shift();
            socket.send(msg);
        }
    };

    socket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            listeners.forEach(callback => callback(data));
        } catch (e) {
            console.error("Failed to parse WebSocket message:", e);
        }
    };

    socket.onclose = () => {
        console.warn("WebSocket disconnected, retrying in 3 seconds...");
        socketReady = false;
        // Notify listeners of connection status
        listeners.forEach(callback => callback({ type: 'ws_status', status: 'disconnected' }));
        setTimeout(initWebSocket, 3000);
    };

    socket.onerror = (error) => {
        console.error("WebSocket error:", error);
    };
}

export function broadcastState(data) {
    const msg = JSON.stringify(data);
    if (socket && socketReady) {
        socket.send(msg);
    } else {
        messageQueue.push(msg);
        if (messageQueue.length > 100) {
            messageQueue.shift();
        }
    }
}

export function addMessageListener(callback) {
    listeners.add(callback);
}

export function removeMessageListener(callback) {
    listeners.delete(callback);
}
