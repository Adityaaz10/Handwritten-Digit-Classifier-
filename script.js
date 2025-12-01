// --- GLOBAL VARIABLES ---
let model;
let data;
let isModelTrained = false;

// Canvas setup for drawing
const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
const statusMsg = document.getElementById('statusMsg');
const trainBtn = document.getElementById('trainBtn');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const predictionResult = document.getElementById('predictionResult');

// Setup drawing on canvas (White on Black to match MNIST)
ctx.lineWidth = 15;
ctx.lineCap = 'round';
ctx.strokeStyle = 'white'; // Draw white lines
ctx.fillStyle = 'black';   // Black background
ctx.fillRect(0, 0, canvas.width, canvas.height);

let isDrawing = false;

// Drawing Event Listeners
canvas.addEventListener('mousedown', (e) => { isDrawing = true; ctx.beginPath(); ctx.moveTo(e.offsetX, e.offsetY); });
canvas.addEventListener('mousemove', (e) => { if (isDrawing) { ctx.lineTo(e.offsetX, e.offsetY); ctx.stroke(); } });
canvas.addEventListener('mouseup', () => { isDrawing = false; });
canvas.addEventListener('mouseout', () => { isDrawing = false; });

// --- 1. MODEL ARCHITECTURE (IMPROVED for better accuracy) ---
function getModel() {
    const model = tf.sequential();
    
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;

    // Layer 1: Convolutional layer with 16 filters (increased from 8)
    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
        kernelSize: 3, // Smaller kernel size can sometimes capture fine detail better
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'heNormal' // Changed initializer for better convergence
    }));

    // Layer 2: Pooling
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

    // Layer 3: Second Convolutional layer with 32 filters (increased from 16)
    model.add(tf.layers.conv2d({
        kernelSize: 3,
        filters: 32,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'heNormal'
    }));

    // Layer 4: Second Pooling
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
    // Layer 5: Flatten the output
    model.add(tf.layers.flatten());

    // Layer 6: Dense Hidden Layer (New intermediate layer for better feature combination)
    model.add(tf.layers.dense({
        units: 128, // Arbitrary size, often a good starting point
        activation: 'relu'
    }));
    
    // Layer 7: Output Dense Layer
    model.add(tf.layers.dense({
        units: 10,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
    }));

    // Use a slightly lower learning rate with Adam, as we have more layers
    const optimizer = tf.train.adam(0.001); 
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    return model;
}

// --- 2. TRAINING FUNCTION (Increased data size and epochs) ---
async function train(model, data) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = { name: 'Model Training', tab: 'Training' };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 15000; // Increased from 5500
    const TEST_DATA_SIZE = 2500;   // Increased from 1000

    // Prepare training data
    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
        return [
            d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
            d.labels
        ];
    });

    // Prepare validation data
    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(TEST_DATA_SIZE);
        return [
            d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
            d.labels
        ];
    });

    return model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 15, // Increased from 10
        shuffle: true,
        callbacks: fitCallbacks
    });
}

// --- 3. PREDICTION FROM CANVAS (New Input) ---
function predictDrawing() {
    // IMPORTANT: Use a custom UI alert since window.alert() is forbidden.
    if (!isModelTrained) {
        document.getElementById('predictionResult').innerText = "ERROR: Please train the model first!";
        document.getElementById('predictionResult').style.color = '#e74c3c'; // Red for error
        return;
    }
    
    // Clear previous error/result style
    document.getElementById('predictionResult').style.color = '#27ae60';

    tf.tidy(() => {
        // 1. Get image data from canvas (280x280)
        let tensor = tf.browser.fromPixels(canvas, 1); // 1 channel (grayscale)
        
        // 2. Resize to 28x28 (MNIST size)
        const resized = tf.image.resizeBilinear(tensor, [28, 28]);
        
        // 3. Normalize (0-255 -> 0-1). White pixels (255) become 1.0, black (0) stays 0.
        const normalized = resized.div(255.0);

        // 4. Expand dims to get shape [1, 28, 28, 1] for model input
        const batched = normalized.expandDims(0);

        // 5. Predict
        const prediction = model.predict(batched);
        const index = prediction.argMax(1).dataSync()[0];
        const confidence = prediction.max().dataSync()[0];

        predictionResult.innerText = `Prediction: ${index} (Confidence: ${(confidence*100).toFixed(1)}%)`;
    });
}

// --- 4. INITIALIZATION ---
async function init() {
    statusMsg.innerText = "Downloading MNIST Data...";
    data = new MnistData();
    await data.load();
    statusMsg.innerText = "Data Loaded. Ready to Train.";
    trainBtn.disabled = false;
    model = getModel();
    // Show model architecture in the visor
    tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, model);
}

// Event Listeners
trainBtn.addEventListener('click', async () => {
    trainBtn.disabled = true;
    statusMsg.innerText = "Training Model... (Check the Visor tab on the right for live graphs)";
    try {
        await train(model, data);
        isModelTrained = true;
        statusMsg.innerText = "Training Complete! Try drawing a digit now.";
        predictBtn.disabled = false;
        clearBtn.disabled = false;
        trainBtn.innerText = "Re-Train Model";
    } catch (error) {
        statusMsg.innerText = "Training Failed! See console for details.";
        console.error("Training failed:", error);
    } finally {
        trainBtn.disabled = false;
    }
});

predictBtn.addEventListener('click', predictDrawing);

clearBtn.addEventListener('click', () => {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    predictionResult.innerText = "";
    document.getElementById('predictionResult').style.color = '#27ae60'; // Reset color
});

// Start App
init();