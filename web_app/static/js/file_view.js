// Beat Detection App - File View JavaScript

/*
POLLING LOGIC:
1. When polling starts:
   - During initialization (init function):
     - When status is 'ANALYZING' - polling starts
     - When status is 'GENERATING_VIDEO' - polling starts
   - During user actions:
     - When user clicks "Generate Video" button (handleConfirmAnalysis function) - polling starts

2. When polling stops:
   - When status is 'ANALYZED' with valid stats and no video generation in progress - polling stops
   - When status is 'COMPLETED' - polling stops
   - When status is 'ERROR', 'ANALYZING_FAILURE', or 'VIDEO_ERROR' - polling stops

3. Error states (no polling):
   - When status is 'ANALYZED' but no valid stats are found - this is an error state, polling does not start
*/

// Global state
let currentFileId = null;
let statusCheckInterval = null;
let debugMode = true;
let currentTaskId = null;  // Track the currently active task ID
let currentTaskType = null;  // Track the task type (beat_detection or video_generation)

// Initialize the application when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing Beat Detection App');
    
    // Check for initialFileData
    if (window.initialFileData && window.initialFileData.fileId) {
        console.log('Using window.initialFileData:', window.initialFileData);
        initWithFileData(window.initialFileData);
    } else {
        console.error('No initialFileData available. Cannot initialize application.');
    }
});

// Initialize app with file data
function initWithFileData(fileData) {
    // Validate file ID before setting it
    const fileId = fileData.fileId;
    if (!fileId || typeof fileId !== 'string') {
        console.error('Invalid file ID format:', fileId);
        return;
    }
    
    // Set current file ID for use in API calls
    currentFileId = fileId;
    
    // Initialize UI based on current status
    handleStatus(fileData);
    
    // Set up event listeners
    setupEventListeners();
    
    // Set up debug panel if enabled
    if (debugMode) {
        setupDebugPanel();
    }
}

// Set up event listeners
function setupEventListeners() {
    // Confirm button - Generate Video
    const confirmButton = document.getElementById('confirm-button');
    if (confirmButton) {
        confirmButton.addEventListener('click', function() {
            if (!currentFileId) return;
            
            // Disable the button
            confirmButton.disabled = true;
            
            // Send request to confirm analysis
            fetch(`/confirm/${currentFileId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            })
            .then(response => {
                if (response.redirected) {
                    window.location.href = response.url;
                } else if (response.ok) {
                    window.location.reload();
                } else {
                    return response.json().then(data => {
                        throw new Error(data.detail || 'Failed to confirm analysis');
                    });
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error: ' + error.message);
                confirmButton.disabled = false;
            });
        });
    }
    
    // Cancel button
    const cancelButton = document.getElementById('cancel-button');
    if (cancelButton) {
        cancelButton.addEventListener('click', function() {
            window.location.href = '/';
        });
    }
    
    // Restart button
    const restartButton = document.getElementById('restart-button');
    if (restartButton) {
        restartButton.addEventListener('click', function() {
            window.location.href = currentFileId ? 
                `/?restart=${currentFileId}` : '/';
        });
    }
}

// Handle different file statuses
function handleStatus(statusData) {
    if (!statusData) {
        console.error('No status data provided');
        return;
    }
    
    const status = statusData.status;
    console.log('Processing status:', status);
    
    // Get DOM elements
    const analysisSection = document.getElementById('analysis-section');
    const analysisProgress = document.getElementById('analysis-progress');
    const analysisResults = document.getElementById('analysis-results');
    const analysisButtons = document.getElementById('analysis-buttons');
    const videoSection = document.getElementById('video-section');
    const videoProgress = document.getElementById('video-progress');
    const videoResults = document.getElementById('video-results');
    
    // Process based on status
    switch (status) {
        case 'ANALYZING':
            console.log('Showing analyzing UI');
            // Show analysis progress
            if (analysisSection) analysisSection.classList.remove('hidden');
            if (analysisProgress) analysisProgress.classList.remove('hidden');
            if (analysisResults) analysisResults.classList.add('hidden');
            if (analysisButtons) analysisButtons.classList.add('hidden');
            
            // Hide video section
            if (videoSection) videoSection.classList.add('hidden');
            
            // Set the beat detection task ID for polling
            if (statusData.beatDetectionTaskId) {
                currentTaskId = statusData.beatDetectionTaskId;
                currentTaskType = 'beat_detection';
                // Start polling
                startStatusPolling();
            }
            break;
            
        case 'ANALYZED':
            console.log('Showing analyzed UI');
            // Show analysis results and confirm button
            if (analysisSection) analysisSection.classList.remove('hidden');
            if (analysisProgress) analysisProgress.classList.add('hidden');
            if (analysisResults) analysisResults.classList.remove('hidden');
            if (analysisButtons) analysisButtons.classList.remove('hidden');
            
            // Hide video section
            if (videoSection) videoSection.classList.add('hidden');
            
            // Display results using the flattened data structure
            displayAnalysisResults(statusData);
            
            // Since analysis is complete, clear the task ID and stop polling
            currentTaskId = null;
            currentTaskType = null;
            stopPolling();
            break;
            
        case 'ANALYZING_FAILURE':
            // Show error for failed beat analysis
            const analysisErrorMessage = document.createElement('div');
            analysisErrorMessage.className = 'error-message';
            
            // Get error message from status data
            let errorMessage = 'Error: Beat analysis failed.';
            if (statusData.error) {
                if (typeof statusData.error === 'object' && statusData.error.message) {
                    errorMessage = statusData.error.message;
                } else if (typeof statusData.error === 'string') {
                    errorMessage = statusData.error;
                }
            }
            
            analysisErrorMessage.textContent = errorMessage;
            if (analysisResults) analysisResults.prepend(analysisErrorMessage);
            
            // Clear task ID and stop polling
            currentTaskId = null;
            currentTaskType = null;
            stopPolling();
            break;
            
        case 'GENERATING_VIDEO':
            console.log('Showing video generation progress UI');
            // Show analysis results and video progress
            if (analysisSection) {
                analysisSection.classList.remove('hidden');
                analysisSection.classList.add('with-video');
            }
            if (analysisProgress) analysisProgress.classList.add('hidden');
            if (analysisResults) analysisResults.classList.remove('hidden');
            if (analysisButtons) analysisButtons.classList.add('hidden');
            
            // Show video section with progress
            if (videoSection) videoSection.classList.remove('hidden');
            if (videoProgress) videoProgress.classList.remove('hidden');
            if (videoResults) videoResults.classList.add('hidden');
            
            // Display analysis results
            displayAnalysisResults(statusData);
            
            // Set the video generation task ID for polling
            if (statusData.videoGenerationTaskId) {
                currentTaskId = statusData.videoGenerationTaskId;
                currentTaskType = 'video_generation';
                // Start polling
                startStatusPolling();
            }
            break;
            
        case 'COMPLETED':
            console.log('Showing completed UI with video');
            // Show analysis results and video
            if (analysisSection) {
                analysisSection.classList.remove('hidden');
                analysisSection.classList.add('with-video');
            }
            if (analysisProgress) analysisProgress.classList.add('hidden');
            if (analysisResults) analysisResults.classList.remove('hidden');
            if (analysisButtons) analysisButtons.classList.add('hidden');
            
            // Show video section
            if (videoSection) videoSection.classList.remove('hidden');
            if (videoProgress) videoProgress.classList.add('hidden');
            if (videoResults) videoResults.classList.remove('hidden');
            
            // Display results
            displayAnalysisResults(statusData);
            
            // Display video
            displayVideo();
            
            // Clear task ID and stop polling
            currentTaskId = null;
            currentTaskType = null;
            stopPolling();
            break;
            
        case 'VIDEO_ERROR':
            // Show error for failed video generation
            const videoErrorMessage = document.createElement('div');
            videoErrorMessage.className = 'error-message';
            videoErrorMessage.textContent = 'Error: Video generation failed.';
            if (videoResults) videoResults.prepend(videoErrorMessage);
            
            // Clear task ID and stop polling
            currentTaskId = null;
            currentTaskType = null;
            stopPolling();
            break;
            
        case 'ERROR':
            console.error('Processing error state:', status);
            // Show error message
            alert('An error occurred during processing. Please try again or contact support.');
            
            // Clear task ID and stop polling
            currentTaskId = null;
            currentTaskType = null;
            stopPolling();
            break;
            
        default:
            console.warn('Unexpected status:', status);
    }
}

// Display analysis results with flattened data structure
function displayAnalysisResults(data) {
    // Update result display elements
    const resultBpm = document.getElementById('result-bpm');
    const resultTotalBeats = document.getElementById('result-total-beats');
    const resultDuration = document.getElementById('result-duration');
    const resultMeter = document.getElementById('result-meter');
    
    if (resultBpm) {
        resultBpm.textContent = data.bpm ? `${Number(data.bpm).toFixed(1)} BPM` : 'N/A';
    }
    
    if (resultTotalBeats) {
        resultTotalBeats.textContent = data.totalBeats || 'N/A';
    }
    
    if (resultDuration) {
        resultDuration.textContent = data.duration ? `${Number(data.duration).toFixed(1)}s` : 'N/A';
    }
    
    if (resultMeter) {
        resultMeter.textContent = data.detectedMeter || 'N/A';
    }
}

// Display video - simplified as we just need the file ID
function displayVideo() {
    console.log('displayVideo called');
    
    const resultVideo = document.getElementById('result-video');
    if (!resultVideo) {
        console.error('Video element not found');
        return;
    }
    
    console.log('Setting video source to:', `/download/${currentFileId}`);
    
    // Set the video source
    resultVideo.src = `/download/${currentFileId}`;
    
    // Add event listeners to debug video loading
    resultVideo.addEventListener('loadstart', () => console.log('Video loadstart event fired'));
    resultVideo.addEventListener('loadeddata', () => console.log('Video loadeddata event fired'));
    resultVideo.addEventListener('canplay', () => console.log('Video canplay event fired'));
    resultVideo.addEventListener('error', (e) => console.error('Video error event:', e.target.error));
    
    // Force the video to load
    resultVideo.load();
    console.log('Video load() called');
    
    // Try to play the video automatically
    resultVideo.play().then(() => {
        console.log('Video playback started');
    }).catch(err => {
        console.warn('Auto-play failed:', err);
    });
    
    // Update download link
    const downloadLink = document.querySelector('a[download]');
    if (downloadLink) {
        downloadLink.href = `/download/${currentFileId}`;
        console.log('Download link updated');
    }
    
    // Make sure video container is visible
    const videoResults = document.getElementById('video-results');
    if (videoResults) {
        videoResults.classList.remove('hidden');
        console.log('Video results container made visible');
    }
}

// Stop polling for status updates
function stopPolling() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
        console.log('Polling stopped');
    }
}

// Start polling for status updates
function startStatusPolling() {
    // Clear any existing interval
    stopPolling();
    
    // Set up new polling interval (every 2 seconds)
    statusCheckInterval = setInterval(checkStatus, 2000);
    
    // Do an immediate check
    checkStatus();
}

// Check the current status of the file
function checkStatus() {
    if (!currentTaskId) {
        console.log('No task ID available for status check');
        stopPolling();
        return;
    }
    
    console.log('Checking task status for task:', currentTaskId);
    
    // Fetch the current task status
    fetch(`/task/${currentTaskId}`)
        .then(response => {
            if (!response.ok) {
                if (response.status === 404) {
                    console.log('Task not found, stopping polling');
                    stopPolling();
                }
                throw new Error('Failed to fetch task status');
            }
            return response.json();
        })
        .then(data => {
            console.log('Task status:', data);
            
            // If task is completed, reload the page
            if (data.state === 'SUCCESS' || data.state === 'FAILURE' || data.state === 'ERROR') {
                console.log('Task completed, reloading page');
                window.location.reload();
                return;
            }
            
            // Update progress bar
            updateProgressBar(data);
        })
        .catch(error => {
            console.error('Error checking task status:', error);
        });
}

// Update progress bar based on task data
function updateProgressBar(taskData) {
    // Determine which progress bar to update
    const selector = currentTaskType === 'video_generation' ? 
        '#video-progress' : 
        '#analysis-progress';
    
    const progressContainer = document.querySelector(selector);
    if (!progressContainer) return;
    
    const progressBar = progressContainer.querySelector('.progress-fill');
    if (!progressBar) return;
    
    // Extract progress value
    let progressValue = 0;
    let progressStatus = '';
    
    // Check task data for progress information
    if (taskData.progress) {
        if (typeof taskData.progress === 'object') {
            progressValue = taskData.progress.percent || 0;
            progressStatus = taskData.progress.status || '';
        } else {
            progressValue = taskData.progress;
        }
    } else if (taskData.result && taskData.result.progress) {
        if (typeof taskData.result.progress === 'object') {
            progressValue = taskData.result.progress.percent || 0;
            progressStatus = taskData.result.progress.status || '';
        } else {
            progressValue = taskData.result.progress;
        }
    }
    
    // Update the progress bar
    progressBar.style.width = `${progressValue}%`;
    
    // Update progress text if exists
    const progressText = progressContainer.querySelector('p');
    if (progressText) {
        if (progressStatus) {
            // Use the detailed status message from the progress field without percentage
            progressText.textContent = progressStatus;
        } else {
            // Fallback message if no detailed status is available
            if (currentTaskType === 'video_generation') {
                progressText.textContent = 'Generating video...';
            } else {
                progressText.textContent = 'Analyzing beats...';
            }
        }
    }
}

// Setup debug panel
function setupDebugPanel() {
    const debugPanel = document.getElementById('debug-panel');
    if (!debugPanel) return;
    
    debugPanel.style.display = 'block';
    debugPanel.style.backgroundColor = 'rgba(0,0,0,0.8)';
    debugPanel.style.color = '#fff';
    debugPanel.style.padding = '10px';
    debugPanel.style.margin = '20px 0';
    debugPanel.style.borderRadius = '5px';
    debugPanel.style.fontFamily = 'monospace';
    debugPanel.style.fontSize = '14px';
    debugPanel.style.maxHeight = '300px';
    debugPanel.style.overflow = 'auto';
    
    updateDebugPanel();
    
    // Expose the update function globally so it can be called from elsewhere
    window.updateDebugPanel = updateDebugPanel;
}

// Update debug panel with current state information
function updateDebugPanel() {
    const debugPanel = document.getElementById('debug-panel');
    if (!debugPanel) return;
    
    // Get data from initialFileData
    const data = window.initialFileData || {};
    
    // Build debug content
    let debugContent = `
        <h3>Debug Information</h3>
        <div><strong>File ID:</strong> ${currentFileId}</div>
        <div><strong>Status:</strong> ${data.status}</div>
        <div><strong>Polling:</strong> ${statusCheckInterval ? 'Active' : 'Inactive'}</div>
        <div><strong>Current Task:</strong> ${currentTaskId || 'None'} (${currentTaskType || 'None'})</div>
        <div><strong>BEAT_COUNTER_APP_DIR:</strong> ${data.appDir || 'Not Set'}</div>
        
        <div><strong>BPM:</strong> ${data.bpm}</div>
        <div><strong>Total Beats:</strong> ${data.totalBeats}</div>
        <div><strong>Duration:</strong> ${data.duration}s</div>
        <div><strong>Meter:</strong> ${data.detectedMeter}</div>
        
        <details>
            <summary style="cursor:pointer">Raw Data</summary>
            <pre style="max-height:150px;overflow:auto">${JSON.stringify(data, null, 2)}</pre>
        </details>
    `;
    
    // Display in debug panel
    debugPanel.innerHTML = debugContent;
}