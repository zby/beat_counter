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
   - When status is 'ERROR' or 'FAILURE' - polling stops

3. Error states (no polling):
   - When status is 'ANALYZED' but no valid stats are found - this is an error state, polling does not start
*/

// Global state
let currentFileId = null;
let statusCheckInterval = null;
let debugMode = true;

// Initialize the application when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing Beat Detection App');
    
    // Get data from the app-data div
    const appDataElement = document.getElementById('app-data');
    if (!appDataElement) {
        console.error('App data element not found');
        return;
    }
    
    try {
        // Parse the JSON data
        const fileInfo = JSON.parse(appDataElement.getAttribute('data-file-info') || '{}');
        
        // Set up initialFileData for global access
        window.initialFileData = {
            fileId: fileInfo.fileId,
            status: fileInfo.status
        };
        
        // Set current file ID for use in API calls
        currentFileId = fileInfo.fileId;
        
        // Initialize UI based on current status
        handleStatus(window.initialFileData.status);
        
        // Set up event listeners
        setupEventListeners();
        
        // Set up debug panel if enabled
        if (debugMode) {
            setupDebugPanel();
        }
    } catch (error) {
        console.error('Error initializing application:', error);
    }
});

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
    if (!statusData) return;
    
    const status = statusData.status;
    console.log('Status:', status);
    
    // Get DOM elements
    const analysisSection = document.getElementById('analysis-section');
    const analysisProgress = document.getElementById('analysis-progress');
    const analysisResults = document.getElementById('analysis-results');
    const analysisButtons = document.getElementById('analysis-buttons');
    const videoSection = document.getElementById('video-section');
    const videoProgress = document.getElementById('video-progress');
    const videoResults = document.getElementById('video-results');
    
    // Handle based on status
    switch (status) {
        case 'ANALYZING':
            // Show analysis progress
            if (analysisSection) analysisSection.classList.remove('hidden');
            if (analysisProgress) analysisProgress.classList.remove('hidden');
            if (analysisResults) analysisResults.classList.add('hidden');
            
            // Update progress bar
            updateProgressBar(statusData.beat_detection_task);
            
            // Start polling
            startStatusPolling();
            break;
            
        case 'ANALYZED':
            // Show analysis results
            if (analysisSection) analysisSection.classList.remove('hidden');
            if (analysisProgress) analysisProgress.classList.add('hidden');
            if (analysisResults) analysisResults.classList.remove('hidden');
            
            // Display results
            displayAnalysisResults(statusData);
            
            // Check if we have valid stats
            const beatTask = statusData.beat_detection_task || {};
            const stats = getStats(beatTask);
            
            if (Object.keys(stats).length === 0) {
                // Show error for no stats
                const errorMessage = document.createElement('div');
                errorMessage.className = 'error-message';
                errorMessage.textContent = 'Error: Beat analysis completed but no statistics were found.';
                if (analysisResults) analysisResults.prepend(errorMessage);
            }
            break;
            
        case 'GENERATING_VIDEO':
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
            
            // Update video progress bar
            updateVideoProgressBar(statusData.video_generation_task);
            
            // Start polling
            startStatusPolling();
            break;
            
        case 'COMPLETED':
            // Show analysis results and video
            if (analysisSection) {
                analysisSection.classList.remove('hidden');
                analysisSection.classList.add('with-video');
            }
            if (analysisProgress) analysisProgress.classList.add('hidden');
            if (analysisResults) analysisResults.classList.remove('hidden');
            if (analysisButtons) analysisButtons.classList.add('hidden');
            
            // Show video results
            if (videoSection) videoSection.classList.remove('hidden');
            if (videoProgress) videoProgress.classList.add('hidden');
            if (videoResults) videoResults.classList.remove('hidden');
            
            // Display results
            displayAnalysisResults(statusData);
            displayVideo(statusData);
            break;
            
        case 'ERROR':
        case 'FAILURE':
            // Show error message
            alert('An error occurred during processing. Please try again or contact support.');
            break;
    }
}

// Update progress bar for beat detection
function updateProgressBar(beatTask) {
    if (!beatTask || !beatTask.progress) return;
    
    const progressBar = document.querySelector('#analysis-progress .progress-fill');
    if (!progressBar) return;
    
    // Get progress value
    const progressValue = typeof beatTask.progress === 'object' && beatTask.progress.percent ? 
        beatTask.progress.percent : beatTask.progress;
    
    // Update progress bar
    progressBar.style.width = `${progressValue}%`;
    
    // Update text if exists
    const progressText = document.querySelector('#analysis-progress .progress-text');
    if (progressText) {
        progressText.textContent = `${Math.round(progressValue)}%`;
    }
}

// Update progress bar for video generation
function updateVideoProgressBar(videoTask) {
    if (!videoTask || !videoTask.progress) return;
    
    const progressBar = document.querySelector('#video-progress .progress-fill');
    if (!progressBar) return;
    
    // Get progress value
    const progressValue = typeof videoTask.progress === 'object' && videoTask.progress.percent ? 
        videoTask.progress.percent : videoTask.progress;
    
    // Update progress bar
    progressBar.style.width = `${progressValue}%`;
    
    // Update text if exists
    const progressText = document.querySelector('#video-progress .progress-text');
    if (progressText) {
        progressText.textContent = `${Math.round(progressValue)}%`;
    }
}

// Display analysis results
function displayAnalysisResults(statusData) {
    // Get beat detection task data
    const beatTask = statusData.beat_detection_task || {};
    
    // Get stats from the beat detection task
    const stats = getStats(beatTask);
    
    // Update result display elements
    const resultBpm = document.getElementById('result-bpm');
    const resultTotalBeats = document.getElementById('result-total-beats');
    const resultDuration = document.getElementById('result-duration');
    const resultMeter = document.getElementById('result-meter');
    
    if (resultBpm) {
        resultBpm.textContent = stats.bpm ? `${Number(stats.bpm).toFixed(1)} BPM` : 'N/A';
    }
    
    if (resultTotalBeats) {
        resultTotalBeats.textContent = stats.total_beats || 'N/A';
    }
    
    if (resultDuration) {
        resultDuration.textContent = stats.duration ? `${Number(stats.duration).toFixed(1)}s` : 'N/A';
    }
    
    if (resultMeter) {
        resultMeter.textContent = stats.detected_meter || 'N/A';
    }
}

// Display video
function displayVideo(statusData) {
    console.log('displayVideo called with status:', statusData.status);
    
    const resultVideo = document.getElementById('result-video');
    if (!resultVideo) {
        console.error('Video element not found');
        return;
    }
    
    // Get video generation task data
    const videoTask = statusData.video_generation_task || {};
    console.log('Video task state:', videoTask.state);
    
    // Get video file from the task result
    const videoFile = videoTask.result?.video_file;
    console.log('Video file info:', videoFile);
    
    // Update video element if we have a video file
    if (videoFile) {
        console.log('Setting video source to:', `/download/${currentFileId}`);
        
        // Set the video source
        resultVideo.src = `/download/${currentFileId}`;
        
        // Add event listeners to debug video loading
        resultVideo.addEventListener('loadstart', () => console.log('Video loadstart event fired'));
        resultVideo.addEventListener('loadeddata', () => console.log('Video loadeddata event fired'));
        resultVideo.addEventListener('canplay', () => console.log('Video canplay event fired'));
        resultVideo.addEventListener('error', (e) => console.error('Video error event:', e));
        
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
    } else {
        console.warn('No video file found in the task result');
    }
}

// Extract stats from beat detection task
function getStats(beatTask) {
    // Check if we have a result with stats
    if (beatTask && beatTask.result && beatTask.result.stats) {
        return beatTask.result.stats;
    }
   
    return {};
}

// Start polling for status updates
function startStatusPolling() {
    // Clear any existing interval
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
    }
    
    // Set up new polling interval (every 2 seconds)
    statusCheckInterval = setInterval(checkStatus, 2000);
    
    // Do an immediate check
    checkStatus();
}

// Check the current status of the file
function checkStatus() {
    if (!currentFileId) {
        console.error('No file ID available for status check');
        return;
    }
    
    console.log('Checking status for file:', currentFileId);
    
    // Fetch the current status
    fetch(`/status/${currentFileId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to fetch status');
            }
            return response.json();
        })
        .then(data => {
            // Update global state for debug panel
            window.initialFileData = {
                fileId: currentFileId,
                status: data
            };
            
            // Update debug panel if enabled
            if (debugMode) {
                updateDebugPanel();
            }
            
            // Handle status change
            const oldStatus = window.initialFileData.status.status;
            const newStatus = data.status;
            
            console.log('Status check result:', { oldStatus, newStatus });
            
            // Always update UI in these cases:
            // 1. If status changed
            // 2. If we're in a polling state (ANALYZING, GENERATING_VIDEO)
            // 3. If status is ANALYZED or COMPLETED (to ensure results are displayed)
            if (oldStatus !== newStatus || 
                ['ANALYZING', 'GENERATING_VIDEO', 'ANALYZED', 'COMPLETED'].includes(newStatus)) {
                console.log('Updating UI for status:', newStatus);
                handleStatus(data);
            }
            
            // Check if we should stop polling
            if (newStatus === 'ANALYZED') {
                // For ANALYZED, only stop polling if we have valid stats
                const beatTask = data.beat_detection_task || {};
                const stats = getStats(beatTask);
                
                if (Object.keys(stats).length > 0) {
                    console.log('Found valid stats, stopping polling');
                    clearInterval(statusCheckInterval);
                    statusCheckInterval = null;
                } else {
                    console.log('No valid stats found, continuing polling');
                }
            } 
            // Always stop polling for these states
            else if (['COMPLETED', 'ERROR', 'FAILURE'].includes(newStatus)) {
                console.log('Terminal state reached, stopping polling');
                clearInterval(statusCheckInterval);
                statusCheckInterval = null;
            }
        })
        .catch(error => {
            console.error('Error checking status:', error);
        });
}

// Setup debug panel
function setupDebugPanel() {
    const debugPanel = document.getElementById('debug-panel');
    if (debugPanel) {
        debugPanel.style.display = 'block';
    }
    updateDebugPanel();
}

// Update debug panel
function updateDebugPanel() {
    const debugPanel = document.getElementById('debug-panel');
    if (!debugPanel) return;
    
    debugPanel.innerHTML = `
        <h3>Debug Info</h3>
        <div style="margin: 10px; padding: 10px; background: rgba(50,50,50,0.5);">
            <h4>Current State:</h4>
            <pre style="white-space: pre-wrap; word-wrap: break-word;">
Status: ${window.initialFileData?.status?.status || 'N/A'}
File ID: ${currentFileId || 'N/A'}
Beat Task: ${JSON.stringify(window.initialFileData?.status?.beat_detection_task || {}, null, 2)}
Video Task: ${JSON.stringify(window.initialFileData?.status?.video_generation_task || {}, null, 2)}
            </pre>
        </div>
    `;
}