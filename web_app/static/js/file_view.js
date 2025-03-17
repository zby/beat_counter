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
        
        // Validate file ID before setting it
        const fileId = fileInfo.fileId;
        if (!fileId || typeof fileId !== 'string' || fileId.length !== 36) {
            console.error('Invalid file ID format:', fileId);
            // Don't set currentFileId so polling won't start
            return;
        }
        
        // Set up initialFileData for global access
        window.initialFileData = {
            fileId: fileId,
            status: fileInfo.status
        };
        
        // Set current file ID for use in API calls
        currentFileId = fileId;
        
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
    
    // Get task data
    const beatTask = statusData.beat_detection_task || {};
    const videoTask = statusData.video_generation_task || {};
    console.log('Beat task state:', beatTask.state, 'Video task state:', videoTask.state);
    
    // Handle based on status
    switch (status) {
        case 'ANALYZING':
            console.log('Showing analysis progress UI');
            // Show analysis progress
            if (analysisSection) analysisSection.classList.remove('hidden');
            if (analysisProgress) analysisProgress.classList.remove('hidden');
            if (analysisResults) analysisResults.classList.add('hidden');
            
            // Update progress bar
            updateProgressBar(beatTask);
            
            // Start polling
            startStatusPolling();
            break;
            
        case 'ANALYZED':
            console.log('Showing analysis results UI');
            // Show analysis results
            if (analysisSection) analysisSection.classList.remove('hidden');
            if (analysisProgress) analysisProgress.classList.add('hidden');
            if (analysisResults) analysisResults.classList.remove('hidden');
            
            // Clear any existing error messages
            if (analysisResults) {
                const existingErrors = analysisResults.querySelectorAll('.error-message');
                existingErrors.forEach(error => error.remove());
            }
            
            // Display results
            displayAnalysisResults(statusData);
            
            // Check if we have valid stats based on task state
            if (beatTask.state === 'SUCCESS') {
                const stats = getStats(beatTask);
                if (Object.keys(stats).length === 0) {
                    // Show error for no stats even though task was successful
                    const errorMessage = document.createElement('div');
                    errorMessage.className = 'error-message';
                    errorMessage.textContent = 'Error: Beat analysis completed but no statistics were found.';
                    if (analysisResults) analysisResults.prepend(errorMessage);
                }
            } else if (beatTask.state === 'FAILURE' || beatTask.state === 'ERROR') {
                // Show error for failed task
                const errorMessage = document.createElement('div');
                errorMessage.className = 'error-message';
                errorMessage.textContent = `Error: Beat analysis failed with state "${beatTask.state}".`;
                if (analysisResults) analysisResults.prepend(errorMessage);
            }
            break;
            
        case 'ANALYZING_FAILURE':
            // Show error for failed beat analysis
            const analysisErrorMessage = document.createElement('div');
            analysisErrorMessage.className = 'error-message';
            analysisErrorMessage.textContent = 'Error: Beat analysis failed.';
            if (analysisResults) analysisResults.prepend(analysisErrorMessage);
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
            
            // Update video progress bar
            updateVideoProgressBar(videoTask);
            
            // Start polling
            startStatusPolling();
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
            
            // Only display video if task was successful
            if (videoTask.state === 'SUCCESS') {
                console.log('Video task successful, displaying video');
                displayVideo(statusData);
            } else {
                console.warn('Video task not successful, showing error');
                if (videoResults) {
                    const errorMessage = document.createElement('div');
                    errorMessage.className = 'error-message';
                    errorMessage.textContent = `Error: Video generation failed with state "${videoTask.state}".`;
                    videoResults.prepend(errorMessage);
                }
            }
            break;
            
        case 'VIDEO_ERROR':
            // Show error for failed video generation
            const videoErrorMessage = document.createElement('div');
            videoErrorMessage.className = 'error-message';
            videoErrorMessage.textContent = 'Error: Video generation failed.';
            if (videoResults) videoResults.prepend(videoErrorMessage);
            break;
            
        case 'ERROR':
            console.error('Processing error state:', status);
            // Show error message with more details
            const errorDetails = [];
            if (beatTask.state === 'FAILURE' || beatTask.state === 'ERROR') {
                errorDetails.push(`Beat analysis: ${beatTask.state}`);
            }
            if (videoTask.state === 'FAILURE' || videoTask.state === 'ERROR') {
                errorDetails.push(`Video generation: ${videoTask.state}`);
            }
            const errorMessage = 'An error occurred during processing: ' +
                (errorDetails.length ? errorDetails.join(', ') : 'Unknown error');
            alert(errorMessage + '. Please try again or contact support.');
            break;
            
        default:
            console.warn('Unexpected status:', status);
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
    
    // Check if video task was successful
    if (videoTask.state !== 'SUCCESS') {
        console.warn('Video task not successful, state:', videoTask.state);
        return;
    }
    
    // Get video file from the task result
    const videoFile = videoTask.result?.video_file;
    console.log('Video file path:', videoFile);
    
    // Update video element if we have a video file
    if (videoFile) {
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
    } else {
        console.warn('No video file found in the successful task result');
    }
}

// Extract stats from beat detection task
function getStats(beatTask) {
    // First, check if we have a valid task with a successful state
    if (!beatTask || beatTask.state !== 'SUCCESS') {
        console.log('Beat task not successful:', beatTask?.state);
        return {};
    }
    
    // If successful, check for stats directly
    if (beatTask.result && beatTask.result.stats) {
        console.log('Found stats in beat task result');
        return beatTask.result.stats;
    }
    
    console.log('No stats found in successful beat task');
    return {};
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
    if (!currentFileId) {
        console.error('No file ID available for status check');
        return;
    }
    
    console.log('Checking status for file:', currentFileId);
    
    // Fetch the current status
    fetch(`/status/${currentFileId}`)
        .then(response => {
            // If we get a 404, the file doesn't exist, so stop polling
            if (response.status === 404) {
                console.error(`File with ID ${currentFileId} not found. Stopping polling.`);
                stopPolling();
                throw new Error('File not found');
            }
            
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
            
            // Extract relevant information
            const oldStatus = window.initialFileData.status.status;
            const newStatus = data.status;
            const beatTask = data.beat_detection_task || {};
            const videoTask = data.video_generation_task || {};
            
            console.log('Status check result:', { 
                oldStatus, 
                newStatus, 
                beatTaskState: beatTask.state, 
                videoTaskState: videoTask.state 
            });
            
            // Determine if UI needs updating
            let shouldUpdateUI = false;
            
            // Update UI in these cases:
            // 1. If top-level status changed
            if (oldStatus !== newStatus) {
                console.log('Status changed from', oldStatus, 'to', newStatus);
                shouldUpdateUI = true;
            }
            // 2. If we're in a polling state
            else if (['ANALYZING', 'GENERATING_VIDEO'].includes(newStatus)) {
                console.log('In polling state:', newStatus);
                shouldUpdateUI = true;
            }
            // 3. If status is ANALYZED or COMPLETED (to ensure results are displayed)
            else if (['ANALYZED', 'COMPLETED'].includes(newStatus)) {
                console.log('In result display state:', newStatus);
                shouldUpdateUI = true;
            }
            
            // Update UI if needed
            if (shouldUpdateUI) {
                console.log('Updating UI for status:', newStatus);
                handleStatus(data);
            }
            
            // Determine if polling should stop
            let shouldStopPolling = false;
            
            // Stop polling logic
            if (newStatus === 'ANALYZED') {
                // For ANALYZED, stop polling if beat task is completed successfully with stats
                if (beatTask.state === 'SUCCESS' && beatTask.result && beatTask.result.stats) {
                    stopPolling();
                    console.log('Beat analysis completed successfully with stats');
                } else {
                    console.log('Beat analysis completed but no stats found, continuing polling');
                }
            } else if (['FAILURE', 'ERROR'].includes(beatTask.state)) {
                console.log('Beat analysis failed, stopping polling');
                shouldStopPolling = true;
            } else if (newStatus === 'COMPLETED' || newStatus === 'VIDEO_ERROR') {
                // Stop polling when video generation is completed or fails
                console.log(`Video generation ${newStatus === 'COMPLETED' ? 'completed' : 'failed'}, stopping polling`);
                shouldStopPolling = true;
            }
            
            // Stop polling if determined
            if (shouldStopPolling) {
                stopPolling();
            }
        })
        .catch(error => {
            console.error('Error checking status:', error);
            
            // If polling is for a non-existent file, stop polling
            if (error.message === 'File not found') {
                console.log('Stopping polling for non-existent file');
                // Already cleared interval in the 404 handler
            }
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
    
    // Get status data
    const statusData = window.initialFileData?.status || {};
    const beatTask = statusData.beat_detection_task || {};
    const videoTask = statusData.video_generation_task || {};
    
    // Format states more clearly
    const getStateColor = (state) => {
        if (!state) return 'gray';
        switch(state) {
            case 'SUCCESS': return 'green';
            case 'FAILURE': 
            case 'ERROR': return 'red';
            default: return 'orange';
        }
    };
    
    // Create a nicer task status display
    const formatTaskStatus = (task, name) => {
        if (!task || Object.keys(task).length === 0) {
            return `<div><strong>${name}:</strong> <span style="color: gray;">Not Started</span></div>`;
        }
        
        const state = task.state || 'UNKNOWN';
        const stateColor = getStateColor(state);
        let progressInfo = '';
        
        if (task.progress) {
            const percent = typeof task.progress === 'object' && task.progress.percent 
                ? task.progress.percent 
                : task.progress;
            const status = typeof task.progress === 'object' && task.progress.status 
                ? task.progress.status 
                : '';
            progressInfo = `<div>Progress: ${Math.round(percent)}%${status ? ` (${status})` : ''}</div>`;
        }
        
        return `
            <div style="margin-bottom: 5px;">
                <strong>${name}:</strong> <span style="color: ${stateColor};">${state}</span>
                ${progressInfo}
            </div>
        `;
    };
    
    debugPanel.innerHTML = `
        <h3>Debug Info</h3>
        <div style="margin: 10px; padding: 10px; background: rgba(50,50,50,0.5);">
            <h4>Application Status</h4>
            <div style="margin-bottom: 10px;">
                <div><strong>Status:</strong> ${statusData.status || 'N/A'}</div>
                <div><strong>File ID:</strong> ${currentFileId || 'N/A'}</div>
                ${formatTaskStatus(beatTask, 'Beat Detection')}
                ${formatTaskStatus(videoTask, 'Video Generation')}
            </div>
            
            <h4>Detailed Task Data</h4>
            <div style="font-size: 0.9em;">
                <details>
                    <summary>Beat Detection Task</summary>
                    <pre style="white-space: pre-wrap; word-wrap: break-word;">${JSON.stringify(beatTask, null, 2)}</pre>
                </details>
                <details>
                    <summary>Video Generation Task</summary>
                    <pre style="white-space: pre-wrap; word-wrap: break-word;">${JSON.stringify(videoTask, null, 2)}</pre>
                </details>
            </div>
        </div>
    `;
}