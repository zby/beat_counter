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

// Global variables for DOM elements
let analysisSection, analysisProgress, analysisResults, confirmButton, cancelButton;
let videoSection, videoProgress, videoResults, resultVideo, restartButton;
let resultBpm, resultTotalBeats, resultDuration, resultMeter;

// Global state
let currentFileId = null;
let statusCheckInterval = null;

// Debug logging
let debugMode = true; // Set to true to enable debug panel
const debugMessages = [];

// Override console methods to capture logs
if (debugMode) {
    const originalLog = console.log;
    const originalWarn = console.warn;
    const originalError = console.error;
    
    console.log = function() {
        debugMessages.push({type: 'log', message: Array.from(arguments).join(' ')});
        updateDebugPanel();
        originalLog.apply(console, arguments);
    };
    
    console.warn = function() {
        debugMessages.push({type: 'warn', message: Array.from(arguments).join(' ')});
        updateDebugPanel();
        originalWarn.apply(console, arguments);
    };
    
    console.error = function() {
        debugMessages.push({type: 'error', message: Array.from(arguments).join(' ')});
        updateDebugPanel();
        originalError.apply(console, arguments);
    };
    
    // Make sure the debug panel is visible when debug mode is enabled
    document.addEventListener('DOMContentLoaded', function() {
        const debugPanel = document.getElementById('debug-panel');
        if (debugPanel) {
            debugPanel.style.display = 'block';
        }
    });
}

// Update the debug panel with latest messages
function updateDebugPanel() {
    if (!debugMode) return;
    
    const debugPanel = document.getElementById('debug-panel');
    if (!debugPanel) return;
    
    // Keep only the last 20 messages
    const recentMessages = debugMessages.slice(-20);
    
    // Clear and rebuild the debug panel
    debugPanel.innerHTML = '<h3>Debug Messages</h3>';
    
    // Add current state section
    const stateSection = document.createElement('div');
    stateSection.className = 'debug-state';
    stateSection.innerHTML = `
        <h4>Current State:</h4>
        <pre>status: ${window.initialFileData?.status?.status || 'N/A'}
fileId: ${currentFileId || 'N/A'}
beat_task: ${JSON.stringify(window.initialFileData?.status?.beat_detection_task || {}, null, 2)}
video_task: ${JSON.stringify(window.initialFileData?.status?.video_generation_task || {}, null, 2)}
</pre>
    `;
    debugPanel.appendChild(stateSection);
    
    // Add messages section
    const messagesSection = document.createElement('div');
    messagesSection.className = 'debug-messages';
    messagesSection.innerHTML = '<h4>Recent Messages:</h4>';
    recentMessages.forEach(msg => {
        const msgElement = document.createElement('div');
        msgElement.className = `debug-message debug-${msg.type}`;
        msgElement.textContent = msg.message;
        messagesSection.appendChild(msgElement);
    });
    debugPanel.appendChild(messagesSection);
}

// Initialize the application
function init() {
    console.log('Initializing application...');
    
    // Get DOM elements after the DOM is loaded
    analysisSection = document.getElementById('analysis-section');
    analysisProgress = document.getElementById('analysis-progress');
    analysisResults = document.getElementById('analysis-results');
    confirmButton = document.getElementById('confirm-button');
    cancelButton = document.getElementById('cancel-button');
    videoSection = document.getElementById('video-section');
    videoProgress = document.getElementById('video-progress');
    videoResults = document.getElementById('video-results');
    resultVideo = document.getElementById('result-video');
    restartButton = document.getElementById('restart-button');

    // Result display elements
    resultBpm = document.getElementById('result-bpm');
    resultTotalBeats = document.getElementById('result-total-beats');
    resultDuration = document.getElementById('result-duration');
    resultMeter = document.getElementById('result-meter');
    
    // Log DOM elements to verify they exist
    console.log('DOM Elements loaded:');
    console.log('- analysisSection:', analysisSection);
    console.log('- analysisProgress:', analysisProgress);
    console.log('- analysisResults:', analysisResults);
    console.log('- confirmButton:', confirmButton);
    console.log('- cancelButton:', cancelButton);
    console.log('- videoSection:', videoSection);
    
    // Ensure the debug panel is created and visible
    if (!document.getElementById('debug-panel')) {
        const debugPanel = document.createElement('div');
        debugPanel.id = 'debug-panel';
        debugPanel.className = 'debug-panel';
        document.body.appendChild(debugPanel);
        
        // Add some basic styling
        const style = document.createElement('style');
        style.textContent = `
            .debug-panel {
                position: fixed;
                bottom: 0;
                right: 0;
                width: 400px;
                max-height: 80vh;
                overflow-y: auto;
                background: rgba(0,0,0,0.9);
                color: white;
                padding: 10px;
                font-family: monospace;
                z-index: 9999;
                border-top-left-radius: 5px;
            }
            .debug-state {
                margin: 10px 0;
                padding: 10px;
                background: rgba(50,50,50,0.5);
                border-radius: 4px;
            }
            .debug-state pre {
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            .debug-messages {
                margin-top: 10px;
            }
            .debug-message {
                margin: 5px 0;
                padding: 5px;
                border-left: 3px solid #ccc;
                font-size: 12px;
            }
            .debug-log { border-color: #4CAF50; }
            .debug-warn { border-color: #FFC107; color: #FFC107; }
            .debug-error { border-color: #F44336; color: #F44336; }
        `;
        document.head.appendChild(style);
    }
    
    // Set up event listeners
    setupEventListeners();
    
    // Initialize with file data from the server
    if (window.initialFileData) {
        // Add very explicit debugging
        console.log('%c DEBUGGING INITIALIZATION', 'background: #222; color: #bada55; font-size: 20px;');
        console.log('Initial file data:', window.initialFileData);
        console.log('File ID:', window.initialFileData.fileId);
        console.log('Status object:', window.initialFileData.status);
        
        // Log beat and video tasks specifically
        const beatTask = window.initialFileData.status.beat_detection_task;
        const videoTask = window.initialFileData.status.video_generation_task;
        console.log('Beat detection task:', beatTask);
        console.log('Video generation task:', videoTask);
       
        currentFileId = window.initialFileData.fileId;
        
        // Log the overall status explicitly
        const overallStatus = window.initialFileData.status.status;
        console.log('Overall status:', overallStatus);
        
        // Check if the status is undefined and log a warning
        if (typeof overallStatus === 'undefined') {
            console.warn('Overall status is undefined!');
        }
        
        // Ensure the status is used correctly in the logic
        if (overallStatus === 'COMPLETED') {
            console.log('Status is COMPLETED, updating UI accordingly.');
            // Show both analysis results and completed video
            analysisSection.classList.remove('hidden');
            analysisSection.classList.add('with-video');
            // Hide both progress bars
            analysisProgress.classList.add('hidden');
            videoProgress.classList.add('hidden');
            // Show both results sections
            analysisResults.classList.remove('hidden');
            document.getElementById('analysis-buttons').classList.add('hidden');
            displayAnalysisResults(window.initialFileData.status);
            
            videoSection.classList.remove('hidden');
            videoResults.classList.remove('hidden');
            displayVideo(window.initialFileData.status);
        }
        
        // Handle different file states
        const status = window.initialFileData.status.status;
        
        if (status === 'ANALYZING') {
            // Show analysis section with progress
            analysisSection.classList.remove('hidden');
            analysisProgress.classList.remove('hidden');
            analysisResults.classList.add('hidden');
            
            // If we have progress information, update the progress bar
            const beatTask = window.initialFileData.status.beat_detection_task;
            if (beatTask && beatTask.progress) {
                // Handle both formats of progress data
                const progressValue = typeof beatTask.progress === 'object' && beatTask.progress.percent ? 
                    beatTask.progress.percent : beatTask.progress;
                updateProgressBar(analysisProgress.querySelector('.progress-fill'), progressValue);
            }
            
            // Start polling for status updates
            startStatusPolling();
        } 
        else if (status === 'ANALYZED') {
            // Show analysis results
            analysisSection.classList.remove('hidden');
            analysisProgress.classList.add('hidden');
            analysisResults.classList.remove('hidden');
            
            // Get beat detection task data if available
            const beatTask = window.initialFileData.status.beat_detection_task || {};
            
            // Get stats from the beat detection task
            const stats = getStats(beatTask);
            
            // Only start polling if we don't have valid stats yet
            if (Object.keys(stats).length === 0) {
                console.log('Status is ANALYZED but no valid stats found. Starting polling to get stats.');
                startStatusPolling();
            } else {
                console.log('Status is ANALYZED and we have valid stats. No need to poll for updates.');
            }

            // Display the analysis results
            displayAnalysisResults(window.initialFileData.status);
            
            // If we don't have valid stats, show an error message instead of starting polling
            if (Object.keys(stats).length === 0) {
                console.error('Status is ANALYZED but no valid stats found. This is an error state.');
                
                // Show an error message to the user
                const errorMessage = document.createElement('div');
                errorMessage.className = 'error-message';
                errorMessage.textContent = 'Error: Beat analysis completed but no statistics were found. Please try again or contact support.';
                analysisResults.prepend(errorMessage);
                
                // Don't start polling as this is an error state
                if (statusCheckInterval) {
                    clearInterval(statusCheckInterval);
                    statusCheckInterval = null;
                }
            }
        } 
        else if (status === 'GENERATING_VIDEO') {
            // Show both analysis results and video generation in progress
            analysisSection.classList.remove('hidden');
            analysisSection.classList.add('with-video');
            analysisProgress.classList.add('hidden');
            analysisResults.classList.remove('hidden');
            document.getElementById('analysis-buttons').classList.add('hidden');
            displayAnalysisResults(window.initialFileData.status);
            
            videoSection.classList.remove('hidden');
            videoProgress.classList.remove('hidden');
            videoResults.classList.add('hidden');
            
            // If we have progress information, update the progress bar
            const videoTask = window.initialFileData.status.video_generation_task;
            if (videoTask && videoTask.progress) {
                // Handle both formats of progress data
                const progressValue = typeof videoTask.progress === 'object' && videoTask.progress.percent ? 
                    videoTask.progress.percent : videoTask.progress;
                updateProgressBar(videoProgress.querySelector('.progress-fill'), progressValue);
            }
            
            // Start polling for status updates
            startStatusPolling();
        } 
        else if (status === 'COMPLETED') {
            // Show both analysis results and completed video
            analysisSection.classList.remove('hidden');
            analysisSection.classList.add('with-video');
            // Hide both progress bars
            analysisProgress.classList.add('hidden');
            videoProgress.classList.add('hidden');
            // Show both results sections
            analysisResults.classList.remove('hidden');
            document.getElementById('analysis-buttons').classList.add('hidden');
            displayAnalysisResults(window.initialFileData.status);
            
            videoSection.classList.remove('hidden');
            videoResults.classList.remove('hidden');
            displayVideo(window.initialFileData.status);
        }
        else if (status === 'ERROR') {
            // Show error message
            alert('An error occurred during processing. Please try again or contact support.');
        }
    }
}

// Setup event listeners for buttons and other interactive elements
function setupEventListeners() {
    console.log('Setting up event listeners...');
    
    // Set up confirm button
    if (confirmButton) {
        console.log('Adding event listener to confirm button');
        confirmButton.addEventListener('click', handleConfirmAnalysis);
    } else {
        console.warn('Confirm button not found');
    }
    
    // Set up cancel button
    if (cancelButton) {
        console.log('Adding event listener to cancel button');
        cancelButton.addEventListener('click', handleCancelAnalysis);
    } else {
        console.warn('Cancel button not found');
    }
    
    // Set up restart button
    if (restartButton) {
        console.log('Adding event listener to restart button');
        restartButton.addEventListener('click', handleRestartAnalysis);
    } else {
        console.warn('Restart button not found');
    }
    
    console.log('Event listeners set up successfully');
}

// Initialize the application when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM fully loaded, initializing application');
    
    // Get data from the app-data div
    const appDataElement = document.getElementById('app-data');
    if (appDataElement) {
        try {
            // Parse the JSON data from the data attributes
            const fileInfo = JSON.parse(appDataElement.getAttribute('data-file-info') || '{}');
            const beatDetection = JSON.parse(appDataElement.getAttribute('data-beat-detection') || '{}');
            const videoGeneration = JSON.parse(appDataElement.getAttribute('data-video-generation') || '{}');
            
            // Set up the initialFileData object
            window.initialFileData = {
                fileId: fileInfo.fileId,
                status: fileInfo.status
            };
            
            console.log('Initialized with file data:', window.initialFileData);
            
            // Initialize the application
            init();
        } catch (error) {
            console.error('Error parsing app data:', error);
        }
    } else {
        console.error('App data element not found');
    }
});

// Handle confirm analysis button click
function handleConfirmAnalysis() {
    console.log('Confirm analysis button clicked');
    if (!currentFileId) {
        console.error('No file ID available');
        return;
    }
    
    // Disable the button to prevent multiple clicks
    confirmButton.disabled = true;
    
    // Send request to confirm analysis
    fetch(`/confirm/${currentFileId}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        if (response.redirected) {
            // If redirected, follow the redirect
            window.location.href = response.url;
        } else if (response.ok) {
            // If successful but not redirected, reload the page
            window.location.reload();
        } else {
            // If error, parse the error message
            return response.json().then(data => {
                throw new Error(data.detail || 'Failed to confirm analysis');
            });
        }
    })
    .catch(error => {
        console.error('Error confirming analysis:', error);
        alert('Error: ' + error.message);
        // Re-enable the button
        confirmButton.disabled = false;
    });
}

// Handle cancel analysis button click
function handleCancelAnalysis() {
    console.log('Cancel analysis button clicked');
    // Redirect to home page
    window.location.href = '/';
}

// Handle restart analysis button click
function handleRestartAnalysis() {
    console.log('Restart analysis button clicked');
    if (!currentFileId) {
        console.error('No file ID available');
        return;
    }
    
    // Redirect to upload page with file ID as parameter
    window.location.href = `/?restart=${currentFileId}`;
}

// Start polling for status updates
function startStatusPolling() {
    console.log('Starting status polling');
    
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
            console.log('Status update received:', data);
            handleStatusUpdate(data);
        })
        .catch(error => {
            console.error('Error checking status:', error);
        });
}

// Handle a status update
function handleStatusUpdate(statusData) {
    // Update the debug panel
    window.initialFileData = {
        fileId: currentFileId,
        status: statusData
    };
    updateDebugPanel();
    
    const status = statusData.status;
    console.log('Handling status update:', status);
    
    // Handle different statuses
    if (status === 'ANALYZING') {
        console.log('Status is ANALYZING, updating progress bar');
        // Update progress bar if we have progress information
        if (analysisSection && analysisProgress && analysisResults) {
            // Show analysis section with progress
            analysisSection.classList.remove('hidden');
            analysisProgress.classList.remove('hidden');
            analysisResults.classList.add('hidden');
            
            const beatTask = statusData.beat_detection_task;
            if (beatTask && beatTask.progress) {
                const progressValue = typeof beatTask.progress === 'object' && beatTask.progress.percent ? 
                    beatTask.progress.percent : beatTask.progress;
                const progressFill = analysisProgress.querySelector('.progress-fill');
                if (progressFill) {
                    updateProgressBar(progressFill, progressValue);
                }
            }
        } else {
            console.warn('DOM elements for analysis not found');
        }
    }
    else if (status === 'ANALYZED') {
        console.log('Status is ANALYZED, showing analysis results');
        // Stop polling and show analysis results
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
        
        if (analysisSection && analysisProgress && analysisResults) {
            // Show analysis results
            analysisSection.classList.remove('hidden');
            analysisProgress.classList.add('hidden');
            analysisResults.classList.remove('hidden');
            
            // Display the analysis results
            displayAnalysisResults(statusData);
        } else {
            console.warn('DOM elements for analysis not found');
        }
    }
    else if (status === 'GENERATING_VIDEO') {
        console.log('Status is GENERATING_VIDEO, updating video progress');
        if (videoSection && videoProgress) {
            // Update progress bar if we have progress information
            const videoTask = statusData.video_generation_task;
            if (videoTask && videoTask.progress) {
                const progressValue = typeof videoTask.progress === 'object' && videoTask.progress.percent ? 
                    videoTask.progress.percent : videoTask.progress;
                const progressFill = videoProgress.querySelector('.progress-fill');
                if (progressFill) {
                    updateProgressBar(progressFill, progressValue);
                }
            }
        } else {
            console.warn('DOM elements for video not found');
        }
    }
    else if (status === 'COMPLETED') {
        console.log('Status is COMPLETED, showing completed video');
        // Stop polling and show completed video
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
        
        if (analysisSection && analysisProgress && analysisResults && 
            videoSection && videoProgress && videoResults) {
            // Show both analysis results and completed video
            analysisSection.classList.remove('hidden');
            analysisSection.classList.add('with-video');
            // Hide both progress bars
            analysisProgress.classList.add('hidden');
            videoProgress.classList.add('hidden');
            // Show both results sections
            analysisResults.classList.remove('hidden');
            
            const analysisButtons = document.getElementById('analysis-buttons');
            if (analysisButtons) {
                analysisButtons.classList.add('hidden');
            }
            
            displayAnalysisResults(statusData);
            
            videoSection.classList.remove('hidden');
            videoResults.classList.remove('hidden');
            displayVideo(statusData);
            
            // Reload the page to ensure everything is up to date
            window.location.reload();
        } else {
            console.warn('DOM elements for completed state not found');
        }
    }
    else if (status === 'ERROR' || status === 'FAILURE') {
        console.log('Status is ERROR or FAILURE, showing error message');
        // Stop polling and show error
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
        
        // Show error message
        alert('An error occurred during processing. Please try again or contact support.');
    }
}

// Update a progress bar
function updateProgressBar(progressElement, percent) {
    if (!progressElement) return;
    
    // Ensure percent is between 0 and 100
    percent = Math.max(0, Math.min(100, percent));
    
    // Update the width of the progress bar
    progressElement.style.width = `${percent}%`;
    
    // Update the text if there's a progress-text element
    const progressText = progressElement.parentElement.querySelector('.progress-text');
    if (progressText) {
        progressText.textContent = `${Math.round(percent)}%`;
    }
}

// Display analysis results
function displayAnalysisResults(statusData) {
    // Get beat detection task data
    const beatTask = statusData.beat_detection_task || {};
    
    // Get stats from the beat detection task
    const stats = getStats(beatTask);
    
    // Update result display elements
    if (resultBpm) resultBpm.textContent = stats.bpm ? `${stats.bpm.toFixed(1)} BPM` : 'N/A';
    if (resultTotalBeats) resultTotalBeats.textContent = stats.total_beats || 'N/A';
    if (resultDuration) resultDuration.textContent = stats.duration ? `${stats.duration.toFixed(1)}s` : 'N/A';
    if (resultMeter) resultMeter.textContent = stats.meter || 'N/A';
}

// Display video
function displayVideo(statusData) {
    // Get video generation task data
    const videoTask = statusData.video_generation_task || {};
    
    // Get video file from the task result
    const videoFile = videoTask.result?.video_file;
    
    // Update video element if we have a video file
    if (videoFile && resultVideo) {
        // Set the video source
        resultVideo.src = `/download/${currentFileId}`;
        resultVideo.load();
        
        // Update download link
        const downloadLink = document.getElementById('download-link');
        if (downloadLink) {
            downloadLink.href = `/download/${currentFileId}`;
        }
    }
}

// Extract stats from beat detection task
function getStats(beatTask) {
    const stats = {};
    console.log('Beat task:', beatTask);
    
    // Check if we have a result
    if (beatTask && beatTask.result && beatTask.result.stats) {
        return beatTask.result.stats;
    }
    
    return stats;
}

// ... rest of the existing code ...